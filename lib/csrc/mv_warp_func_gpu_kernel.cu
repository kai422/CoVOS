#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>


// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n)                                   \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
         i += blockDim.x * gridDim.x)

// Use 1024 threads per block, which requires cuda sm_2x or above
const int CUDA_NUM_THREADS = 1024;
const int kMaxGridNum = 65535;

// CUDA: number of blocks for threads.
inline int GET_BLOCKS(const int N)
{
    return std::min(kMaxGridNum, (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS);
}

// clamp x to range [a, b]
__device__ float clamp(float x, float a, float b)
{
    return max(a, min(b, x));
}

__device__ int clamp(int x, int a, int b)
{
    return max(a, min(b, x));
}

namespace {

template <typename scalar_t>
__device__ scalar_t get_nearest_value(
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> arr,
    const int ref_index, 
    const int cc,
    const int height,
    const int width,
    float x,
    float y) 
    {

      int x_idx = clamp(int(round(x)), 0, width-1);
      int y_idx = clamp(int(round(y)), 0, height-1);

      scalar_t I = arr[ref_index][cc][y_idx][x_idx];

      return I;
}

template <typename scalar_t>
__global__ void mv_warp_cuda_kernel(
    const int n,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> output,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> L0_mvx,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> L0_mvy,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> L1_mvx,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> L1_mvy,
    const torch::PackedTensorAccessor32<int32_t,2,torch::RestrictPtrTraits> L0_ref,
    const torch::PackedTensorAccessor32<int32_t,2,torch::RestrictPtrTraits> L1_ref,
    const int num_imgs, const int channels, const int height, const int width, const int pos) 
  {
    //[cc][yy][xx]
    CUDA_KERNEL_LOOP(index, n)
    {
        // index of output matrix
        const int cc = index % channels;
        const int yy = (index / channels) % height;
        const int xx = (index / channels / height) % width;

        if(L0_ref[yy][xx]!= 0 && L1_ref[yy][xx]!= 0)
        {
          int L0_ref_index = pos - L0_ref[yy][xx];
          int L1_ref_index = pos + L1_ref[yy][xx];
          //printf("yy:%d, xx:%d, cc:%d, l0_off:%d, l1_off:%d ,l0_ref_index:%d, l1_ref_index:%d \n ",yy, xx, cc, L0_ref[yy][xx], L1_ref[yy][xx], L0_ref_index, L1_ref_index);

          float x0 = xx + L0_mvx[yy][xx];
          float y0 = yy + L0_mvy[yy][xx];
          float x1 = xx + L1_mvx[yy][xx];
          float y1 = yy + L1_mvy[yy][xx];

          output[cc][yy][xx] = get_nearest_value(input, L0_ref_index, cc, height, width, x0, y0)/2 + get_nearest_value(input, L1_ref_index, cc, height, width, x1, y1)/2;
        }

        else if(L0_ref[yy][xx]!=0 && L1_ref[yy][xx]==0)
        {
          int L0_ref_index = pos - L0_ref[yy][xx];

          float x0 = xx + L0_mvx[yy][xx];
          float y0 = yy + L0_mvy[yy][xx];

          output[cc][yy][xx] = get_nearest_value(input, L0_ref_index, cc, height, width, x0, y0); 

        }
        
        else if(L0_ref[yy][xx]==0 && L1_ref[yy][xx]!=0)
        {
          int L1_ref_index = pos + L1_ref[yy][xx];

          float x1 = xx + L1_mvx[yy][xx];
          float y1 = yy + L1_mvy[yy][xx];
          
          output[cc][yy][xx] = get_nearest_value(input, L1_ref_index, cc, height, width, x1, y1); 

        }
        
    }
}
}


torch::Tensor mv_warp_cuda(
      torch::Tensor input,
      torch::Tensor L0_mvx,
      torch::Tensor L0_mvy,
      torch::Tensor L1_mvx,
      torch::Tensor L1_mvy,
      torch::Tensor L0_ref,
      torch::Tensor L1_ref,
      int pos){


  const auto num_imgs = input.size(0);
  const auto channels = input.size(1);
  const auto height = input.size(2);
  const auto width = input.size(3);

  auto output = torch::zeros({channels, height, width}, input.options());
  int num_kernels = channels * height * width;
  AT_DISPATCH_FLOATING_TYPES(output.type(), "mv_warp_cuda", ([&] {
    mv_warp_cuda_kernel<scalar_t><<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
        num_kernels,
        output.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        input.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        L0_mvx.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        L0_mvy.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        L1_mvx.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        L1_mvy.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        L0_ref.packed_accessor32<int32_t,2,torch::RestrictPtrTraits>(),
        L1_ref.packed_accessor32<int32_t,2,torch::RestrictPtrTraits>(),
        num_imgs, channels, height, width, pos);
  }));

  return output;
}

