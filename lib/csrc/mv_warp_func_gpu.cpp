#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

torch::Tensor mv_warp_cuda(
      torch::Tensor input,
      torch::Tensor L0_mvx,
      torch::Tensor L0_mvy,
      torch::Tensor L1_mvx,
      torch::Tensor L1_mvy,
      torch::Tensor L0_ref,
      torch::Tensor L1_ref, int pos);


// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor mv_warp_forward(
    torch::Tensor input,
    torch::Tensor L0_mvx,
    torch::Tensor L0_mvy,
    torch::Tensor L1_mvx,
    torch::Tensor L1_mvy,
    torch::Tensor L0_ref,
    torch::Tensor L1_ref, int pos) {

  CHECK_INPUT(input);
  CHECK_INPUT(L0_mvx);
  CHECK_INPUT(L0_mvy);
  CHECK_INPUT(L1_mvx);
  CHECK_INPUT(L1_mvy);
  CHECK_INPUT(L0_ref);
  CHECK_INPUT(L1_ref);
 

  return mv_warp_cuda(input, L0_mvx, L0_mvy, L1_mvx, L1_mvy, L0_ref, L1_ref, pos);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &mv_warp_forward, "mv_warp (CUDA)");

}