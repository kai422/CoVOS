from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='mv_warp_gpu',
    version='1.0',
    description='Motion Vector Warping Function with CUDA',
    author='Kai Xu',
    url='https://github.com/kai422/CoVOS',
    ext_modules=[
        CUDAExtension('mv_warp_func_gpu', [
            'lib/csrc/mv_warp_func_gpu.cpp',
            'lib/csrc/mv_warp_func_gpu_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })