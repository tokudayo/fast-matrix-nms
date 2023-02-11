from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="matrix_nms",
    ext_modules=[
        CUDAExtension(
            "matrix_nms",
            [
                "matrix_nms.cpp",
                "matrix_nms_cuda.cu",
            ],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
