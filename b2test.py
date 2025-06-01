import os
import sys
import subprocess

from build_utils import get_b2, build_cpp, nvcc_compile_and_link, get_paths

if __name__ == "__main__":
    # get_b2()

    DIR, VCTOOLSINSTALLDIR, CL_PATH, CUDA_PATH, INCLUDE = get_paths()
    #
    # files = ["vectoradd.cu", "vectoradd.h", "cuSigKernel.cu", "cuSigKernel.h"]
    # nvcc_compile_and_link(files, DIR, CL_PATH, CUDA_PATH, INCLUDE)
    #
    # build_cpp(CUDA_PATH)
    # print(os.environ)
    print(VCTOOLSINSTALLDIR)
    VC0 = VCTOOLSINSTALLDIR[:VCTOOLSINSTALLDIR.find(r'\Tools')]
    print(VC0)
    subprocess.run(["C:\\Users\Shmelev\source\\repos\pySigLib\\build_copy1.bat", VC0, VCTOOLSINSTALLDIR])

