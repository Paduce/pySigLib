from tqdm import tqdm
from timing_utils import timepysiglib_kernel, timesigkernel, plot_times

import plotting_params
plotting_params.set_plotting_params(8, 10, 12)

if __name__ == '__main__':

    dyadicOrder = 0
    batchSize = 32
    length = 1000
    N = 10
    device = "cuda"

    dimArr = [i for i in range(10, 1100, 100)]
    sigkerneltime = []
    pysiglibtime = []

    for dimension in tqdm(dimArr):
        sigkerneltime.append(timesigkernel(batchSize, length, dimension, dyadicOrder, device, N))
        pysiglibtime.append(timepysiglib_kernel(batchSize, length, dimension, dyadicOrder, device, N))

    print(sigkerneltime)
    print(pysiglibtime)

    for scale in ["linear", "log"]:
        plot_times(
                X = dimArr,
                Ys = [sigkerneltime, pysiglibtime],
                legend = ["sigkernel", "pysiglib"],
                title = "Signature Kernels " + device,
                xlabel = "Path Dimension",
                ylabel = "Elapsed Time (s)",
                scale = scale,
                filename = "sigkernel_times_dim_" + scale + "_" + device
        )