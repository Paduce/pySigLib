from tqdm import tqdm
from timing_utils import timesigkernel, timepysiglib_kernel, plot_times

import plotting_params
plotting_params.set_plotting_params(8, 10, 12)

if __name__ == '__main__':

    dyadicOrder = 0
    batchSize = 120
    dimension = 5
    N = 10
    device = "cuda"

    lengthArr = [i for i in range(10, 2100, 100)]
    sigkerneltime = []
    pysiglibtime = []


    for length in tqdm(lengthArr):
        sigkerneltime.append(timesigkernel(batchSize, length, dimension, dyadicOrder, device, N))
        pysiglibtime.append(timepysiglib_kernel(batchSize, length, dimension, dyadicOrder, device, N))

    print(sigkerneltime)
    print(pysiglibtime)

    for scale in ["linear", "log"]:
        plot_times(
                X = lengthArr[:9],
                Ys = [sigkerneltime[:9], pysiglibtime[:9]],
                legend = ["sigkernel", "pysiglib"],
                title = "Times " + device,
                xlabel = "Path Length",
                ylabel = "Elapsed Time (s)",
                scale = scale,
                filename = "sigkernel_times_len_" + scale + "_" + device + "_1"
        )
        plot_times(
                X = lengthArr,
                Ys = [sigkerneltime, pysiglibtime],
                legend = ["sigkernel", "pysiglib"],
                title = "Times " + device,
                xlabel = "Path Length",
                ylabel = "Elapsed Time (s)",
                scale = scale,
                filename = "sigkernel_times_len_" + scale + "_" + device + "_2"
        )
