from tqdm import tqdm

from timing_utils import timeiisig, timepysiglib, plot_times

import plotting_params
plotting_params.set_plotting_params(8, 10, 12)

if __name__ == '__main__':

    batchSize = 32
    length = 128
    dimension = 6
    N = 10
    device = "cpu"

    degreeArr = [i for i in range(1, 8)]
    iisigtime = []
    #signatorytime = []
    #esigtime = []
    mysigtime = []
    mysigtimehorner = []
    mysigtimehornerparallel = []

    for degree in tqdm(degreeArr):
        iisigtime.append(timeiisig(batchSize, length, dimension, degree, device, N))
        #signatorytime.append(timesignatory(degree))
        #esigtime.append(timeesig(degree))
        mysigtime.append(timepysiglib(batchSize, length, dimension, degree, False, False, device, N))
        mysigtimehorner.append(timepysiglib(batchSize, length, dimension, degree, True, False, device, N))
        mysigtimehornerparallel.append(timepysiglib(batchSize, length, dimension, degree, True, True, device, N))

    print(iisigtime)
    #print(signatorytime)
    #print(esigtime)
    print(mysigtime)
    print(mysigtimehorner)
    print(mysigtimehornerparallel)

    for scale in ["linear", "log"]:
        plot_times(
            X = degreeArr,
            Ys = [iisigtime, mysigtime, mysigtimehorner, mysigtimehornerparallel],
            legend = ["iisignature", "pySigLib (Naive)", "pySigLib (Horner)", "pySigLib (Horner, Parallel)"],
            title = "Signature " + device,
            xlabel = "Truncation Level",
            ylabel = "Elapsed Time (s)",
            scale = scale,
            filename = "signature_times_" + scale + "_" + device
        )
