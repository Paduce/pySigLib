from tqdm import tqdm

from timing_utils import timeiisig, timepysiglib, timesignatory, plot_times

import plotting_params
plotting_params.set_plotting_params(8, 10, 12)

# pip install signatory==1.2.6.1.9.0 --no-cache-dir --force-reinstall

if __name__ == '__main__':

    batchSize = 32
    length = 1024
    dimension = 5
    N = 5
    device = "cpu"

    degreeArr = [i for i in range(1, 8)]
    iisigtime = []
    signatorytime = []
    pysiglibtime = []
    pysiglibtimeparallel = []
    pysiglibtimehorner = []
    pysiglibtimehornerparallel = []

    for degree in tqdm(degreeArr):
        iisigtime.append(timeiisig(batchSize, length, dimension, degree, device, N))
        signatorytime.append(timesignatory(batchSize, length, dimension, degree, device, N))
        pysiglibtime.append(timepysiglib(batchSize, length, dimension, degree, False, False, device, N))
        pysiglibtimeparallel.append(timepysiglib(batchSize, length, dimension, degree, False, True, device, N))
        pysiglibtimehorner.append(timepysiglib(batchSize, length, dimension, degree, True, False, device, N))
        pysiglibtimehornerparallel.append(timepysiglib(batchSize, length, dimension, degree, True, True, device, N))

    print(iisigtime)
    print(signatorytime)
    print(pysiglibtime)
    print(pysiglibtimeparallel)
    print(pysiglibtimehorner)
    print(pysiglibtimehornerparallel)

    for scale in ["linear", "log"]:
        plot_times(
            X = degreeArr,
            Ys = [iisigtime, pysiglibtime, pysiglibtimehorner],
            legend = ["iisignature (Direct)", "pySigLib (Direct)", "pySigLib (Horner)"],
            title = "Truncated Signatures (Serial)",
            xlabel = "Truncation Level",
            ylabel = "Elapsed Time (s)",
            scale = scale,
            filename = "signature_times_" + scale + "_serial"
        )

        plot_times(
            X=degreeArr,
            Ys=[signatorytime, pysiglibtimeparallel, pysiglibtimehornerparallel],
            legend=["signatory (Horner)", "pySigLib (Direct)", "pySigLib (Horner)"],
            title= "Truncated Signatures (Parallel)",
            xlabel="Truncation Level",
            ylabel="Elapsed Time (s)",
            scale=scale,
            filename="signature_times_" + scale + "_parallel"
        )
