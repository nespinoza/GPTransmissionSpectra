import numpy as np
import argparse
import utils
import pickle
import os

parser = argparse.ArgumentParser()

# This parses in the option file:
parser.add_argument("-ofile", default=None)
parser.add_argument("--nopickle", dest="nopickle", action="store_true")
parser.set_defaults(nopickle=False)
args = parser.parse_args()
ofile = args.ofile
nopickle = args.nopickle

# Read input file:
(
    datafile,
    ld_law,
    idx_time,
    all_comps,
    P,
    Psd,
    a,
    asd,
    pmean,
    psd,
    b,
    bsd,
    t0,
    t0sd,
    fixed_eccentricity,
    ecc,
    eccsd,
    omega,
    omegasd,
) = utils.read_optfile(ofile)

######################################
target, pfilename = datafile.split("/")
out_folder = "outputs/" + datafile.split(".")[0] + "/wavelength"
out_ofolder = "outputs/" + datafile.split(".")[0]
if not os.path.exists(out_folder):
    os.mkdir(out_folder)
if not nopickle:
    data = pickle.load(open(datafile, "rb"))
else:
    data = {}
    t, m1lc = np.loadtxt(
        "outputs/" + datafile.split(".")[0] + "/white-light/lc.dat",
        unpack=True,
        usecols=(0, 1),
    )
    data["t"] = t
    import glob

    binfolders = glob.glob(out_folder + "/*")
    data["wbins"] = np.arange(len(binfolders))
    data["oLCw"] = np.random.uniform(1, 10, [3, len(binfolders)])
# Generate idx_time, number of bins:
exec('idx_time = np.arange(len(data["t"]))' + idx_time)
nwbins = len(data["wbins"])
for wi in range(nwbins):
    if (
        np.mean(data["oLCw"][:, wi]) != 0.0
        and len(np.where(data["oLCw"][:, wi] < 0)[0]) < 1
        and not nopickle
    ):
        # 0. Chech which comparisons are non-zero in this wavelength bin:
        comps = []
        for i in range(len(all_comps)):
            if np.mean(data["cLCw"][:, all_comps[i], wi]) != 0.0:
                comps.append(all_comps[i])

        # 1. Save (mean-substracted) target and comparison lightcurves (in magnitude-space):
        if not os.path.exists(out_folder + "/wbin" + str(wi)):
            os.mkdir(out_folder + "/wbin" + str(wi))
            lcout = open(out_folder + "/wbin" + str(wi) + "/lc.dat", "w")
            lccompout = open(
                out_folder + "/wbin" + str(wi) + "/comps.dat", "w"
            )
            for i in idx_time:
                lcout.write(
                    "{0:.10f} {1:.10f} 0\n".format(
                        data["t"][i],
                        -2.51 * np.log10(data["oLCw"][i, wi])
                        - np.median(
                            -2.51 * np.log10(data["oLCw"][idx_time, wi])
                        ),
                    )
                )
                for j in range(len(comps)):
                    if j != len(comps) - 1:
                        lccompout.write(
                            "{0:.10f} \t".format(
                                -2.51 * np.log10(data["cLCw"][i, comps[j], wi])
                                - np.median(
                                    -2.51
                                    * np.log10(
                                        data["cLCw"][idx_time, comps[j], wi]
                                    )
                                )
                            )
                        )
                    else:
                        lccompout.write(
                            "{0:.10f}\n".format(
                                -2.51 * np.log10(data["cLCw"][i, comps[j], wi])
                                - np.median(
                                    -2.51
                                    * np.log10(
                                        data["cLCw"][idx_time, comps[j], wi]
                                    )
                                )
                            )
                        )
            lcout.close()
            lccompout.close()

for wi in range(nwbins):
    print("Working on wbin ", wi, "...")
    if (
        np.mean(data["oLCw"][:, wi]) != 0.0
        and len(np.where(data["oLCw"][:, wi] < 0)[0]) < 1
        and not nopickle
    ):
        # 1.5 Count the comps:
        comps = []
        for i in range(len(all_comps)):
            if np.mean(data["cLCw"][:, all_comps[i], wi]) != 0.0:
                comps.append(all_comps[i])
    else:
        comps = all_comps
    if (
        np.mean(data["oLCw"][:, wi]) != 0.0
        and len(np.where(data["oLCw"][:, wi] < 0)[0]) < 1
    ):
        # 2. Run code, BMA the posteriors, save:
        if not os.path.exists(
            out_folder + "/wbin" + str(wi) + "/BMA_posteriors.pkl"
        ):
            lnZ = np.zeros(len(comps))
            nmin = np.inf
            for i in range(1, len(comps) + 1):
                os.system(
                    "python GPTransitDetrendWavelength.py -outfolder "
                    + out_folder
                    + "/wbin"
                    + str(wi)
                    + "/ -compfile "
                    + out_folder
                    + "/wbin"
                    + str(wi)
                    + "/comps.dat -lcfile "
                    + out_folder
                    + "/wbin"
                    + str(wi)
                    + "/lc.dat -eparamfile "
                    + out_ofolder
                    + "/eparams.dat -ldlaw "
                    + ld_law
                    + " -P "
                    + str(P)
                    + " -a "
                    + str(a)
                    + " -pmean "
                    + str(pmean)
                    + " -psd "
                    + str(psd)
                    + " -b "
                    + str(b)
                    + " -t0 "
                    + str(t0)
                    + " -ecc "
                    + str(ecc)
                    + " -omega "
                    + str(omega)
                    + " --PCA -pctouse "
                    + str(i)
                )
                if not os.path.exists(
                    out_folder + "/wbin" + str(wi) + "/PCA_" + str(i)
                ):
                    os.mkdir(out_folder + "/wbin" + str(wi) + "/PCA_" + str(i))
                    os.system(
                        "mv "
                        + out_folder
                        + "/wbin"
                        + str(wi)
                        + "/out* "
                        + out_folder
                        + "/wbin"
                        + str(wi)
                        + "/PCA_"
                        + str(i)
                        + "/."
                    )
                os.system(
                    "mv "
                    + out_folder
                    + "/wbin"
                    + str(wi)
                    + "/*.pkl "
                    + out_folder
                    + "/wbin"
                    + str(wi)
                    + "/PCA_"
                    + str(i)
                    + "/."
                )
                os.system(
                    "mv detrended_lc.dat "
                    + out_folder
                    + "/wbin"
                    + str(wi)
                    + "/PCA_"
                    + str(i)
                    + "/."
                )
                os.system(
                    "mv model_lc.dat "
                    + out_folder
                    + "/wbin"
                    + str(wi)
                    + "/PCA_"
                    + str(i)
                    + "/."
                )
                fin = open(
                    out_folder
                    + "/wbin"
                    + str(wi)
                    + "/PCA_"
                    + str(i)
                    + "/posteriors_trend_george.pkl",
                    "rb",
                )
                posteriors = pickle.load(fin)
                if len(posteriors["posterior_samples"]["p"]) < nmin:
                    nmin = len(posteriors["posterior_samples"]["p"])
                lnZ[i - 1] = posteriors["lnZ"]
                fin.close()
            # Calculate posterior probabilities of the models from the Bayes Factors:
            lnZ = lnZ - np.max(lnZ)
            Z = np.exp(lnZ)
            Pmodels = Z / np.sum(Z)
            # Prepare array that saves outputs:
            p = np.array([])
            q1 = np.array([])
            if ld_law != "linear":
                q2 = np.array([])
            jitter = np.array([])
            max_GPvariance = np.array([])
            # Check how many alphas were fitted:
            acounter = 0
            for vrs in list(posteriors["posterior_samples"].keys()):
                if "alpha" in vrs:
                    exec("alpha" + str(acounter) + " = np.array([])")
                    acounter = acounter + 1
            mmean = np.array([])
            # With the number at hand, extract draws from the  posteriors with a fraction equal to the posterior probabilities to perform the
            # model averaging scheme:
            for i in range(1, len(comps) + 1):
                fin = open(
                    out_folder
                    + "/wbin"
                    + str(wi)
                    + "/PCA_"
                    + str(i)
                    + "/posteriors_trend_george.pkl",
                    "rb",
                )
                posteriors = pickle.load(fin)
                fin.close()
                nextract = int(Pmodels[i - 1] * nmin)
                idx_extract = np.random.choice(
                    np.arange(len(posteriors["posterior_samples"]["p"])),
                    nextract,
                    replace=False,
                )
                # Extract transit parameters:
                p = np.append(
                    p, posteriors["posterior_samples"]["p"][idx_extract]
                )
                q1 = np.append(
                    q1, posteriors["posterior_samples"]["q1"][idx_extract]
                )
                if ld_law != "linear":
                    q2 = np.append(
                        q2, posteriors["posterior_samples"]["q2"][idx_extract]
                    )
                # Note bayesian average posterior jitter saved is in mmag (MultiNest+george sample the log-variance, not the log-sigma):
                jitter = np.append(
                    jitter,
                    np.sqrt(
                        np.exp(
                            posteriors["posterior_samples"]["ljitter"][
                                idx_extract
                            ]
                        )
                    ),
                )
                # Mean lightcurve in magnitude units:
                mmean = np.append(
                    mmean,
                    posteriors["posterior_samples"]["mmean"][idx_extract],
                )
                # Max GP variance:
                max_GPvariance = np.append(
                    max_GPvariance,
                    posteriors["posterior_samples"]["max_var"][idx_extract],
                )
                # Alphas:
                for ai in range(acounter):
                    exec(
                        "alpha"
                        + str(ai)
                        + " = np.append(alpha"
                        + str(ai)
                        + ",posteriors['posterior_samples']['alpha"
                        + str(ai)
                        + "'][idx_extract])"
                    )

            # Now save final BMA posteriors:
            out = {}
            out["p"] = p
            out["wbin"] = data["wbins"][wi]
            out["jitter"] = jitter
            out["q1"] = q1
            if ld_law != "linear":
                out["q2"] = q2
            out["mmean"] = mmean
            out["max_var"] = max_GPvariance
            for ai in range(acounter):
                exec("out['alpha" + str(ai) + "'] = alpha" + str(ai))
            pickle.dump(
                out,
                open(
                    out_folder + "/wbin" + str(wi) + "/BMA_posteriors.pkl",
                    "wb",
                ),
            )
            fout = open(out_folder + "/wbin" + str(wi) + "/results.dat", "w")
            fout.write("# Variable \t Value \t SigmaUp \t SigmaDown\n")
            for variable in list(out.keys()):
                if variable != "wbin":
                    v, vup, vdown = utils.get_quantiles(out[variable])
                    fout.write(
                        variable
                        + " \t {0:.10f} \t {1:.10f} \t {2:.10f}\n".format(
                            v, vup - v, v - vdown
                        )
                    )
            fout.close()
        else:
            out = pickle.load(
                open(
                    out_folder + "/wbin" + str(wi) + "/BMA_posteriors.pkl",
                    "rb",
                )
            )
