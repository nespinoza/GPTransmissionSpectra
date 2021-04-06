import numpy as np
import argparse
import utils
import pickle
import os
import importlib
import shutil
from pathlib import Path
from astropy.constants import G as const_G

# Define constants on the code:
G = const_G.cgs.value  # Gravitational constant, cgs

parser = argparse.ArgumentParser()

# This parses in the option file:
parser.add_argument("-ofile", default=None)
args = parser.parse_args()
ofile = args.ofile

# Read input file:
c = importlib.import_module(ofile)
out_folder_base = c.out_folder_base
datafile = c.datafile
ld_law = c.ld_law
comps = c.comps
Pmean, Psd = c.Pmean, c.Psd
if hasattr(c, "amean"):
    use_rho_star = False
    amean, asd = c.amean, c.asd
else:
    use_rho_star = True
    rhomean, rhosd = c.rhomean, c.rhosd
if hasattr(c, "bmean"):
    use_r1_r2 = False
    bmean, bsd = c.bmean, c.bsd
    pmean, psd = c.pmean, c.psd
else:
    use_r1_r2 = True
    pl, pu = c.pl, c.pu
    Ar = (pu - pl) / (2.0 + pl + pu)
t0mean, t0sd = c.t0mean, c.t0sd
fixed_eccentricity = c.fixed_eccentricity
eccmean, eccsd = c.eccmean, c.eccsd
omegamean, omegasd = c.omegamean, c.omegasd
PCA = c.PCA
GPkernel = c.GPkernel
nopickle = c.nopickle
nlive = c.nlive
print("Loaded options for:", datafile)

######################################
target, _ = datafile.split("/")
out_folder = f"{out_folder_base}/{datafile.split('.')[0]}"

if not os.path.exists(out_folder):
    Path(out_folder).mkdir(parents=True)
    data = pickle.load(open(datafile, "rb"))
    # Generate input idx_time:
    if hasattr(c, "idx_time") and hasattr(c, "bad_idx_time"):
        raise ValueError("Only idx_time or bad_idx_time can be specified")
    elif hasattr(c, "idx_time") and not hasattr(c, "bad_idx_time"):
        exec('idx_time = np.arange(len(data["t"]))' + c.idx_time)
    else:
        if c.bad_idx_time == "None" or c.bad_idx_time == "[]":
            bad_idx_time = []
        else:
            bad_idx_time = utils._bad_idxs(c.bad_idx_time)
        idx_time = np.delete(np.arange(len(data["t"])), bad_idx_time)
    if not os.path.exists(out_folder + "/white-light"):
        os.mkdir(out_folder + "/white-light")
        # 0. Save wl_options.dat file:
        shutil.copy2(ofile + ".py", out_folder + "/white-light")
        # 1. Save external parameters:
        out_eparam = open(out_folder + "/eparams.dat", "w")
        # Get median of FWHM, background flux, accross all wavelengths, and trace position of zero point.
        # First, find chips-names of target:
        names = []
        for name in list(data["fwhm"].keys()):
            if target in name:
                names.append(name)
        if len(names) == 1:
            Xfwhm = data["fwhm"][names[0]]
            Xsky = data["sky"][names[0]]
        else:
            Xfwhm = np.hstack((data["fwhm"][names[0]], data["fwhm"][names[1]]))
            Xsky = np.hstack((data["sky"][names[0]], data["sky"][names[1]]))
        fwhm = np.zeros(Xfwhm.shape[0])
        sky = np.zeros(Xfwhm.shape[0])
        trace = np.zeros(Xfwhm.shape[0])
        for i in range(len(fwhm)):
            idx = np.where(Xfwhm[i, :] != 0)[0]
            fwhm[i] = np.median(Xfwhm[i, idx])
            idx = np.where(Xsky[i, :] != 0)[0]
            sky[i] = np.median(Xsky[i, idx])
            trace[i] = np.polyval(
                data["traces"][target][i], Xfwhm.shape[1] / 2
            )
        print("Saving eparams...")
        # Save external parameters:
        out_eparam.write(
            "#Times \t                 Airmass \t Delta Wav \t FWHM \t        Sky Flux \t      Trace Center \n"
        )
        for i in idx_time:
            out_eparam.write(
                "{0:.10f} \t {1:.10f} \t {2:.10f} \t {3:.10f} \t {4:.10f} \t {5:.10f} \n".format(
                    data["t"][i],
                    data["Z"][i],
                    data["deltas"][target + "_final"][i],
                    fwhm[i],
                    sky[i],
                    trace[i],
                )
            )
        out_eparam.close()

        # 2. Save (mean-substracted) target and comparison lightcurves (in magnitude-space):
        lcout = open(out_folder + "/white-light/lc.dat", "w")
        lccompout = open(out_folder + "/white-light/comps.dat", "w")
        for i in idx_time:
            lcout.write(
                "{0:.10f} {1:.10f} 0\n".format(
                    data["t"][i],
                    -2.51 * np.log10(data["oLC"][i])
                    - np.median(-2.51 * np.log10(data["oLC"][idx_time])),
                )
            )
            for j in range(len(comps)):
                if j != len(comps) - 1:
                    lccompout.write(
                        "{0:.10f} \t".format(
                            -2.51 * np.log10(data["cLC"][i, comps[j]])
                            - np.median(
                                -2.51
                                * np.log10(data["cLC"][idx_time, comps[j]])
                            )
                        )
                    )
                else:
                    lccompout.write(
                        "{0:.10f}\n".format(
                            -2.51 * np.log10(data["cLC"][i, comps[j]])
                            - np.median(
                                -2.51
                                * np.log10(data["cLC"][idx_time, comps[j]])
                            )
                        )
                    )
        lcout.close()
        lccompout.close()

# 3. If not already done, run code for all PCAs, extract best-fit parameters, model-average them, save them. For this,
# first check maximum number of samples sampled from posterior from all the fits:

if not os.path.exists(out_folder + "/white-light/BMA_posteriors.pkl"):
    lnZ = np.zeros(len(comps))
    nmin = np.inf
    if fixed_eccentricity:
        print("Fixing eccentricity in the fit...")
    for i in range(1, len(comps) + 1):
        # Load inputs
        os_string = (
            "python GPTransitDetrendWL.py"
            + f" -nlive {nlive}"
            + f" -outfolder {out_folder}/white-light/"
            + f" -compfile {out_folder}/white-light/comps.dat"
            + f" -lcfile {out_folder}/white-light/lc.dat"
            + f" -eparamfile {out_folder}/eparams.dat"
            + f" -ldlaw {ld_law}"
            + f" -Pmean {Pmean}"
            + f" -Psd {Psd}"
            + f" -t0mean {t0mean}"
            + f" -t0sd {t0sd}"
            + f" -eccmean {eccmean}"
            + f" -eccsd {eccsd}"
            + f" -omegamean {omegamean}"
            + f" -omegasd {omegasd}"
            + f" -PCA {PCA}"
            + f" -GPkernel {GPkernel}"
            + f" -fixed_eccentricity {fixed_eccentricity}"
            + f" -pctouse {i}"
        )
        if use_rho_star:
            os_string += f" -rhomean {rhomean}"
            os_string += f" -rhosd {rhosd}"
        else:
            os_string += f" -amean {amean}"
            os_string += f" -asd {asd}"
        if use_r1_r2:
            os_string += f" -pl {pl}"
            os_string += f" -pu {pu}"
        else:
            os_string += f" -pmean {pmean}"
            os_string += f" -psd {psd}"
            os_string += f" -bmean {bmean}"
            os_string += f" -bsd {bsd}"

        # Run sampler
        os.system(os_string)

        # Save output
        if not os.path.exists(out_folder + "/white-light/PCA_" + str(i)):
            os.mkdir(out_folder + "/white-light/PCA_" + str(i))
            os.system(
                "mv "
                + out_folder
                + "/white-light/out* "
                + out_folder
                + "/white-light/PCA_"
                + str(i)
                + "/."
            )

        os.system(
            "mv "
            + out_folder
            + "/white-light/*.pkl "
            + out_folder
            + "/white-light/PCA_"
            + str(i)
            + "/."
        )
        os.system(
            "mv detrended_lc.dat "
            + out_folder
            + "/white-light/PCA_"
            + str(i)
            + "/."
        )
        os.system(
            "mv model_lc.dat "
            + out_folder
            + "/white-light/PCA_"
            + str(i)
            + "/."
        )

        fin = open(
            out_folder
            + "/white-light/PCA_"
            + str(i)
            + "/posteriors_trend_george.pkl",
            "rb",
        )
        posteriors = pickle.load(fin)
        if len(posteriors["posterior_samples"]["q1"]) < nmin:
            nmin = len(posteriors["posterior_samples"]["q1"])
        lnZ[i - 1] = posteriors["lnZ"]
        fin.close()
    # Calculate posterior probabilities of the models from the Bayes Factors:
    lnZ = lnZ - np.max(lnZ)
    Z = np.exp(lnZ)
    Pmodels = Z / np.sum(Z)
    # Prepare array that saves outputs:
    periods = np.array([])
    if use_rho_star:
        rhos = np.array([])
    else:
        aRs = np.array([])
    if use_r1_r2:
        r1s, r2s = np.array([]), np.array([])
    else:
        b, p = np.array([]), np.array([])
    t0 = np.array([])
    ecc = np.array([])
    omega = np.array([])
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
            + "/white-light/PCA_"
            + str(i)
            + "/posteriors_trend_george.pkl",
            "rb",
        )
        posteriors = pickle.load(fin)
        fin.close()
        # Check how many comp star coefficents were fitted:
        xccounter = 0
        for vrs in list(posteriors["posterior_samples"].keys()):
            if "xc" in vrs:
                if f"xc{xccounter}" not in locals():
                    exec(
                        "xc"
                        + str(xccounter)
                        + " = posteriors['posterior_samples']['xc"
                        + str(xccounter)
                        + "']"
                    )
                xccounter = xccounter + 1

        nextract = int(Pmodels[i - 1] * nmin)
        idx_extract = np.random.choice(
            np.arange(len(posteriors["posterior_samples"]["P"])),
            nextract,
            replace=False,
        )

        # Extract transit parameters:
        periods = np.append(
            periods, posteriors["posterior_samples"]["P"][idx_extract]
        )
        if use_rho_star:
            rhos = np.append(
                rhos, posteriors["posterior_samples"]["rho"][idx_extract]
            )
        else:
            aRs = np.append(
                aRs, posteriors["posterior_samples"]["a"][idx_extract]
            )
        if use_r1_r2:
            r1s = np.append(
                r1s, posteriors["posterior_samples"]["r1"][idx_extract]
            )
            r2s = np.append(
                r2s, posteriors["posterior_samples"]["r2"][idx_extract]
            )
        else:
            p = np.append(p, posteriors["posterior_samples"]["p"][idx_extract])
            b = np.append(b, posteriors["posterior_samples"]["b"][idx_extract])
        t0 = np.append(t0, posteriors["posterior_samples"]["t0"][idx_extract])
        if not fixed_eccentricity:
            ecc = np.append(
                ecc, posteriors["posterior_samples"]["ecc"][idx_extract]
            )
            omega = np.append(
                omega, posteriors["posterior_samples"]["omega"][idx_extract]
            )
        q1 = np.append(q1, posteriors["posterior_samples"]["q1"][idx_extract])
        if ld_law != "linear":
            q2 = np.append(
                q2, posteriors["posterior_samples"]["q2"][idx_extract]
            )
        # Note bayesian average posterior jitter saved is in mmag (MultiNest+george sample the log-variance, not the log-sigma):
        jitter = np.append(
            jitter,
            np.sqrt(
                np.exp(posteriors["posterior_samples"]["ljitter"][idx_extract])
            ),
        )
        # Mean lightcurve in magnitude units:
        mmean = np.append(
            mmean, posteriors["posterior_samples"]["mmean"][idx_extract]
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

        # Comp star coefficients
        for xci in range(xccounter):
            exec(
                "xc"
                + str(xci)
                + " = np.append(xc"
                + str(xci)
                + ",posteriors['posterior_samples']['xc"
                + str(xci)
                + "'][idx_extract])"
            )

    # Now save final BMA posteriors:
    out = {}
    out["P"] = periods
    if use_rho_star:
        out["rho"] = rhos
        aRs = (
            (rhos * G * ((periods * 24.0 * 3600.0) ** 2)) / (3.0 * np.pi)
        ) ** (1.0 / 3.0)
    if use_r1_r2:
        b = np.zeros_like(r1s)
        p = np.zeros_like(r1s)
        gtAr = r1s > Ar
        if sum(gtAr) != 0:
            b[gtAr], p[gtAr] = (
                (1 + pl) * (1.0 + (r1s[gtAr] - 1.0) / (1.0 - Ar)),
                (1 - r2s[gtAr]) * pl + r2s[gtAr] * pu,
            )
        if sum(~gtAr) != 0:
            b[~gtAr], p[~gtAr] = (
                (1.0 + pl) + np.sqrt(r1s[~gtAr] / Ar) * r2s[~gtAr] * (pu - pl),
                pu + (pl - pu) * np.sqrt(r1s[~gtAr] / Ar) * (1.0 - r2s[~gtAr]),
            )
        out["r1"] = r1s
        out["r2"] = r2s
    out["b"] = b
    out["p"] = p
    out["aR"] = aRs
    out["inc"] = np.arccos(b / aRs) * 180.0 / np.pi
    out["t0"] = t0
    if not fixed_eccentricity:
        out["ecc"] = ecc
        out["omega"] = omega
    out["jitter"] = jitter
    out["q1"] = q1
    if ld_law != "linear":
        out["q2"] = q2
    out["mmean"] = mmean
    out["max_var"] = max_GPvariance
    for ai in range(acounter):
        exec("out['alpha" + str(ai) + "'] = alpha" + str(ai))
    for xci in range(xccounter):
        exec("out['xc" + str(xci) + "'] = xc" + str(xci))
    pickle.dump(
        out, open(out_folder + "/white-light/BMA_posteriors.pkl", "wb")
    )
    fout = open(out_folder + "/white-light/results.dat", "w")
    fout.write("# Variable \t Value \t SigmaUp \t SigmaDown\n")
    for variable in list(out.keys()):
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
        open(out_folder + "/white-light/BMA_posteriors.pkl", "rb")
    )
