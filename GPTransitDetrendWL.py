import batman
import argparse
import matplotlib
import matplotlib.pyplot as plt
import pymultinest
from scipy import interpolate
import numpy as np
import utils
import os
import sys
import pprint
from astropy.constants import G as const_G

# Define constants on the code:
G = const_G.cgs.value  # Gravitational constant, cgs

parser = argparse.ArgumentParser()

# This reads the output folder:
parser.add_argument("-outfolder", default=None)
# This reads the lightcurve file. First column is time, second column is flux:
parser.add_argument("-lcfile", default=None)
# This reads the external parameters to fit (assumed to go in the columns):
parser.add_argument("-eparamfile", default=None)
# This defines which of the external parameters you want to use, separated by commas.
# Default is all:
parser.add_argument("-eparamtouse", default="all")
# This reads the external parameters to fit (assumed to go in the columns):
parser.add_argument("-compfile", default=None)
# This defines which comparison stars, if any, you want to use, separated by commas.
# Default is all:
parser.add_argument("-comptouse", default="all")
# If PCA, define number of PCs to use:
parser.add_argument("-pctouse", default="all")
# This defines the limb-darkening to be used:
parser.add_argument("-ldlaw", default="quadratic")

# Transit priors. First t0:
parser.add_argument("-t0mean", default=None)
# This reads the standard deviation:
parser.add_argument("-t0sd", default=None)

# Period:
parser.add_argument("-Pmean", default=None)
# This reads the standard deviation:
parser.add_argument("-Psd", default=None)

# Rp/Rs:
parser.add_argument("-pmean", default=None)
# This reads the standard deviation:
parser.add_argument("-psd", default=None)

# a/Rs:
parser.add_argument("-amean", default=None)
# This reads the standard deviation:
parser.add_argument("-asd", default=None)

# rho:
parser.add_argument("-rhomean", default=None)
# This reads the standard deviation:
parser.add_argument("-rhosd", default=None)

# Impact parameter:
parser.add_argument("-bmean", default=None)
# This reads the standard deviation:
parser.add_argument("-bsd", default=None)

# pl, pu for b, p re-parameterization
parser.add_argument("-pl", default=None)
parser.add_argument("-pu", default=None)

# ecc:
parser.add_argument("-eccmean", default=None)
# This reads the standard deviation:
parser.add_argument("-eccsd", default=None)

# omega:
parser.add_argument("-omegamean", default=None)

# This reads the standard deviation:
parser.add_argument("-omegasd", default=None)

# Define if it is a fixed_eccentricity fit.
# In this case, ecc = eccmean, omega = omegamean
# (e.g., if circular, let eccmean = 0, omegamean = 90)
parser.add_argument("-fixed_eccentricity", default="True")

# Define if PCA will be used instead of using comparison stars directly:
parser.add_argument("-PCA", default="True")

# Define kernel. If true, multi-dimensional Matern:
parser.add_argument("-GPkernel", default="multi_sqexp")

# Supported kernels
KERNELS = ["multi_sqexp", "multi_matern"]

# Number of live points:
parser.add_argument("-nlive", default=1000)
args = parser.parse_args()
pp = pprint.PrettyPrinter(compact=True)
print("Using following inputs:")
pp.pprint(vars(args))


def get_quantiles(dist, alpha=0.68, method="median"):
    """
    get_quantiles function

    DESCRIPTION

        This function returns, in the default case, the parameter median and the error%
        credibility around it. This assumes you give a non-ordered
        distribution of parameters.

    OUTPUTS

        Median of the parameter,upper credibility bound, lower credibility bound

    """
    ordered_dist = dist[np.argsort(dist)]
    param = 0.0
    # Define the number of samples from posterior
    nsamples = len(dist)
    nsamples_at_each_side = int(nsamples * (alpha / 2.0) + 1)
    if method == "median":
        med_idx = 0
        if nsamples % 2 == 0.0:  # Number of points is even
            med_idx_up = int(nsamples / 2.0) + 1
            med_idx_down = med_idx_up - 1
            param = (
                ordered_dist[med_idx_up] + ordered_dist[med_idx_down]
            ) / 2.0
            return (
                param,
                ordered_dist[med_idx_up + nsamples_at_each_side],
                ordered_dist[med_idx_down - nsamples_at_each_side],
            )
        else:
            med_idx = int(nsamples / 2.0)
            param = ordered_dist[med_idx]
            return (
                param,
                ordered_dist[med_idx + nsamples_at_each_side],
                ordered_dist[med_idx - nsamples_at_each_side],
            )


# Is it a fixed_eccentricity fit?
fixed_eccentricity = args.fixed_eccentricity
# Kernel?
GPkernel = args.GPkernel
# Are we going to use PCA?
PCA = args.PCA

# Define output folder:
out_folder = args.outfolder
# Extract lightcurve and external parameters. When importing external parameters,
# standarize them and save them on the matrix X:
lcfilename = args.lcfile
tall, fall, f_index = np.genfromtxt(lcfilename, delimiter=',', unpack=True, usecols=(0, 1, 2))
# Float the times (batman doesn't like non-float 64):
tall = tall.astype("float64")

idx = np.where(f_index == 0)[0]
t, f = tall[idx], fall[idx]

eparamfilename = args.eparamfile
eparams = args.eparamtouse
data = np.genfromtxt(eparamfilename, delimiter=',', skip_header=1, unpack=True)
for i in range(len(data)):
    x = (data[i] - np.mean(data[i])) / np.sqrt(np.var(data[i]))
    if i == 0:
        X = x
    else:
        X = np.vstack((X, x))
if eparams != "all":
    idx_params = np.array(eparams.split(",")).astype("int")
    X = X[idx_params, :]

compfilename = args.compfile
if compfilename is not None:
    comps = args.comptouse
    data = np.genfromtxt(compfilename, delimiter=',', unpack=True)
    if len(data.shape) == 2:
        for i in range(len(data)):
            x = (data[i] - np.mean(data[i])) / np.sqrt(np.var(data[i]))
            if i == 0:
                Xc = x
            else:
                Xc = np.vstack((Xc, x))
        if comps != "all":
            idx_params = np.array(comps.split(",")).astype("int")
            Xc = Xc[idx_params, :]
    else:
        Xc = np.zeros([1, len(data)])
        Xc[0, :] = data

# Extract limb-darkening law:
ld_law = args.ldlaw

# Transit parameter priors if any:
t0mean = args.t0mean
if t0mean is not None:
    t0mean = np.double(t0mean)
    t0sd = np.double(args.t0sd)

Pmean = args.Pmean
if Pmean is not None:
    Pmean = np.double(Pmean)
    Psd = np.double(args.Psd)

if (args.bmean is not None) and (args.pmean is not None):
    use_r1_r2 = False
    bmean = args.bmean
    bmean = np.double(bmean)
    bsd = np.double(args.bsd)

    pmean = args.pmean
    pmean = np.double(pmean)
    psd = np.double(args.psd)
else:
    use_r1_r2 = True
    pl, pu = args.pl, args.pu
    pl, pu = np.double(pl), np.double(pu)
    Ar = (pu - pl) / (2.0 + pl + pu)

if args.amean is not None:
    use_rho_star = False
    amean = args.amean
    amean = np.double(amean)
    asd = np.double(args.asd)
else:
    use_rho_star = True
    rhomean = args.rhomean
    rhomean = np.double(rhomean)
    rhosd = np.double(args.rhosd)

if fixed_eccentricity == "False":
    eccmean = args.eccmean
    omegamean = args.omegamean
    if eccmean is not None:
        eccmean = np.double(args.eccmean)
        eccsd = np.double(args.eccsd)
    if omegamean is not None:
        omegamean = np.double(args.omegamean)
        omegasd = np.double(args.omegasd)
else:
    eccmean = np.double(args.eccmean)
    omegamean = np.double(args.omegamean)
    print("Fixed eccentricity and omega:", eccmean, omegamean)
# Other inputs:
n_live_points = int(args.nlive)

# Cook the george kernel:
import george

if GPkernel == "multi_matern":
    kernel = np.var(f) * george.kernels.Matern32Kernel(
        np.ones(X[:, idx].shape[0]),
        ndim=X[:, idx].shape[0],
        axes=list(range(X[:, idx].shape[0])),
    )
elif GPkernel == "multi_sqexp":
    kernel = np.var(f) * george.kernels.ExpSquaredKernel(
        np.ones(X[:, idx].shape[0]),
        ndim=X[:, idx].shape[0],
        axes=list(range(X[:, idx].shape[0])),
    )
else:
    raise ValueError(
        f"{GPkernel} is not supported. Please choose from either: {KERNELS}"
    )
# Cook jitter term
jitter = george.modeling.ConstantModel(np.log((200.0 * 1e-6) ** 2.0))

# Wrap GP object to compute likelihood
gp = george.GP(
    kernel, mean=0.0, fit_mean=False, white_noise=jitter, fit_white_noise=True
)
gp.compute(X[:, idx].T)

# Extract PCs if user wants to:
if PCA == "True":
    if Xc.shape[0] != 1:
        eigenvectors, eigenvalues, PC = utils.classic_PCA(Xc)
        pctouse = args.pctouse
        if pctouse == "all":
            Xc = PC
        else:
            Xc = PC[: int(pctouse), :]

# Define transit-related functions:
def reverse_ld_coeffs(ld_law, q1, q2):
    if ld_law == "quadratic":
        coeff1 = 2.0 * np.sqrt(q1) * q2
        coeff2 = np.sqrt(q1) * (1.0 - 2.0 * q2)
    elif ld_law == "squareroot":
        coeff1 = np.sqrt(q1) * (1.0 - 2.0 * q2)
        coeff2 = 2.0 * np.sqrt(q1) * q2
    elif ld_law == "logarithmic":
        coeff1 = 1.0 - np.sqrt(q1) * q2
        coeff2 = 1.0 - np.sqrt(q1)
    elif ld_law == "linear":
        return q1, q2
    return coeff1, coeff2


def init_batman(t, law):
    """
    This function initializes the batman code.
    """
    params = batman.TransitParams()
    params.t0 = 0.0
    params.per = 1.0
    params.rp = 0.1
    params.a = 15.0
    params.inc = 87.0
    params.ecc = 0.0
    params.w = 90.0
    if law == "linear":
        params.u = [0.5]
    else:
        params.u = [0.1, 0.3]
    params.limb_dark = law
    m = batman.TransitModel(params, t)
    return params, m


def get_transit_model(t, t0, P, p, a, inc, q1, q2, ld_law):
    params, m = init_batman(t, law=ld_law)
    coeff1, coeff2 = reverse_ld_coeffs(ld_law, q1, q2)
    params.t0 = t0
    params.per = P
    params.rp = p
    params.a = a
    params.inc = inc
    if ld_law == "linear":
        params.u = [coeff1]
    else:
        params.u = [coeff1, coeff2]
    return m.light_curve(params)


# Initialize batman:
params, m = init_batman(t, law=ld_law)

# Now define MultiNest priors and log-likelihood:
def prior(cube, ndim, nparams):
    pcounter = 0  # keep track of prior indices in cube
    # Prior on "median flux" is uniform:
    cube[pcounter] = utils.transform_uniform(cube[pcounter], -2.0, 2.0)
    pcounter += 1

    # Pior on the log-jitter term (note this is the log VARIANCE, not sigma); from 0.01 to 100 ppm:
    cube[pcounter] = utils.transform_uniform(
        cube[pcounter], np.log((0.01e-3) ** 2), np.log((100e-3) ** 2)
    )
    pcounter += 1

    # Prior on t0:
    if t0mean is None:
        cube[pcounter] = utils.transform_uniform(
            cube[pcounter], np.min(t), np.max(t)
        )
    else:
        cube[pcounter] = utils.transform_normal(cube[pcounter], t0mean, t0sd)
    pcounter += 1

    # Prior on Period:
    if Pmean is None:
        cube[pcounter] = utils.transform_loguniform(
            cube[pcounter], 0.1, 1000.0
        )
    else:
        cube[pcounter] = utils.transform_normal(cube[pcounter], Pmean, Psd)
    pcounter += 1

    # Prior on impact parameter and planet-to-star radius ratio:
    if use_r1_r2:
        if (pl is None) or (pu is None):
            cube[pcounter] = utils.transform_uniform(cube[pcounter], 0, 1)
            pcounter += 1
            cube[pcounter] = utils.transform_uniform(cube[pcounter], 0, 1)
            pcounter += 1
        else:
            cube[pcounter] = utils.transform_uniform(cube[pcounter], pl, pu)
            pcounter += 1
            cube[pcounter] = utils.transform_uniform(cube[pcounter], pl, pu)
            pcounter += 1
    else:
        if bmean is None:
            cube[pcounter] = utils.transform_uniform(cube[pcounter], 0, 2.0)
        else:
            cube[pcounter] = utils.transform_truncated_normal(
                cube[pcounter], bmean, bsd, a=0.0, b=2.0
            )
        pcounter += 1
        if pmean is None:
            cube[pcounter] = utils.transform_uniform(cube[pcounter], 0, 1)
        else:
            cube[pcounter] = utils.transform_truncated_normal(
                cube[pcounter], pmean, psd
            )
        pcounter += 1

    # Prior on stellar density
    if use_rho_star:
        if rhomean is None:
            cube[pcounter] = utils.transform_uniform(
                cube[pcounter], 0.1, 300.0
            )
        else:
            cube[pcounter] = utils.transform_normal(
                cube[pcounter], rhomean, rhosd
            )
    else:
        # Prior on a/Rs or rho_star:
        if amean is None:
            cube[pcounter] = utils.transform_uniform(
                cube[pcounter], 0.1, 300.0
            )
        else:
            cube[pcounter] = utils.transform_normal(cube[pcounter], amean, asd)
    pcounter += 1

    # Prior either on the linear LD or the transformed first two-parameter law LD (q1):
    cube[pcounter] = utils.transform_uniform(cube[pcounter], 0, 1.0)
    pcounter += 1

    # (Transformed) limb-darkening coefficient for two-parameter laws (q2):
    if ld_law != "linear":
        cube[pcounter] = utils.transform_uniform(cube[pcounter], 0, 1.0)
        pcounter += 1

    if fixed_eccentricity == "False":
        if eccmean is None:
            cube[pcounter] = utils.transform_uniform(cube[pcounter], 0, 1.0)
        else:
            cube[pcounter] = utils.transform_truncated_normal(
                cube[pcounter], eccmean, eccsd, a=0.0, b=1.0
            )
        pcounter += 1
        if omegamean is None:
            cube[pcounter] = utils.transform_uniform(cube[pcounter], 0, 360.0)
        else:
            cube[pcounter] = utils.transform_truncated_normal(
                cube[pcounter], omegamean, omegasd, a=0.0, b=360.0
            )
        pcounter += 1

    # Prior on coefficients of comparison stars:
    if compfilename is not None:
        for i in range(Xc.shape[0]):
            cube[pcounter] = utils.transform_uniform(cube[pcounter], -10, 10)
            pcounter += 1

    # Prior on kernel maximum variance; from 0.01 to 100 mmag:
    cube[pcounter] = utils.transform_loguniform(
        cube[pcounter], (0.01 * 1e-3) ** 2, (100 * 1e-3) ** 2
    )
    pcounter += 1

    # Now priors on the alphas = 1/lambdas; gamma(1,1) = exponential, same as Gibson+:
    for i in range(X.shape[0]):
        cube[pcounter] = utils.transform_exponential(cube[pcounter])
        pcounter += 1


def loglike(cube, ndim, nparams):
    # Evaluate the log-likelihood. For this, first extract all inputs:
    mmean, ljitter, t0, P = [cube[i] for i in range(4)]
    q1 = cube[7]

    if use_rho_star:
        rho = cube[6]
        a = ((rho * G * ((P * 24.0 * 3600.0) ** 2)) / (3.0 * np.pi)) ** (
            1.0 / 3.0
        )
    else:
        a = cube[6]

    if use_r1_r2:
        r1, r2 = cube[4], cube[5]
        if r1 > Ar:
            b, p = (
                (1 + pl) * (1.0 + (r1 - 1.0) / (1.0 - Ar)),
                (1 - r2) * pl + r2 * pu,
            )
        else:
            b, p = (
                (1.0 + pl) + np.sqrt(r1 / Ar) * r2 * (pu - pl),
                pu + (pl - pu) * np.sqrt(r1 / Ar) * (1.0 - r2),
            )
    else:
        b, p = cube[4], cube[5]

    pcounter = 8
    if ld_law != "linear":
        q2 = cube[pcounter]
        coeff1, coeff2 = reverse_ld_coeffs(ld_law, q1, q2)
        params.u = [coeff1, coeff2]
        pcounter += 1
    else:
        params.u = [q1]

    if fixed_eccentricity == "False":
        ecc = cube[pcounter]
        pcounter += 1
        omega = cube[pcounter]
        pcounter += 1
    else:
        ecc = eccmean
        omega = omegamean

    ecc_factor = (1.0 + ecc * np.sin(omega * np.pi / 180.0)) / (1.0 - ecc ** 2)

    inc_inv_factor = (b / a) * ecc_factor
    # Check that b and b/aR are in physically meaningful ranges:
    if b > 1.0 + p or inc_inv_factor >= 1.0:
        return -1e101
    else:
        # Compute inclination of the orbit:
        inc = np.arccos(inc_inv_factor) * 180.0 / np.pi

        # Evaluate transit model:
        params.t0 = t0
        params.per = P
        params.rp = p
        params.a = a
        params.inc = inc
        params.ecc = ecc
        params.w = omega
        lcmodel = m.light_curve(params)

    model = mmean - 2.51 * np.log10(lcmodel)
    if compfilename is not None:
        for i in range(Xc.shape[0]):
            model = model + cube[pcounter] * Xc[i, idx]
            pcounter += 1
    max_var = cube[pcounter]
    pcounter = pcounter + 1
    alphas = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        alphas[i] = cube[pcounter]
        pcounter = pcounter + 1
    gp_vector = np.append(
        np.append(ljitter, np.log(max_var)), np.log(1.0 / alphas)
    )
    # Evaluate model:
    residuals = f - model
    gp.set_parameter_vector(gp_vector)
    return gp.log_likelihood(residuals)

#              v neparams   v max variance
n_params = 8 + X.shape[0] + 1
if compfilename is not None:
    n_params += Xc.shape[0]
if ld_law != "linear":
    n_params += 1
if fixed_eccentricity == "False":
    n_params += 2

print("Number of external parameters:", X.shape[0])
print("Number of comparison stars:", Xc.shape[0])
print("Number of counted parameters:", n_params)
out_file = out_folder + "out_multinest_trend_george_"

import pickle

# If not ran already, run MultiNest, save posterior samples and evidences to pickle file:
if not os.path.exists(out_folder + "posteriors_trend_george.pkl"):
    # Run MultiNest:
    pymultinest.run(
        loglike,
        prior,
        n_params,
        n_live_points=n_live_points,
        outputfiles_basename=out_file,
        resume=False,
        verbose=True,
    )
    # Get output:
    output = pymultinest.Analyzer(
        outputfiles_basename=out_file, n_params=n_params
    )
    # Get out parameters: this matrix has (samples,n_params+1):
    posterior_samples = output.get_equal_weighted_posterior()[:, :-1]

    # Extract parameters:
    mmean, ljitter, t0, P = posterior_samples[:, :4].T
    q1 = posterior_samples[:, 7]

    out = {}
    out["posterior_samples"] = {}

    if use_rho_star:
        rho = posterior_samples[:, 6]
        out["posterior_samples"]["rho"] = rho
    else:
        a = posterior_samples[:, 6]
        out["posterior_samples"]["a"] = a

    if use_r1_r2:
        r1, r2 = posterior_samples[:, 4:6].T
        out["posterior_samples"]["r1"] = r1
        out["posterior_samples"]["r2"] = r2
        b = np.zeros_like(r1)
        p = np.zeros_like(r1)
        gtAr = r1 > Ar
        if sum(gtAr) != 0:
            b[gtAr], p[gtAr] = (
                (1 + pl) * (1.0 + (r1[gtAr] - 1.0) / (1.0 - Ar)),
                (1 - r2[gtAr]) * pl + r2[gtAr] * pu,
            )
        if sum(~gtAr) != 0:
            b[~gtAr], p[~gtAr] = (
                (1.0 + pl) + np.sqrt(r1[~gtAr] / Ar) * r2[~gtAr] * (pu - pl),
                pu + (pl - pu) * np.sqrt(r1[~gtAr] / Ar) * (1.0 - r2[~gtAr]),
            )
    else:
        b, p = posterior_samples[:, 4:6].T
        out["posterior_samples"]["b"] = b
        out["posterior_samples"]["p"] = p

    out["posterior_samples"]["unnamed"] = posterior_samples
    out["posterior_samples"]["mmean"] = mmean
    out["posterior_samples"]["ljitter"] = ljitter
    out["posterior_samples"]["t0"] = t0
    out["posterior_samples"]["P"] = P
    out["posterior_samples"]["q1"] = q1

    pcounter = 8
    if ld_law != "linear":
        q2 = posterior_samples[:, pcounter]
        out["posterior_samples"]["q2"] = q2
        pcounter += 1

    if fixed_eccentricity == "False":
        ecc = posterior_samples[:, pcounter]
        out["posterior_samples"]["ecc"] = ecc
        pcounter += 1
        omega = posterior_samples[:, pcounter]
        out["posterior_samples"]["omega"] = omega
        pcounter += 1

    xc_coeffs = []
    if compfilename is not None:
        for i in range(Xc.shape[0]):
            xc_coeffs.append(posterior_samples[:, pcounter])
            out["posterior_samples"]["xc" + str(i)] = posterior_samples[
                :, pcounter
            ]
            pcounter += 1
    max_var = posterior_samples[:, pcounter]
    out["posterior_samples"]["max_var"] = max_var
    pcounter = pcounter + 1
    alphas = []
    for i in range(X.shape[0]):
        alphas.append(posterior_samples[:, pcounter])
        out["posterior_samples"]["alpha" + str(i)] = posterior_samples[
            :, pcounter
        ]
        pcounter = pcounter + 1

    a_lnZ = output.get_stats()["global evidence"]
    out["lnZ"] = a_lnZ
    pickle.dump(out, open(out_folder + "posteriors_trend_george.pkl", "wb"))
else:
    out = pickle.load(open(out_folder + "posteriors_trend_george.pkl", "rb"))
    posterior_samples = out["posterior_samples"]["unnamed"]

######### NEW EVALUATION METHOD: EXTRACT SAMPLES DIRECTLY FROM THE POSTERIOR DENSITY   ###########
######### INSTEAD OF EXTRACTING MEDIANS (THIS IS THE CORRECT WAY OF DOING THIS, NESTOR ###########
######### FROM THE PAST!)                                                              ###########
nsamples = len(out["posterior_samples"]["mmean"])
idx_samples = np.random.choice(
    np.arange(len(out["posterior_samples"]["mmean"])), nsamples, replace=False
)

detrended_lc = np.zeros([len(tall), nsamples])
detrended_lc_err = np.zeros([len(tall), nsamples])
transit_lc = np.zeros([len(tall), nsamples])
systematic_model_lc = np.zeros([len(tall), nsamples])

counter = 0
for i in idx_samples:
    mmean, ljitter, max_var, t0, P, q1 = (
        out["posterior_samples"]["mmean"][i],
        out["posterior_samples"]["ljitter"][i],
        out["posterior_samples"]["max_var"][i],
        out["posterior_samples"]["t0"][i],
        out["posterior_samples"]["P"][i],
        out["posterior_samples"]["q1"][i],
    )

    if use_rho_star:
        rho = out["posterior_samples"]["rho"][i]
        a = ((rho * G * ((P * 24.0 * 3600.0) ** 2)) / (3.0 * np.pi)) ** (
            1.0 / 3.0
        )
    else:
        a = out["posterior_samples"]["a"][i]

    if use_r1_r2:
        r1, r2, = (
            out["posterior_samples"]["r1"][i],
            out["posterior_samples"]["r2"][i],
        )
        if r1 > Ar:
            b, p = (
                (1 + pl) * (1.0 + (r1 - 1.0) / (1.0 - Ar)),
                (1 - r2) * pl + r2 * pu,
            )
        else:
            b, p = (
                (1.0 + pl) + np.sqrt(r1 / Ar) * r2 * (pu - pl),
                pu + (pl - pu) * np.sqrt(r1 / Ar) * (1.0 - r2),
            )
    else:
        b, p = (
            out["posterior_samples"]["b"][i],
            out["posterior_samples"]["p"][i],
        )

    if ld_law != "linear":
        q2 = out["posterior_samples"]["q2"][i]
        coeff1, coeff2 = reverse_ld_coeffs(ld_law, q1, q2)
        params.u = [coeff1, coeff2]
    else:
        params.u = [q1]
    alphas = np.zeros(X.shape[0])
    for j in range(X.shape[0]):
        alphas[j] = out["posterior_samples"]["alpha" + str(j)][i]

    if fixed_eccentricity == "False":
        ecc = out["posterior_samples"]["ecc"][i]
        omega = out["posterior_samples"]["omega"][i]
    else:
        ecc = eccmean
        omega = omegamean

    ecc_factor = (1.0 + ecc * np.sin(omega * np.pi / 180.0)) / (1.0 - ecc ** 2)
    inc_inv_factor = (b / a) * ecc_factor
    inc = np.arccos(inc_inv_factor) * 180.0 / np.pi

    params.t0 = t0
    params.per = P
    params.rp = p
    params.a = a
    params.inc = inc
    params.ecc = ecc
    params.w = omega

    lcmodel = m.light_curve(params)

    model = -2.51 * np.log10(lcmodel)
    comp_model = mmean
    if compfilename is not None:
        for j in range(Xc.shape[0]):
            comp_model = (
                comp_model
                + out["posterior_samples"]["xc" + str(j)][i] * Xc[j, idx]
            )

    # Evaluate model:
    residuals = f - (model + comp_model)
    gp_vector = np.append(
        np.append(ljitter, np.log(max_var)), np.log(1.0 / alphas)
    )
    gp.set_parameter_vector(gp_vector)

    pred_mean, pred_var = gp.predict(residuals, X.T, return_var=True)
    # fout,fout_err = utils.mag_to_flux(fall-comp_model,np.ones(len(tall))*np.sqrt(np.exp(ljitter)))
    # pred_mean_f,fout_err = utils.mag_to_flux(pred_mean,np.ones(len(tall))*np.sqrt(np.exp(ljitter)))

    detrended_lc[:, counter] = f - (comp_model + pred_mean)
    detrended_lc_err[:, counter] = np.sqrt(np.ones(len(f)) * np.exp(ljitter))
    transit_lc[:, counter] = lcmodel
    systematic_model_lc[:, counter] = pred_mean + comp_model

    counter = counter + 1

##################################################################################################

fileout = open("detrended_lc.dat", "w")
file_model_out = open("model_lc.dat", "w")
fileout.write("# Time   DetFlux   DetFluxErr   Model\n")
file_model_out.write("# Time   Mag   ModelMag   ModelMagUp68   ModelMagDown68   ModelMagUp95   ModelMagDown95\n")
for i in range(detrended_lc.shape[0]):
    val = np.median(detrended_lc[i, :])
    val_err = np.median(detrended_lc_err[i, :])
    dist = 10 ** (-np.random.normal(val, val_err, 1000) / 2.51)
    val, val_err = np.median(dist), np.sqrt(np.var(dist))
    mval = np.median(transit_lc[i, :])
    fileout.write(
        "{0:.10f} {1:.10f} {2:.10f} {3:.10f}\n".format(
            tall[i], val, val_err, mval
        )
    )
    val, valup, valdown = get_quantiles(systematic_model_lc[i, :])
    val95, valup95, valdown95 = get_quantiles(
        systematic_model_lc[i, :], alpha=0.95
    )
    file_model_out.write(
        "{0:.10f} {1:.10f} {2:.10f} {3:.10f} {4:.10f} {5:.10f} {6:.10f}\n".format(
            tall[i], f[i], val, valup, valdown, valup95, valdown95
        )
    )

print("Saved!")
fileout.close()
file_model_out.close()
