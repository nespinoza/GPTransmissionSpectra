import numpy as np
import batman
import george
import utils
import pandas as pd
from astropy.constants import G as const_G

# Define constants on the code:
G = const_G.cgs.value  # Gravitational constant, cgs

# Settings
ld_law = "linear"
eccmean = 0.0
omegamean = 90.0
Npoints = 1000 # Number of points in model
pl, pu = 0, 1

out_wl = f"out_b/WASP50/w50_161211_sp_IMACS_b/white-light"

# load BMA WLC results and lc times
df_results = pd.read_csv(
    f"{out_wl}/results.dat",
    comment='#',
    index_col="Variable",
    skipinitialspace=True,
)

lc_times = np.genfromtxt(f"{out_wl}/lc.dat", delimiter=',', unpack=True, usecols=(0))
tmin, tmax = np.min(lc_times), np.max(lc_times)
t_interp = np.linspace(
    tmin, tmax, Npoints
)  # interpolate lc times to produce a smooth model later
transit_lc = np.zeros([len(t_interp), Npoints])

# Initialize batman:
params, m = utils.init_batman(t_interp, law=ld_law)

# Build transit model
mmeani, t0, P, r1, r2, q1 = (
    df_results.at["mmean", "Value"],
    df_results.at["t0", "Value"],
    df_results.at["P", "Value"],
    df_results.at["r1", "Value"],
    df_results.at["r2", "Value"],
    df_results.at["q1", "Value"],
)

if "rho" in df_results.columns:
    rhos = df_results.at["rho", "Value"]
    aRs = ((rhos * G * ((P * 24.0 * 3600.0) ** 2)) / (3.0 * np.pi)) ** (
        1.0 / 3.0
    )
else:
    aRs = df_results.at["aR", "Value"]

# Transform r1, r2 -> b, p
Ar = (pu - pl) / (2.0 + pl + pu)
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

# Limb darkening
if ld_law != "linear":
    q2 = df_results.at["posterior_samples"]["q2", "Value"]
    coeff1, coeff2 = utils.reverse_ld_coeffs(ld_law, q1, q2)
    params.u = [coeff1, coeff2]
else:
    params.u = [q1]

# Ecc, inclination
ecc = eccmean
omega = omegamean
ecc_factor = (1.0 + ecc * np.sin(omega * np.pi / 180.0)) / (
    1.0 - ecc ** 2
)
inc_inv_factor = (b / aRs) * ecc_factor
inc = np.arccos(inc_inv_factor) * 180.0 / np.pi

params.t0 = t0
params.per = P
params.rp = p
params.a = aRs
params.inc = inc
params.ecc = ecc
params.w = omega

lcmodel_interp = m.light_curve(params)

# write results to table
savepath = f"{out_wl}/full_model_BMA.dat"
x = np.append(t_interp.reshape(len(t_interp), 1), lcmodel_interp.reshape(len(t_interp), 1), axis=1)
np.savetxt(savepath, x)
print(f"Saved BMA WLC model to {savepath}")

################
# BMA DETRENDING
################
BMA = df_results
# Raw data
tall, fall, f_index = np.genfromtxt(f"{out_wl}/lc.dat", delimiter=',', unpack=True)
idx = np.where(f_index == 0)[0]
t, f = tall[idx], fall[idx]

# External params
data = np.genfromtxt(f"{out_wl}/../eparams.dat", delimiter=',', skip_header=1)
X = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

# Comp stars
data = np.genfromtxt(f"{out_wl}/comps.dat", delimiter=',')
Xc = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
if len(Xc.shape) != 1:
    eigenvectors, eigenvalues, PC = utils.classic_PCA(Xc.T)
    Xc = PC.T
else:
    Xc = Xc[:, None]

# Comp star model
mmean = BMA.at["mmean", "Value"]
xcs = [xci for xci in BMA.index if "xc" in xci]
xc = np.array([BMA.at[f"{xci}", "Value"] for xci in xcs])
comp_model = mmean + np.dot(Xc[idx, :], xc)

###############
# Transit model
###############
JITTER=(200.0 * 1e-6)**2.0,
params, m = utils.init_batman(t, law=ld_law)
params.t0 = t0
params.per = P
params.rp = p
params.a = aRs
params.inc = inc
params.ecc = ecc
params.w = omega
lcmodel = m.light_curve(params)
model = -2.51 * np.log10(lcmodel)

#####
# GP
#####
kernel = np.var(f) * george.kernels.Matern32Kernel(
    np.ones(X[idx, :].shape[1]),
    ndim=X[idx, :].shape[1],
    axes=list(range(X[idx, :].shape[1])),
)

jitter = george.modeling.ConstantModel(np.log(JITTER))
ljitter = np.log(BMA.at["jitter", "Value"] ** 2)
max_var = BMA.at["max_var", "Value"]
alpha_names = [k for k in BMA.index if "alpha" in k]
alphas = np.array([BMA.at[alpha, "Value"] for alpha in alpha_names])

gp = george.GP(
    kernel,
    mean=0.0,
    fit_mean=False,
    white_noise=jitter,
    fit_white_noise=True,
)
gp.compute(X[idx, :])
gp_vector = np.r_[ljitter, np.log(max_var), np.log(1.0 / alphas)]
gp.set_parameter_vector(gp_vector)

############
# Detrending
############
residuals = f - (model + comp_model)

pred_mean, pred_var = gp.predict(residuals, X, return_var=True)

detrended_lc = f - (comp_model + pred_mean)
detrended_lc_err = np.sqrt(np.ones(len(f))*np.exp(ljitter))

LC_det = 10 ** (-detrended_lc / 2.51)
LC_det_err = detrended_lc_err
LC_transit_model = lcmodel
LC_systematics_model = comp_model + pred_mean

cube =  {
    "LC_det": LC_det,
    "LC_det_err": LC_det_err,
    "LC_transit_model": LC_transit_model,
    "LC_systematics_model": LC_systematics_model,
    "comp_model": comp_model,
    "pred_mean": pred_mean,
    "t": t,
    "t_interp": t_interp,
    "LC_det_model_interp": lcmodel_interp,
    "t0": t0,
    "P": P,
}

print("Saving BMA WLC cube")
np.save(f"{out_wl}/BMA_WLC.npy", cube)
