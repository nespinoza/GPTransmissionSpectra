import numpy as np
import batman
import utils
import argparse
import os
import importlib

# Define constants on the code:
G = 6.67408e-8  # Gravitational constant, cgs

# This parses in the option file:
parser = argparse.ArgumentParser()
parser.add_argument("-ofile", default=None)
args = parser.parse_args()
ofile = args.ofile

# Read input file:
c = importlib.import_module(ofile)
out_folder_base = c.out_folder_base
datafile = c.datafile
ld_law = c.ld_law
eccmean = c.eccmean
omegamean = c.omegamean
numpoints = 1000  # number of time points to interpolate
out_wl = f"{out_folder_base}/{datafile.split('.')[0]}/white-light"

if hasattr(c, "amean"):
    use_rho_star = False
else:
    use_rho_star = True

if hasattr(c, "bmean"):
    use_r1_r2 = False
else:
    use_r1_r2 = True
    pl, pu = c.pl, c.pu
    Ar = (pu - pl) / (2.0 + pl + pu)

n = 1  # Starting PCA number
path_PCA = f"{out_wl}/PCA_{n}/posteriors_trend_george.pkl"
while os.path.exists(path_PCA):
    # load PCA posterior and lc times
    lc_path = f"{out_wl}/lc.dat"
    pca_post = utils.load_pkl(path_PCA)
    lc_times = np.genfromtxt(lc_path, unpack=True, usecols=(0))

    # hold (N x M) matrix of interpolated transit models where,
    # N = number of interpolated points in time
    # M = number of samples taken from PCA posterior w/o repl. = number of points in PCA posterior
    # each column of the matrix `transit_lc` will then be a transit model determined by
    # a given set of transit parameters that is sampled from the PCA posterior
    tmin, tmax = np.min(lc_times), np.max(lc_times)
    t = np.linspace(
        tmin, tmax, numpoints
    )  # interpolate lc times to produce a smooth model later
    nsamples = len(pca_post["posterior_samples"]["mmean"])
    transit_lc = np.zeros([len(t), nsamples])

    # transit model functions
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

    # Initialize batman:
    params, m = init_batman(t, law=ld_law)

    # sample from PCA posterior w/o replacement
    idx_samples = np.random.choice(
        np.arange(len(pca_post["posterior_samples"]["mmean"])),
        nsamples,
        replace=False,
    )
    Nsamples = len(idx_samples)
    counter = 0
    for i in idx_samples:
        if i % 100 == 0:
            print(counter, "/", Nsamples)

        # Build transit model
        mmeani, t0, P, q1 = (
            pca_post["posterior_samples"]["mmean"][i],
            pca_post["posterior_samples"]["t0"][i],
            pca_post["posterior_samples"]["P"][i],
            pca_post["posterior_samples"]["q1"][i],
        )

        if use_rho_star:
            rho = pca_post["posterior_samples"]["rho"][i]
            a = ((rho * G * ((P * 24.0 * 3600.0) ** 2)) / (3.0 * np.pi)) ** (
                1.0 / 3.0
            )
        else:
            a = pca_post["posterior_samples"]["a"][i]

        if use_r1_r2:
            r1, r2, = (
                pca_post["posterior_samples"]["r1"][i],
                pca_post["posterior_samples"]["r2"][i],
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
                pca_post["posterior_samples"]["b"][i],
                pca_post["posterior_samples"]["p"][i],
            )

        # Limb darkening
        if ld_law != "linear":
            q2 = pca_post["posterior_samples"]["q2"][i]
            coeff1, coeff2 = reverse_ld_coeffs(ld_law, q1, q2)
            params.u = [coeff1, coeff2]
        else:
            params.u = [q1]

        # Ecc, inclination
        ecc = eccmean
        omega = omegamean
        ecc_factor = (1.0 + ecc * np.sin(omega * np.pi / 180.0)) / (
            1.0 - ecc ** 2
        )
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

        # add model to matrix
        transit_lc[:, counter] = lcmodel

        counter = counter + 1

    # median average models together
    mval = np.median(transit_lc, axis=1)

    # write results to table
    x = np.append(t.reshape(len(t), 1), mval.reshape(len(t), 1), axis=1)
    fpath = f"{out_wl}/full_model_PCA_{n}.dat"
    np.savetxt(fpath, x)
    print(f"Saved to {fpath}")
    n += 1
    path_PCA = f"{out_wl}/PCA_{n}/posteriors_trend_george.pkl"

print("Done!")
