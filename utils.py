from scipy.stats import gamma, norm, beta, truncnorm
import numpy as np
import pickle
import batman

def load_pkl(fpath):
    with open(fpath, "rb") as f:
        data = pickle.load(f)
    return data

# READ OPTION FILE:
def _to_arr(idx_or_slc):
    # Converts str to 1d numpy array
    # or slice to numpy array of ints.
    # This format makes it easier for flattening multiple arrays in `_bad_idxs`
    if ":" in idx_or_slc:
        lower, upper = map(int, idx_or_slc.split(":"))
        return np.arange(lower, upper + 1)
    else:
        return np.array([int(idx_or_slc)])

def _bad_idxs(s):
    if s == "[]":
        return []
    else:
        # Merges indices/slices specified in `s` into a single numpy array of
        # indices to omit
        s = s.strip("[]").split(",")
        bad_idxs = list(map(_to_arr, s))
        bad_idxs = np.concatenate(bad_idxs, axis=0)
        return bad_idxs

# TRANSFORMATION OF PRIORS:
def transform_uniform(x, a, b):
    return a + (b - a) * x

def transform_loguniform(x, a, b):
    la = np.log(a)
    lb = np.log(b)
    return np.exp(la + x * (lb - la))

def transform_normal(x, mu, sigma):
    return norm.ppf(x, loc=mu, scale=sigma)

def transform_beta(x, a, b):
    return beta.ppf(x, a, b)

def transform_exponential(x, a=1.0):
    return gamma.ppf(x, a)

def transform_truncated_normal(x, mu, sigma, a=0.0, b=1.0):
    ar, br = (a - mu) / sigma, (b - mu) / sigma
    return truncnorm.ppf(x, ar, br, loc=mu, scale=sigma)

# PCA TOOLS:
def get_sigma(x):
    """
    This function returns the MAD-based standard-deviation.
    """
    median = np.median(x)
    mad = np.median(np.abs(x - median))
    return 1.4826 * mad

def standarize_data(input_data):
    output_data = np.copy(input_data)
    averages = np.median(input_data, axis=1)
    for i in range(len(averages)):
        sigma = get_sigma(output_data[i, :])
        output_data[i, :] = output_data[i, :] - averages[i]
        output_data[i, :] = output_data[i, :] / sigma
    return output_data

def classic_PCA(Input_Data, standarize=True):
    """
    classic_PCA function

    Description

    This function performs the classic Principal Component Analysis on a given dataset.
    """
    if standarize:
        Data = standarize_data(Input_Data)
    else:
        Data = np.copy(Input_Data)
    eigenvectors_cols, eigenvalues, eigenvectors_rows = np.linalg.svd(
        np.cov(Data)
    )
    idx = eigenvalues.argsort()
    eigenvalues = eigenvalues[idx[::-1]]
    eigenvectors_cols = eigenvectors_cols[:, idx[::-1]]
    eigenvectors_rows = eigenvectors_rows[idx[::-1], :]
    # Return: V matrix, eigenvalues and the principal components.
    return eigenvectors_rows, eigenvalues, np.dot(eigenvectors_rows, Data)

# Post-processing tools:
def mag_to_flux(m, merr):
    """
    Convert magnitude to relative fluxes.
    """
    fluxes = np.zeros(len(m))
    fluxes_err = np.zeros(len(m))
    for i in range(len(m)):
        dist = 10 ** (-np.random.normal(m[i], merr[i], 1000) / 2.51)
        fluxes[i] = np.mean(dist)
        fluxes_err[i] = np.sqrt(np.var(dist))
    return fluxes, fluxes_err

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
