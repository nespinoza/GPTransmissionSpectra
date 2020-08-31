# Folder and file where pickle file is (folder/file.pkl); both
# names are important (folder saves the target name; HAS TO BE THE SAME AS THE
# TARGET NAME IN THE PICKLE FILE and file the filename):
datafile = "HATP23b/hp23b_180603_custom.pkl"
outfold = "out_c"
# Limb-darkening law to use:
ld_law = "linear"
# Time indexes you want to omit for the fit in pythonic-language.
# If times are in a variable t the times to be fitted will be np.delete(t, bad_idx_time)
bad_idx_time = "[0:17]"
# Which comparison stars to use. Same ordering as in pickle file:
comps = [1, 0]
# Priors (assumed gaussian) Sada & Ramon-Fox et al. (2016)
Pmean = 1.2128864726
amean = 4.28971
pmean, psd = 0.1164954324, 0.01
bmean = 0.4751723328
t0mean = 2454852.2654074058
# Fix the eccentricity? If True, pass only eccmean and omegamean.
# Those will be fixed in the fit.
fixed_eccentricity = True
eccmean = 0.0
omegamean = 90.0
PCA = True
GPkernel = "multi_matern"
nlive = 300
nopickle = False
