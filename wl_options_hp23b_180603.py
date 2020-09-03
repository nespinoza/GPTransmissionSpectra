# Top level name for outputs directory
out_folder_base = "out_ab"
# Folder and file where pickle file is (folder/file.pkl); both
# names are important (folder saves the target name; HAS TO BE THE SAME AS THE
# TARGET NAME IN THE PICKLE FILE and file the filename):
datafile = "HATP23b/hp23b_180603_custom.pkl"
# Limb-darkening law to use:
ld_law = "linear"
# Time indexes you want to omit for the fit in pythonic-language.
# If times are in a variable t the times to be fitted will be np.delete(t, bad_idx_time)
bad_idx_time = "[0:17]"
# Which comparison stars to use. Same ordering as in pickle file:
comps = [1, 0]
# Priors (assumed gaussian) Sada & Ramon-Fox et al. (2016)
Pmean, Psd = 1.2128867, 0.0000002
amean, asd = 4.26, 0.5
# rhomean, rhosd = 0.99471000, 0.23240140
bmean, bsd = 0.36, 0.11
pmean, psd = 0.1113, 0.1
# pl, pu = 0.0, 1.0
t0mean, t0sd = 2454852.26548, 0.00017
# Fix the eccentricity? If True, pass only eccmean and omegamean.
# Those will be fixed in the fit.
fixed_eccentricity = True
eccmean, eccsd = 0.0, 0.0
omegamean, omegasd = 90.0, 0.0
PCA = True
GPkernel = "multi_matern"
nlive = 300
nopickle = False
