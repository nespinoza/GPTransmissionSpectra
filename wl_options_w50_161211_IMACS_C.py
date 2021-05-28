# Top level name for outputs directory
# Same as out_l, only comps available
out_folder_base = "out_l_C"
# Folder and file where pickle file is (folder/file.pkl); both
# names are important (folder saves the target name; HAS TO BE THE SAME AS THE
# TARGET NAME IN THE PICKLE FILE and file the filename):
datafile = "WASP50/w50_161211_IMACS.pkl"
# Limb-darkening law to use:
ld_law = "linear"
# Time indexes you want to omit for the fit in pythonic-language.
# If times are in a variable t the times to be fitted will be np.delete(t, bad_idx_time)
# window_width=15, ferr=0.002
bad_idx_time = "[14, 17, 18, 37, 40, 41, 42, 43, 44, 45, 48, 51, 53, 55, 56, 57, 58, 77, 88, 134, 151, 155, 162, 170, 181, 182, 211, 212, 220, 246, 248, 259, 286, 288, 295, 296, 301, 305, 308, 317, 321]"
# Which comparison stars to use. Same ordering as in pickle file:
comps = [0, 1]
# Priors (assumed gaussian) TESS multi-sector fit
Pmean, Psd = 1.9550931258, 3.963e-7
t0mean, t0sd = 2455558.61237, 0.00020 # Bonomo et al. (2017)
rhomean, rhosd = 2.0503245, 0.4676309 # TICv8
pl, pu = 0.0, 1.0
# Fix the eccentricity? If True, pass only eccmean and omegamean.
# Those will be fixed in the fit.
fixed_eccentricity = True
eccmean, eccsd = 0.0, 0.0
omegamean, omegasd = 90.0, 0.0
PCA = True
GPkernel = "multi_matern"
nlive = 1000
nopickle = False
