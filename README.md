GPTransmissionSpectra
---

This code detrends lightcurves and generates transmission spectra given data in a pickle file in the format of tepspec. To use it you need:

- MultiNest
- batman
- george

The code fits the lightcurve simultaneously using a transit model, the comparison stars using PCA and a gaussian process with a 
multi-dimensional squared exponential kernel, which takes time, FWHM, airmass, trace position, sky flux, and shift in wavelength of the 
trace as inputs.

USAGE
---

To use the code is simple: 

1. Put the .pkl file of outputs in a folder which has the same name as the target in the pickle file. Let's assume 
the name of the target in my pickle file is `WASP19`. Then, my pickle file (say, named `w19_140322.pkl`) should be in the folder 
`WASP19/w19_140322.pkl`.

2. Create an options file for the white-light lightcurve (see the `wl_options.dat` file for an example with WASP-19, `wl_options_h5.dat` 
for HATS-5), which will contain the priors for the fit (assumed to be truncated gaussians).

3. Run the white-light analysis by doing `python run_wl_analysis.py -ofile youroptionsfile.dat`. This will run the white-light analysis. Once 
is done, the results will be outputted in a folder named `outputs`.

4. When the white-light fit ends, the `outputs` folder will have a `white-light` folder, inside of which you will find a `results.dat` file. 
This contain the posterior parameters of the best-fit. Inside `white-light` there will also be folders named `PCA_n`, where `n` is the number 
of PCA components used for each fit (the code tries them all, and then bayesian-model average the results to obtain the `results.dat` file); 
inside each `PCA_n` folder there will be a `detrended_lc.dat` file with the detrended lightcurves (first column is time, second detrended 
lightcurve, third noise on the detrended lightcurve and fourth the best-fit transit model).

5. Use the `results.dat` to create an options file for the wavelength-dependant fits, where every parameter of the transit will be fixed except for 
the limb-darkening parameters and `p=rp/rs` (and, of course, the GP and PCA components of the fit). See the `wavelength_options_w19.dat` file for 
an example.

6. Run the code by doing `python run_wavelength_analysis.py -ofile yourNEWoptionsfile.dat`.

7. Once it runs, generate the transmission spectrum by running `python compile_transpec.py -ofile yourNEWoptionsfile.dat`. This will be saved in the 
folder `wavelengths` inside the outputs folder of your target lightcurve.
