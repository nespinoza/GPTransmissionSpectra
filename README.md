GPTransmissionSpectra
---

This code detrends lightcurves and generates transmission spectra given data either (1) in a pickle file in the output format of tepspec 
or (2) given lightcurves and a set of external parameters. To use it you need:

- MultiNest
- batman
- george

The code fits the lightcurve simultaneously using a transit model, the comparison stars using PCA and a gaussian process with a 
multi-dimensional squared exponential kernel (or a Matern 3/2; for this, simply run the code adding the flag `--matern`), which takes time, FWHM, airmass, trace position, sky flux, and shift in wavelength of the 
trace as inputs in the case of output in the tepspec format. If this is not the input format, the user defines which external paraemters 
to include instead.

USAGE
---

To use the code is simple: 

1. *(a)* If the input is going to be given by the user with the tepspec input format, put the .pkl file of outputs in a folder which has the same 
   name as the target in the pickle file. Let's assume the name of the target in my pickle file is `WASP19`. Then, my pickle file (say, named 
   `w19_140322.pkl`) should be in the folder `WASP19/w19_140322.pkl`. *(b)* If you have a set of lightcurves and external parameters, first 
   create a folder called "outputs". Inside it, create a folder with the name of your target (in our case, `WASP19`). Inside this folder, 
   create different folders for different datasets (e.g., different nights) of the same target --- in the case of our example, we should create 
   a folder called `w19_140322`. Inside each of these, put your external parameters in a file called eparams.dat, so that each row is the value 
   of the external parameters at different times, and each (space separated) column is a different external parameter. Inside this folder, create 
   two extra folders: a folder called `white-light` and a folder called `wavelength`. Inside the `white-light` folder, create a file called 
   `lc.dat` which contains the data for the target lightcurve: in its first column the time, the second the (median-substracted) *magnitude* 
   (-2.51 x log10(flux)) and the third column contains zeros. Create another file called `comps.dat` where you input the (median-substracted) 
   *magnitude* of the comparison stars; one comparison star per column. Insite the `wavelength` folder, create one different folder for each 
   wavelength bin named as `wbin0`, `wbin1`, etc., and inside save the (wavelength-dependant) lightcurves of the target and comparison stars 
   in the same format as was done in the `white-light` folder. 

2. Create an options file for the white-light lightcurve (see the `wl_options.dat` file for an example with WASP-19, 
   `wl_options_h5.dat` for HATS-5), which will contain the priors for the fit (assumed to be truncated gaussians). If 
   your input is not in the tespect format (i.e, you followed step *(b)* in 1.), the `datafile` parameter should be still 
   filled with the .pkl extension (even though you don't have a pickle file --- this is just to allow backcompatibility of 
   the code). In the case of our example, it should be `WASP19/w19_140322.pkl`.
   

3. Run the white-light analysis by doing `python run_wl_analysis.py -ofile youroptionsfile.dat`; if you don't have a pickle file in the 
   format of tepspec, then add the --nopickle flag (i.e., do `python run_wl_analysis.py -ofile youroptionsfile.dat --nopickle`). This will 
   run the white-light analysis. Once is done, the results will be outputted in a folder named `outputs`.

4. When the white-light fit ends, if you used the tepspec pickle mode, the `outputs` folder will have a `white-light` folder, inside of 
   which you will find a `results.dat` file. If you didn't, these same files will be written there. This contain the posterior parameters 
   of the best-fit. Inside `white-light` there will also be folders named `PCA_n`, where `n` is the number of PCA components used for each 
   fit (the code tries them all, and then bayesian-model average the results to obtain the `results.dat` file); inside each `PCA_n` folder 
   there will be a `detrended_lc.dat` file with the detrended lightcurves (first column is time, second detrended lightcurve, third noise 
   on the detrended lightcurve and fourth the best-fit transit model) and a `model_lc` with the raw magnitude of the target and the full 
   systematic model (i.e., the full model minus the transit). Note you can join the data in these two files to generate the full model 
   fitted to the data.

5. Use the `results.dat` to create an options file for the wavelength-dependant fits, where every parameter of the transit will be fixed 
   except for the limb-darkening parameters and `p=rp/rs` (and, of course, the GP and PCA components of the fit). See the 
   `wavelength_options_w19.dat` file for an example. Note: if analysing multiple nights, consider using `results.dat` values averaged
   over all nights.

6. Run the code by doing either `python run_wavelength_analysis.py -ofile yourNEWoptionsfile.dat` if you want to perform PCA + GP on each 
   wavelength range, or `python run_wavelength_cmc_analysis.py -ofile yourNEWoptionsfile.dat -wofile youroptionsfile.dat` if you want to 
   run Common Mode Correction (CMC) } GP, where `youroptionsfile.dat` is the same white-light option file created for step 3 above. Here, 
   if your input format is not the tepspec pickle, you can use the --nopickle flag again. Here, for the common-mode correction, only the 
   first star in the list of your `yourNEWoptionsfile.dat` options file will be used; the white-light lightcurve of the target and that 
   comparison star will be divided, the best-fit transit lightcurve from step 3 and 4 will be divided to that resulting lightcurve, and this 
   will be the common-mode correction signal. This signal, in turn, will be divided to the resulting lightcurve of the division between the 
   target and the same comparison on every wavelength, and this lightcurve will be fitted directly without the PCA component (but with a 
   zero-point and a GP). 

7. Once it runs, generate the transmission spectrum by running `python compile_transpec.py -ofile yourNEWoptionsfile.dat` if you did a PCA + GP fit, 
   or add `--CMC` in case you want to compile the results from the common-mode correction fit. This will be saved as `transpec.dat` inside the 
   outputs folder of your target lightcurve in the case of a PCA + GP fit or `transpec_cmc.dat` in the case of a common-mode correction fit.

TODO
---
- Try other kernels?
