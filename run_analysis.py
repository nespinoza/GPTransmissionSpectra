import numpy as np
import pickle
import os

datafile = 'WASP19/w19_140322.pkl' 
ld_law_white = 'squareroot'
ld_law_wavelength = 'linear'
comps = [0,1,2,3,4,6]
target,pfilename = datafile.split('/')
out_folder = 'outputs/'+datafile.split('.')[0]

if not os.path.exists('outputs'):
    os.mkdir('outputs')

if not os.path.exists('outputs/'+target):
    os.mkdir('outputs/'+target)

if not os.path.exists(out_folder):
    os.mkdir(out_folder)
    data = pickle.load(open(datafile,'rb'))
    if not os.path.exists(out_folder+'/white-light'):
        os.mkdir(out_folder+'/white-light')
        # 1. Save external parameters:
        out_eparam = open(out_folder+'/eparams.dat','w')
        # Get median of FWHM, background flux, accross all wavelengths, and trace position of zero point.
        # First, find chips-names of target:
        names = []
        for name in data['fwhm'].keys():
            if target in name:
                names.append(name)
        if len(names) == 1:
            Xfwhm = data['fwhm'][names[0]]
            Xsky = data['sky'][names[0]]
        else:
            Xfwhm = np.hstack((data['fwhm'][names[0]],data['fwhm'][names[1]]))
            Xsky = np.hstack((data['sky'][names[0]],data['sky'][names[1]]))
        fwhm = np.zeros(Xfwhm.shape[0])
        sky = np.zeros(Xfwhm.shape[0])
        trace = np.zeros(Xfwhm.shape[0])
        for i in range(len(fwhm)):
            idx = np.where(Xfwhm[i,:]!=0)[0]
            fwhm[i] = np.median(Xfwhm[i,idx])
            idx = np.where(Xsky[i,:]!=0)[0]
            sky[i] = np.median(Xsky[i,idx])
            trace[i] = np.polyval(data['traces'][target][i],Xfwhm.shape[1]/2)
        print 'Saving eparams...'
        # Save external parameters:
        out_eparam.write('#Times \t                 Airmass \t Delta Wav \t FWHM \t        Sky Flux \t      Trace Center \n')
        for i in range(len(data['t'])):
            out_eparam.write('{0:.10f} \t {1:.10f} \t {2:.10f} \t {3:.10f} \t {4:.10f} \t {5:.10f} \n'.format(data['t'][i],\
                              data['Z'][i],data['deltas'][target+'_final'][i],fwhm[i],sky[i],trace[i]))
        out_eparam.close()
  
        # 2. Save (mean-substracted) target and comparison lightcurves (in magnitude-space):
        lcout = open(out_folder+'/white-light/lc.dat','w')
        lccompout = open(out_folder+'/white-light/comps.dat','w')
        for i in range(len(data['t'])):
            lcout.write('{0:.10f} {1:.10f} 0\n'.format(data['t'][i],-2.51*np.log10(data['oLC'][i])-np.median(-2.51*np.log10(data['oLC']))))
            for j in range(len(comps)): 
                if j != len(comps)-1:
                    lccompout.write('{0:.10f} \t'.format(-2.51*np.log10(data['cLC'][i,comps[j]]) - np.median(-2.51*np.log10(data['cLC'][:,comps[j]]))))
                else:
                    lccompout.write('{0:.10f}\n'.format(-2.51*np.log10(data['cLC'][i,comps[j]]) - np.median(-2.51*np.log10(data['cLC'][:,comps[j]]))))
        lcout.close()
        lccompout.close() 

# 3. Run code for all PCAs:
for i in range(1,len(comps)+1): 
    if not os.path.exists(out_folder+'/white-light/PCA_'+str(i)):
        os.system('python GPTransitDetrend.py -outfolder '+out_folder+'/white-light/ -compfile '+out_folder+\
                      '/white-light/comps.dat -lcfile '+out_folder+'/white-light/lc.dat -eparamfile '+out_folder+\
                      '/eparams.dat -ldlaw '+ld_law_white+' -Pmean 0.788839316 -Psd 0.000000017 -amean 3.50 -asd 0.1 '+\
                      '-pmean 0.14 -psd 0.01 -bmean 0.6 -bsd 0.1 -t0mean 2456739.547178 -t0sd 0.001 -eccmean 0.0046 '+\
                      '-eccsd 0.0044 -omegamean 3.0 -omegasd 70.0 --PCA -pctouse '+str(i))
        os.mkdir(out_folder+'/white-light/PCA_'+str(i))
        os.system('mv '+out_folder+'/white-light/out* '+out_folder+'/white-light/PCA_'+str(i)+'/.')
        os.system('mv '+out_folder+'/white-light/*.pkl '+out_folder+'/white-light/PCA_'+str(i)+'/.')
        os.system('mv detrended_lc.dat '+out_folder+'/white-light/PCA_'+str(i)+'/.')
