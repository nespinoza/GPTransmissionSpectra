import numpy as np
import utils
import pickle
import os

# Define folder/dataset:
datafile = 'WASP19/w19_140322.pkl' 
# Define LD law to use, comparison stars to use:
ld_law = 'linear'
all_comps = [0,1,2,3,4,6]
# Fixed parameters to be used. First the period:
P = 0.788839316
# Same for a/Rstar:
a = 3.4681372991
# Impact parameter:
b = 0.6777092890 
# Time of transit center:
t0 = 2456739.5472350949
# Eccentricity and omega:
ecc = 0.0046
omega = 3.0

# Now prior on Rp/Rs:
pmean,psd = 0.14,0.01

######################################
target,pfilename = datafile.split('/')
out_folder = 'outputs/'+datafile.split('.')[0]+'/wavelength'
out_ofolder = 'outputs/'+datafile.split('.')[0]
if not os.path.exists(out_folder):
    os.mkdir(out_folder)

data = pickle.load(open(datafile,'rb'))
nwbins = len(data['wbins'])
for wi in range(nwbins):
    # 0. Chech which comparisons are non-zero in this wavelength bin:
    comps = []
    for i in range(len(all_comps)):
        if np.mean(data['cLCw'][:,all_comps[i],wi]) != 0.:
            comps.append(all_comps[i])

    # 1. Save (mean-substracted) target and comparison lightcurves (in magnitude-space):
    if not os.path.exists(out_folder+'/wbin'+str(wi)):
        os.mkdir(out_folder+'/wbin'+str(wi))
        lcout = open(out_folder+'/wbin'+str(wi)+'/lc.dat','w')
        lccompout = open(out_folder+'/wbin'+str(wi)+'/comps.dat','w')
        for i in range(len(data['t'])):
            lcout.write('{0:.10f} {1:.10f} 0\n'.format(data['t'][i],-2.51*np.log10(data['oLCw'][i,wi])-np.median(-2.51*np.log10(data['oLCw'][:,wi]))))
            for j in range(len(comps)): 
                if j != len(comps)-1:
                    lccompout.write('{0:.10f} \t'.format(-2.51*np.log10(data['cLCw'][i,comps[j],wi]) - np.median(-2.51*np.log10(data['cLCw'][:,comps[j],wi]))))
                else:
                    lccompout.write('{0:.10f}\n'.format(-2.51*np.log10(data['cLCw'][i,comps[j],wi]) - np.median(-2.51*np.log10(data['cLCw'][:,comps[j],wi]))))
        lcout.close()
        lccompout.close() 

    # 2. Run code, BMA the posteriors, save:
    if not os.path.exists(out_folder+'/wbin'+str(wi)+'/BMA_posteriors.pkl'):
	lnZ = np.zeros(len(comps))
	nmin = np.inf
	for i in range(1,len(comps)+1): 
	    if not os.path.exists(out_folder+'/wbin'+str(wi)+'/PCA_'+str(i)):
		os.system('python GPTransitDetrendWavelength.py -outfolder '+out_folder+'/wbin'+str(wi)+'/ -compfile '+out_folder+\
			      '/wbin'+str(wi)+'/comps.dat -lcfile '+out_folder+'/wbin'+str(wi)+'/lc.dat -eparamfile '+out_ofolder+\
			      '/eparams.dat -ldlaw '+ld_law+' -P '+str(P)+' -a '+str(a)+' -pmean '+str(pmean)+' -psd '+str(psd)+' -b '+str(b)+' -t0 '+str(t0)+\
                              ' -ecc '+str(ecc)+' -omega '+str(omega)+' --PCA -pctouse '+str(i))
		os.mkdir(out_folder+'/wbin'+str(wi)+'/PCA_'+str(i))
		os.system('mv '+out_folder+'/wbin'+str(wi)+'/out* '+out_folder+'/wbin'+str(wi)+'/PCA_'+str(i)+'/.')
		os.system('mv '+out_folder+'/wbin'+str(wi)+'/*.pkl '+out_folder+'/wbin'+str(wi)+'/PCA_'+str(i)+'/.')
		os.system('mv detrended_lc.dat '+out_folder+'/wbin'+str(wi)+'/PCA_'+str(i)+'/.')
	    fin = open(out_folder+'/wbin'+str(wi)+'/PCA_'+str(i)+'/posteriors_trend_george.pkl','r')
	    posteriors = pickle.load(fin)
	    if len(posteriors['posterior_samples']['p'])<nmin:
		nmin = len(posteriors['posterior_samples']['p'])
	    lnZ[i-1] = posteriors['lnZ']
	    fin.close()
	# Calculate posterior probabilities of the models from the Bayes Factors:
	lnZ = lnZ - np.max(lnZ)
	Z = np.exp(lnZ)
	Pmodels = Z/np.sum(Z)
	# Prepare array that saves outputs:
	p = np.array([])
	q1 = np.array([])
	q2 = np.array([])
	jitter = np.array([])
	max_GPvariance = np.array([])
	alpha0 = np.array([])
	alpha1 = np.array([])
	alpha2 = np.array([])
	alpha3 = np.array([])
	alpha4 = np.array([])
	alpha5 = np.array([])
	mmean = np.array([])
	# With the number at hand, extract draws from the  posteriors with a fraction equal to the posterior probabilities to perform the 
	# model averaging scheme:
	for i in range(1,len(comps)+1):
	    fin = open(out_folder+'/wbin'+str(wi)+'/PCA_'+str(i)+'/posteriors_trend_george.pkl','r')
	    posteriors = pickle.load(fin)
	    fin.close()
	    nextract = int(Pmodels[i-1]*nmin)
	    idx_extract = np.random.choice(np.arange(len(posteriors['posterior_samples']['P'])),nextract,replace=False)
	    # Extract transit parameters:
	    p = np.append(p,posteriors['posterior_samples']['p'][idx_extract])
	    q1 = np.append(q1,posteriors['posterior_samples']['q1'][idx_extract])
            if ld_law != 'linear':
	        q2 = np.append(q2,posteriors['posterior_samples']['q2'][idx_extract])
	    # Note bayesian average posterior jitter saved is in mmag (MultiNest+george sample the log-variance, not the log-sigma):
	    jitter = np.append(jitter,np.sqrt(np.exp(posteriors['posterior_samples']['ljitter'][idx_extract])))
	    # Mean lightcurve in magnitude units:
	    mmean = np.append(mmean,posteriors['posterior_samples']['mmean'][idx_extract])
	    # Max GP variance:
	    max_GPvariance = np.append(max_GPvariance,posteriors['posterior_samples']['max_var'][idx_extract])
	    # Alphas:
	    alpha0 = np.append(alpha0,posteriors['posterior_samples']['alpha0'][idx_extract])
	    alpha1 = np.append(alpha1,posteriors['posterior_samples']['alpha1'][idx_extract])
	    alpha2 = np.append(alpha2,posteriors['posterior_samples']['alpha2'][idx_extract])
	    alpha3 = np.append(alpha3,posteriors['posterior_samples']['alpha3'][idx_extract])
	    alpha4 = np.append(alpha4,posteriors['posterior_samples']['alpha4'][idx_extract])
	    alpha5 = np.append(alpha5,posteriors['posterior_samples']['alpha5'][idx_extract])

	# Now save final BMA posteriors:
	out = {}
	out['p'] = p
        out['wbin'] = data['wbins'][wi]
	out['jitter'] = jitter
	out['q1'] = q1
        if ld_law != 'linear':
	    out['q2'] = q2
	out['mmean'] = mmean
	out['max_var'] = max_GPvariance
	out['alpha0'] = alpha0
	out['alpha1'] = alpha1
	out['alpha2'] = alpha2
	out['alpha3'] = alpha3
	out['alpha4'] = alpha4
	out['alpha5'] = alpha5
	pickle.dump(out,open(out_folder+'/wbin'+str(wi)+'/BMA_posteriors.pkl','wb'))
	fout = open(out_folder+'/wbin'+str(wi)+'/results.dat','w')
	fout.write('# Variable \t Value \t SigmaUp \t SigmaDown\n')
	for variable in out.keys():
	    v,vup,vdown = utils.get_quantiles(out[variable])
	    print variable
	    fout.write(variable+' \t {0:.10f} \t {1:.10f} \t {2:.10f}\n'.format(v,vup-v,v-vdown))
	fout.close()
    else:
	out = pickle.load(open(out_folder+'/wbin'+str(wi)+'/BMA_posteriors.pkl','rb'))
