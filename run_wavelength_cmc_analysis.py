import numpy as np
import batman
import argparse
import utils
import pickle
import os

parser = argparse.ArgumentParser()

# This parses in the option file:
parser.add_argument('-ofile',default=None)
parser.add_argument('-wofile',default=None)
parser.add_argument('--nopickle', dest='nopickle', action='store_true')
parser.set_defaults(nopickle=False)
args = parser.parse_args()
ofile = args.ofile
wofile = args.wofile
nopickle = args.nopickle

# Read input file:
datafile, ld_law, idx_time, all_comps, P, Psd, \
a, asd, pmean, psd, b, bsd, t0,\
t0sd, fixed_eccentricity, ecc, eccsd, \
omega, omegasd = utils.read_optfile(ofile)

# Read white-light input file:
try:
    datafilew, ld_laww, idx_timew, compsw, Pmeanw, Psdw, \
    ameanw, asdw, pmeanw, psdw, bmeanw, bsdw, t0meanw,\
    t0sdw, fixed_eccentricityw, eccmeanw, eccsdw, \
    omegameanw, omegasdw = utils.read_optfile(wofile)
except:
    nlivew, datafilew, ld_laww, idx_timew, compsw, Pmeanw, Psdw, \
    ameanw, asdw, pmeanw, psdw, bmeanw, bsdw, t0meanw,\
    t0sdw, fixed_eccentricityw, eccmeanw, eccsdw, \
    omegameanw, omegasdw = utils.read_optfile(wofile)

######################################
target,pfilename = datafile.split('/')
out_folder = 'outputs/'+datafile.split('.')[0]+'/wavelength-cmc'
out_ofolder = 'outputs/'+datafile.split('.')[0]
if not os.path.exists(out_folder):
    os.mkdir(out_folder)
if not nopickle:
    data = pickle.load(open(datafile,'rb'))
    t,m1lc = np.loadtxt('outputs/'+datafile.split('.')[0]+'/white-light/lc.dat',unpack=True,usecols=(0,1))
    t = data['t']
    idx = np.where(~np.isnan(data['oLC']))[0]
    m1lc = -2.51*np.log10(data['oLC'])-np.median(-2.51*np.log10(data['oLC'][idx]))
    c1lc = -2.51*np.log10(data['cLC'][:,all_comps[0]])-np.median(-2.51*np.log10(data['cLC'][:,all_comps[0]]))
else:
    data = {}
    t,m1lc = np.loadtxt('outputs/'+datafile.split('.')[0]+'/white-light/lc.dat',unpack=True,usecols=(0,1))
    c1lc = np.loadtxt('outputs/'+datafile.split('.')[0]+'/white-light/comps.dat',unpack=True,usecols=(0))
    data['t'] = t
    import glob
    binfolders = glob.glob(out_folder+'/*') 
    data['wbins'] = np.arange(len(binfolders))
    data['oLCw'] = np.random.uniform(1,10,[3,len(binfolders)])

# Define transit-related functions:
def reverse_ld_coeffs(ld_law, q1, q2):
    if ld_law == 'quadratic':
        coeff1 = 2.*np.sqrt(q1)*q2
        coeff2 = np.sqrt(q1)*(1.-2.*q2)
    elif ld_law=='squareroot':
        coeff1 = np.sqrt(q1)*(1.-2.*q2)
        coeff2 = 2.*np.sqrt(q1)*q2
    elif ld_law=='logarithmic':
        coeff1 = 1.-np.sqrt(q1)*q2
        coeff2 = 1.-np.sqrt(q1)
    elif ld_law == 'linear':
        return q1,q2
    return coeff1,coeff2

def init_batman(t,law):
    """  
    This function initializes the batman code.
    """
    params = batman.TransitParams()
    params.t0 = 0.
    params.per = 1.
    params.rp = 0.1 
    params.a = 15. 
    params.inc = 87. 
    params.ecc = 0.
    params.w = 90. 
    if law == 'linear':
        params.u = [0.5]
    else:
        params.u = [0.1,0.3]
    params.limb_dark = law 
    m = batman.TransitModel(params,t)
    return params,m

def get_transit_model(t,t0,P,p,a,inc,q1,q2,ld_law):
    params,m = init_batman(t,law=ld_law)
    coeff1,coeff2 = reverse_ld_coeffs(ld_law, q1, q2) 
    params.t0 = t0
    params.per = P 
    params.rp = p 
    params.a = a 
    params.inc = inc 
    if ld_law == 'linear':
        params.u = [coeff1]
    else:
                params.u = [coeff1,coeff2]
    return m.light_curve(params)

# Generate white-light lightcurve, substract it from m1lc - c1lc to form the common mode signal:
posteriors = pickle.load(open('outputs/'+datafile.split('.')[0]+'/white-light/BMA_posteriors.pkl','r'))

white_light_lcmodel = get_transit_model(t.astype('float64'),np.median(posteriors['t0']),np.median(posteriors['P']),np.median(posteriors['p']),\
                                        np.median(posteriors['aR']),np.median(posteriors['inc']),np.median(posteriors['q1']),np.median(posteriors['q2']),ld_laww)

cmc = m1lc - c1lc + 2.51*np.log10(white_light_lcmodel)

# Generate idx_time, number of bins:
exec 'idx_time = np.arange(len(data["t"]))'+idx_time
nwbins = len(data['wbins'])
all_wbins = []
for wi in range(nwbins):
  if np.mean(data['oLCw'][:,wi]) != 0. and len(np.where(data['oLCw'][:,wi]<0)[0])<1 and not nopickle:
    # Check that the selected comparison star is non-zero. If not, go to next bin:
    if np.mean(data['cLCw'][:,all_comps[0],wi]) == 0.:
        continue
    all_wbins.append(wi)
    # 1. Save (mean-substracted), common-mode corrected comparison lightcurves (in magnitude-space):
    if not os.path.exists(out_folder+'/wbin'+str(wi)):
        os.mkdir(out_folder+'/wbin'+str(wi))
        lcout = open(out_folder+'/wbin'+str(wi)+'/lc.dat','w')
        lcoutcmc = open(out_folder+'/wbin'+str(wi)+'/lc_cmc.dat','w')
        lccompout = open(out_folder+'/wbin'+str(wi)+'/comps.dat','w')
        for i in idx_time:
            lcout.write('{0:.10f} {1:.10f} 0\n'.format(data['t'][i],-2.51*np.log10(data['oLCw'][i,wi])-np.median(-2.51*np.log10(data['oLCw'][idx_time,wi]))))
            lcoutcmc.write('{0:.10f} {1:.10f} 0\n'.format(data['t'][i],-2.51*np.log10(data['oLCw'][i,wi])-np.median(-2.51*np.log10(data['oLCw'][idx_time,wi])) - cmc[i] -
                                                       (-2.51*np.log10(data['cLCw'][i,all_comps[0],wi]) - np.median(-2.51*np.log10(data['cLCw'][idx_time,all_comps[0],wi])))))
            lccompout.write('{0:.10f} \n'.format(-2.51*np.log10(data['cLCw'][i,all_comps[0],wi]) - np.median(-2.51*np.log10(data['cLCw'][idx_time,all_comps[0],wi]))))
        lcout.close()
        lcoutcmc.close()
        lccompout.close() 

for wi in all_wbins:
    print 'Working on wbin ',wi,'...'
    # 1. Run code, BMA the posteriors, save:
    if not os.path.exists(out_folder+'/wbin'+str(wi)+'/BMA_posteriors.pkl'):
	lnZ = np.zeros(1)
	nmin = np.inf
	for i in range(1,2):
	    if not os.path.exists(out_folder+'/wbin'+str(wi)+'/PCA_'+str(i)):
		os.system('python GPTransitDetrendWavelength.py -outfolder '+out_folder+'/wbin'+str(wi)+'/ -lcfile '+out_folder+'/wbin'+str(wi)+'/lc_cmc.dat -eparamfile '+\
                              out_ofolder+'/eparams.dat -ldlaw '+ld_law+' -P '+str(P)+' -a '+str(a)+' -pmean '+str(pmean)+' -psd '+str(psd)+' -b '+str(b)+' -t0 '+str(t0)+\
                              ' -ecc '+str(ecc)+' -omega '+str(omega))
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
        # Check how many alphas were fitted:
        acounter = 0
        for vrs in posteriors['posterior_samples'].keys():
            if 'alpha' in vrs:
                exec 'alpha'+str(acounter)+' = np.array([])'
                acounter = acounter + 1
	mmean = np.array([])
	# With the number at hand, extract draws from the  posteriors with a fraction equal to the posterior probabilities to perform the 
	# model averaging scheme:
	for i in range(1,2):
	    fin = open(out_folder+'/wbin'+str(wi)+'/PCA_'+str(i)+'/posteriors_trend_george.pkl','r')
	    posteriors = pickle.load(fin)
	    fin.close()
	    nextract = int(Pmodels[i-1]*nmin)
	    idx_extract = np.random.choice(np.arange(len(posteriors['posterior_samples']['p'])),nextract,replace=False)
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
            for ai in range(acounter):
                exec "alpha"+str(ai)+" = np.append(alpha"+str(ai)+",posteriors['posterior_samples']['alpha"+str(ai)+"'][idx_extract])"

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
        for ai in range(acounter):
            exec "out['alpha"+str(ai)+"'] = alpha"+str(ai)
	pickle.dump(out,open(out_folder+'/wbin'+str(wi)+'/BMA_posteriors.pkl','wb'))
	fout = open(out_folder+'/wbin'+str(wi)+'/results.dat','w')
	fout.write('# Variable \t Value \t SigmaUp \t SigmaDown\n')
	for variable in out.keys():
            if variable != 'wbin':
	        v,vup,vdown = utils.get_quantiles(out[variable])
	        fout.write(variable+' \t {0:.10f} \t {1:.10f} \t {2:.10f}\n'.format(v,vup-v,v-vdown))
	fout.close()
    else:
	out = pickle.load(open(out_folder+'/wbin'+str(wi)+'/BMA_posteriors.pkl','rb'))
