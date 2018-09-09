from scipy.stats import gamma,norm,beta,truncnorm
import numpy as np

# READ OPTION FILE:
def read_optfile(fname):
    fin = open(fname,'r')
    x = fin.read()
    data = x.split('\n')
    out = {}
    for i in range(len(data)):
        if data[i] != '':   
          if data[i][0] != '#': 
             variable = data[i].split(':')[0].split()[0]
             if variable.lower() == 'datafile':
                 out['datafile'] = data[i].split(':')[-1].split()[0]
             elif variable.lower() == 'ld_law':
                 out['ld_law'] = data[i].split(':')[-1].split()[0]
             elif variable.lower() == 'idx_time':
                 out['idx_time'] = ':'.join(data[i].split(':')[1:]).split()[0]
             elif variable.lower() == 'fixed_eccentricity':
                 if data[i].split(':')[-1].split()[0].lower() == 'false':
                     out['fixed_eccentricity'] = False
                 else:
                     out['fixed_eccentricity'] = True
             elif variable.lower() == 'comps':
                 vals = data[i].split(':')[-1].split()[0].split(',')
                 out['comps'] = list(np.array(vals).astype(int))
             else:
                 variables = variable.split(',')
                 if len(variables) == 1:
                     out[variable] = np.double(data[i].split(':')[-1].split()[0])
                     out[variable.split('mean')[0]+'sd'] = 0.0
                 else:
                     values = data[i].split(':')[-1].split()[0].split(',')
                     out[variables[0]] = np.double(values[0])
                     out[variables[1]] = np.double(values[1])
        else:
            break
    return out['datafile'], out['ld_law'], out['idx_time'], out['comps'], out['Pmean'], out['Psd'], \
    out['amean'], out['asd'], out['pmean'], out['psd'], out['bmean'], out['bsd'], out['t0mean'],\
    out['t0sd'], out['fixed_eccentricity'], out['eccmean'], out['eccsd'], \
    out['omegamean'], out['omegasd']

# TRANSFORMATION OF PRIORS:
def transform_uniform(x,a,b):
    return a + (b-a)*x

def transform_loguniform(x,a,b):
    la=np.log(a)
    lb=np.log(b)
    return np.exp(la + x*(lb-la))

def transform_normal(x,mu,sigma):
    return norm.ppf(x,loc=mu,scale=sigma)

def transform_beta(x,a,b):
    return beta.ppf(x,a,b)

def transform_exponential(x,a=1.):
    return gamma.ppf(x, a)

def transform_truncated_normal(x,mu,sigma,a=0.,b=1.):
    ar, br = (a - mu) / sigma, (b - mu) / sigma
    return truncnorm.ppf(x,ar,br,loc=mu,scale=sigma)

# PCA TOOLS:
def get_sigma(x):
    """
    This function returns the MAD-based standard-deviation.
    """
    median = np.median(x)
    mad = np.median(np.abs(x-median))
    return 1.4826*mad

def standarize_data(input_data):
    output_data = np.copy(input_data)
    averages = np.median(input_data,axis=1)
    for i in range(len(averages)):
        sigma = get_sigma(output_data[i,:])
        output_data[i,:] = output_data[i,:] - averages[i]
        output_data[i,:] = output_data[i,:]/sigma
    return output_data

def classic_PCA(Input_Data, standarize = True):
    """  
    classic_PCA function

    Description

    This function performs the classic Principal Component Analysis on a given dataset.
    """
    if standarize:
        Data = standarize_data(Input_Data)
    else:
        Data = np.copy(Input_Data)
    eigenvectors_cols,eigenvalues,eigenvectors_rows = np.linalg.svd(np.cov(Data))
    idx = eigenvalues.argsort()
    eigenvalues = eigenvalues[idx[::-1]]
    eigenvectors_cols = eigenvectors_cols[:,idx[::-1]]
    eigenvectors_rows = eigenvectors_rows[idx[::-1],:]
    # Return: V matrix, eigenvalues and the principal components.
    return eigenvectors_rows,eigenvalues,np.dot(eigenvectors_rows,Data)

# Post-processing tools:
def mag_to_flux(m,merr):
    """ 
    Convert magnitude to relative fluxes. 
    """
    fluxes = np.zeros(len(m))
    fluxes_err = np.zeros(len(m))
    for i in range(len(m)):
        dist = 10**(-np.random.normal(m[i],merr[i],1000)/2.51)
        fluxes[i] = np.mean(dist)
        fluxes_err[i] = np.sqrt(np.var(dist))
    return fluxes,fluxes_err

def get_quantiles(dist,alpha = 0.68, method = 'median'):
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
    nsamples_at_each_side = int(nsamples*(alpha/2.)+1)
    if(method == 'median'):
       med_idx = 0 
       if(nsamples%2 == 0.0): # Number of points is even
          med_idx_up = int(nsamples/2.)+1
          med_idx_down = med_idx_up-1
          param = (ordered_dist[med_idx_up]+ordered_dist[med_idx_down])/2.
          return param,ordered_dist[med_idx_up+nsamples_at_each_side],\
                 ordered_dist[med_idx_down-nsamples_at_each_side]
       else:
          med_idx = int(nsamples/2.)
          param = ordered_dist[med_idx]
          return param,ordered_dist[med_idx+nsamples_at_each_side],\
                 ordered_dist[med_idx-nsamples_at_each_side]

def ConvertToInt(string, length): #Made to use the idx_time parameter
    #To take out front and back brackets
    string = string[1:-1]
    if ':' in string: #means that it's a sliced list so only 2 numbers
        a = range(length)
        ColPos = string.index(':')
        firstNum = string[:ColPos]
        lastNum = string[ColPos+1:]
        integer = a[int(firstNum):int(lastNum)]
    else: #must be a list containing all the integers that are useful
        number_string = string.split(',')
        integer = []
        for s in number_string:
            integer.append(int(s))
    return integer