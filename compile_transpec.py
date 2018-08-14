import numpy as np
import glob
import argparse
import utils
import pickle
import os

parser = argparse.ArgumentParser()

# This parses in the option file:
parser.add_argument('-ofile',default=None)
args = parser.parse_args()
ofile = args.ofile

# Read input file:
datafile, ld_law, idx_time, all_comps, P, Psd, \
a, asd, pmean, psd, b, bsd, t0,\
t0sd, fixed_eccentricity, ecc, eccsd, \
omega, omegasd = utils.read_optfile(ofile)

######################################
target,pfilename = datafile.split('/')
out_folder = 'outputs/'+datafile.split('.')[0]+'/wavelength'
out_ofolder = 'outputs/'+datafile.split('.')[0]

fout = open(out_ofolder+'/transpec.dat','w')
fout.write('# Wav (Angstroms) \t Rp/Rs \t Rp/RsErrUp \t Rp/RsErrDown \t Depth (ppm) \t Depthup (ppm) \t DepthDown (ppm)\n')
wbinlist = glob.glob(out_folder+'/wbin*')
wis = np.array([])
for wbinfolders in wbinlist:
    wis = np.append(wis,int(wbinfolders.split('wbin')[-1]))
wis = wis[np.argsort(wis)]
wis = wis.astype(int)
for wi in wis:
    print 'Working on wbin ',wi,'...'
    try:
        out = pickle.load(open(out_folder+'/wbin'+str(wi)+'/BMA_posteriors.pkl','rb'))
        wbins = out['wbin']
        p = out['p']
        v,vup,vdown = utils.get_quantiles(p)
        vd,vupd,vdownd = utils.get_quantiles((p**2)*1e6)
        vld,vupld,vdownld = utils.get_quantiles(out['q1'])
        if ld_law == 'linear':
            try:
                fout.write('{0:.2f} \t {1:.2f} \t {2:.10f} \t {3:.10f} \t {4:.10f} \t {5:.10f} \t {6:.10f} \t {7:.10f} \t {8:.10f} \t {9:.10f} \t {10:.10f} \n'.format(wbins[0],wbins[1],v,vup-v,v-vdown,vd,vupd-vd,vd-vdownd,vld,vupld-vld,vld-vdownld))
            except:
                fout.write('{0:d} \t {1:d} \t {2:.10f} \t {3:.10f} \t {4:.10f} \t {5:.10f} \t {6:.10f} \t {7:.10f} \t {8:.10f} \t {9:.10f} \t {10:.10f} \n'.format(wbins,wbins,v,vup-v,v-vdown,vd,vupd-vd,vd-vdownd,vld,vupld-vld,vld-vdownld))
        else:
            vld2,vupld2,vdownld2 = utils.get_quantiles(out['q2'])
            try:
                fout.write('{0:.2f} \t {1:.2f} \t {2:.10f} \t {3:.10f} \t {4:.10f} \t {5:.10f} \t {6:.10f} \t {7:.10f} \t {8:.10f} \t {9:.10f} \t {10:.10f} \t {11:.10f} \t {12:.10f} \t {13:.10f} \n'.format(wbins[0],wbins[1],v,vup-v,v-vdown,vd,vupd-vd,vd-vdownd,vld,vupld-vld,vld-vdownld,vld2,vupld2-vld2,vld2-vdownld2))
            except:
                fout.write('{0:d} \t {1:d} \t {2:.10f} \t {3:.10f} \t {4:.10f} \t {5:.10f} \t {6:.10f} \t {7:.10f} \t {8:.10f} \t {9:.10f} \t {10:.10f} \t {11:.10f} \t {12:.10f} \t {13:.10f}\n'.format(wbins,wbins,v,vup-v,v-vdown,vd,vupd-vd,vd-vdownd,vld,vupld-vld,vld-vdownld,vld2,vupld2-vld2,vld2-vdownld2))
    except:
        print 'No data (yet)'
fout.close()
