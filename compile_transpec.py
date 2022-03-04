import numpy as np
import glob
import argparse
import utils
import pickle
import os
import importlib
import pandas as pd

parser = argparse.ArgumentParser()

# This parses in the option file:
parser.add_argument("-ofile", default=None)

args = parser.parse_args()
ofile = args.ofile

# Read input file:
c = importlib.import_module(ofile)
print("Loaded options for:", c.datafile)
######################################
datafile = c.datafile
target, pfilename = datafile.split("/")

outfold = c.out_folder_base
out_folder = ("%s/" + datafile.split(".")[0] + "/wavelength") % outfold
out_ofolder = ("%s/" + datafile.split(".")[0]) % outfold

print(out_folder)

wbinlist = glob.glob(out_folder + "/wbin*")
wis = np.array([])
for wbinfolders in wbinlist:
    wis = np.append(wis, int(wbinfolders.split("wbin")[-1]))
wis = wis[np.argsort(wis)]
wis = wis.astype(int)
data = []
columns = [
    "Wav_d",
    "Wav_u",
    "Rp/Rs",
    "Rp/RsErrUp",
    "Rp/RsErrDown",
    "Depth (ppm)",
    "Depthup (ppm)",
    "DepthDown (ppm)",
    "q1",
    "q1Up",
    "q1Down",
]
if c.ld_law != "linear":
    columns += ["q2", "q2Up", "q2Down"]
for wi in wis:
    print("Working on wbin ", wi, "...")
    try:
        out = pickle.load(
            open(out_folder + "/wbin" + str(wi) + "/BMA_posteriors.pkl", "rb"),
            encoding="latin1",
        )
        wbins = out["wbin"]
        p = out["p"]
        v, vup, vdown = utils.get_quantiles(p)
        vd, vupd, vdownd = utils.get_quantiles((p ** 2) * 1e6)
        vld, vupld, vdownld = utils.get_quantiles(out["q1"])
        if c.ld_law == "linear":
            try:
                data.append(
                    [
                        wbins[0],
                        wbins[1],
                        v,
                        vup - v,
                        v - vdown,
                        vd,
                        vupd - vd,
                        vd - vdownd,
                        vld,
                        vupld - vld,
                        vld - vdownld,
                    ]
                )
                # fout.write('{0:.2f} \t {1:.2f} \t {2:.10f} \t {3:.10f} \t {4:.10f} \t {5:.10f} \t {6:.10f} \t {7:.10f} \t {8:.10f} \t {9:.10f} \t {10:.10f} \n'.format(wbins[0],wbins[1],v,vup-v,v-vdown,vd,vupd-vd,vd-vdownd,vld,vupld-vld,vld-vdownld))
            except:
                data.append(
                    [
                        wbins,
                        wbins,
                        v,
                        vup - v,
                        v - vdown,
                        vd,
                        vupd - vd,
                        vd - vdownd,
                        vld,
                        vupld - vld,
                        vld - vdownld,
                    ]
                )
                # fout.write('{0:d} \t {1:d} \t {2:.10f} \t {3:.10f} \t {4:.10f} \t {5:.10f} \t {6:.10f} \t {7:.10f} \t {8:.10f} \t {9:.10f} \t {10:.10f} \n'.format(wbins,wbins,v,vup-v,v-vdown,vd,vupd-vd,vd-vdownd,vld,vupld-vld,vld-vdownld))
        else:
            vld2, vupld2, vdownld2 = utils.get_quantiles(out["q2"])
            try:
                data.append(
                    [
                        wbins[0],
                        wbins[1],
                        v,
                        vup - v,
                        v - vdown,
                        vd,
                        vupd - vd,
                        vd - vdownd,
                        vld,
                        vupld - vld,
                        vld - vdownld,
                        vld2,
                        vupld2 - vld2,
                        vld2 - vdownld2,
                    ]
                )
                # fout.write('{0:.2f} \t {1:.2f} \t {2:.10f} \t {3:.10f} \t {4:.10f} \t {5:.10f} \t {6:.10f} \t {7:.10f} \t {8:.10f} \t {9:.10f} \t {10:.10f} \t {11:.10f} \t {12:.10f} \t {13:.10f} \n'.format(wbins[0],wbins[1],v,vup-v,v-vdown,vd,vupd-vd,vd-vdownd,vld,vupld-vld,vld-vdownld,vld2,vupld2-vld2,vld2-vdownld2))
            except:
                data.append(
                    [
                        wbins,
                        wbins,
                        v,
                        vup - v,
                        v - vdown,
                        vd,
                        vupd - vd,
                        vd - vdownd,
                        vld,
                        vuApld - vld,
                        vld - vdownld,
                        vld2,
                        vupld2 - vld2,
                        vld2 - vdownld2,
                    ]
                )
                # fout.write('{0:d} \t {1:d} \t {2:.10f} \t {3:.10f} \t {4:.10f} \t {5:.10f} \t {6:.10f} \t {7:.10f} \t {8:.10f} \t {9:.10f} \t {10:.10f} \t {11:.10f} \t {12:.10f} \t {13:.10f}\n'.format(wbins,wbins,v,vup-v,v-vdown,vd,vupd-vd,vd-vdownd,vld,vupld-vld,vld-vdownld,vld2,vupld2-vld2,vld2-vdownld2))
    except:
        print("No data (yet)")

# Save to file
fname = "transpec.csv"
df = pd.DataFrame(data, columns=columns)
df.iloc[:, :2].apply(lambda x: round(x, 2))
df.iloc[:, 2:].apply(lambda x: round(x, 10))
fpath = f"{out_ofolder}/{fname}"
df.to_csv(fpath, index=False)
print(f"Saved to: {fpath}")
