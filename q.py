import argparse
import subprocess
import wl_options_hp23b_180603 as c
import importlib

# This parses in the option file:
parser = argparse.ArgumentParser()
parser.add_argument("-ofile", default=None)
args = parser.parse_args()
ofile = args.ofile

# Read input file:
c = importlib.import_module(ofile)

job_name = c.datafile.split("/")[-1].split(".")[0]
if hasattr(c, "amean"):
    job_name += "_a"
else:
    job_name += "_rho"
if hasattr(c, "bmean"):
    job_name += "_bp"
else:
    job_name += "_r1r2"

print("job name: ", job_name)

# submit job script
command = "qsub -N {0} -o {0}.log wlc.job".format(job_name)
p = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
