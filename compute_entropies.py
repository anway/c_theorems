import argparse
import math
import numpy as np
import scipy
from scipy import linalg
import timeit
from timeit import default_timer

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--filename", help="file name")
args = parser.parse_args()

entropies = []

data = np.load(args.filename)
evals = data["evals"]
evecs = data["evecs"]

# Assume we are interested in ground state entropies
minind = np.argmin(evals)
ground_state = np.matrix(evecs[minind])
n = int(math.log(len(evecs[minind]), 2))

start_time = timeit.default_timer()
# Iterate over possible lengths of block
for i in range(1, int(n / 2) + 1):
   rho = np.outer(ground_state, ground_state.getH())
   rho_ab = rho.reshape([2**i, 2**(n - i), 2**i, 2**(n - i)])
   rho_a = np.trace(rho_ab, axis1=1, axis2=3)
   evals = scipy.linalg.eigh(rho_a, eigvals_only=True) 
   entropy = sum(map(lambda x : 0. if np.isclose(x,0.) else -x * math.log(x, 2.), evals))
   entropies.append(entropy)
tot_time = timeit.default_timer() - start_time

np.savez("entropies_" + args.filename, entropies=entropies)
#print(entropies) 
print(tot_time)
