import argparse
import math
import numpy as np
import scipy
from scipy import linalg
import timeit
from timeit import default_timer

M_aklt = np.zeros((2, 3, 2))
M_aklt[0, 0, 1] = math.sqrt(2. / 3.)
M_aklt[0, 1, 0] = -math.sqrt(1. / 3.)
M_aklt[1, 1, 1] = math.sqrt(1. / 3.)
M_aklt[1, 2, 0] = -math.sqrt(2. / 3.)

M_cluster = np.zeros((2, 4, 2))
M_cluster[0, 0, 0] = 1. / 2.
M_cluster[1, 0, 0] = 1. / 2.
M_cluster[0, 1, 1] = 1. / 2.
M_cluster[1, 1, 1] = 1. / 2.
M_cluster[0, 2, 0] = 1. / 2.
M_cluster[1, 2, 0] = -1. / 2.
M_cluster[0, 3, 1] = -1. / 2.
M_cluster[1, 3, 1] = 1. / 2.

M_ising = np.zeros((1, 2, 1))
M_ising[0, 0, 0] = 1 / math.sqrt(2)
M_ising[0, 1, 0] = -1 / math.sqrt(2)

M_mg = np.zeros((3, 2, 3))
M_mg[0, 0, 1] = 1.
M_mg[2, 0, 0] = 1. / math.sqrt(2)
M_mg[0, 1, 2] = 1.
M_mg[1, 1, 0] = -1. / math.sqrt(2)

def build_mps(M, n):
   psi = M
   for i in range(1, n):
      psi = np.tensordot(psi, M, 1) 
   psi = np.trace(psi, axis1=0, axis2=-1)
   return psi

def compute_entropies(psi, n, d):
   entropies = []
   # Iterate over possible lengths of block
   for i in range(1, int(n / 2) + 1):
      psi_reshape = np.matrix(psi.reshape((d**i, d**(n-i))))
      rho_a = psi_reshape * psi_reshape.getH()
      evals = scipy.linalg.eigh(rho_a, eigvals_only=True) 
      print(i)
      print(evals)
      entropy = sum(map(lambda x : 0. if np.isclose(x,0.) else -x * math.log(x, 2.), evals))
      entropies.append(entropy)
   return entropies

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--number", help="number of spins")
parser.add_argument("-m", "--mps_type", help="mps type, e.g. aklt")
parser.add_argument("-d", "--dimension", help="mps dimension if not using pre-defined type")
args = parser.parse_args()

n = int(args.number)
m_type = args.mps_type

try:
   if n < 2:
      raise ValueError("n needs to be >= 2")
except ValueError as exp:
   print("Error", exp)

if m_type =="aklt":
   M = M_aklt
   d = 3 
if m_type =="ising":
   M = M_ising
   d = 2 
if m_type =="cluster":
   M = M_cluster
   d = 4 
if m_type =="mg":
   M = M_mg
   d = 2 
start_time = timeit.default_timer()
psi = build_mps(M, n)
entropies = compute_entropies(psi, n, d)
tot_time = timeit.default_timer() - start_time

np.savez(args.number + "_" + args.mps_type + "_entropies.npz", entropies=entropies)
np.savez(args.number + "_" + args.mps_type + "_psi.npz", psi=psi)
print(entropies)
print(tot_time)
