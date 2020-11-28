import argparse
import numpy as np
import scipy
from scipy import linalg
from scipy import sparse
from scipy.sparse import linalg
import timeit
from timeit import default_timer

sigma_0 = sparse.identity(2)
sigma_x = sparse.csr_matrix(np.array([[0., 1.], [1., 0.]]))
sigma_y = sparse.csr_matrix(np.array([[0., -1j], [1j, 0.]]))
sigma_z = sparse.csr_matrix(np.array([[1., 0.], [0, -1.]]))

xx = sparse.kron(sigma_x, sigma_x)
yy = sparse.kron(sigma_y, sigma_y)
zz = sparse.kron(sigma_z, sigma_z)

def xy_hamiltonian(n, gamma, lamb):
   coeff_x = (1. + gamma) / 2.
   coeff_y = (1. - gamma) / 2.

   # First two spins
   h = coeff_x * xx  + coeff_y * yy + lamb * sparse.kron(sigma_z, sigma_0) + lamb * sparse.kron(sigma_0, sigma_z)

   # Remaining spins
   for i in range(2, n):
      h = sparse.kron(h, sigma_0) + coeff_x * sparse.kron(sparse.identity(2**(i-1)), xx) + \
         coeff_y * sparse.kron(sparse.identity(2**(i-1)), yy) + lamb * sparse.kron(sparse.identity(2**i), sigma_z)

   """
   # Wrap around, uncomment for ring rather than chain
   if n > 2:
      h += coeff_x * sparse.kron(sigma_x, sparse.kron(sparse.identity(2**(n-2)), sigma_x)) + \
         coeff_y * sparse.kron(sigma_y, sparse.kron(sparse.identity(2**(n-2)), sigma_y))
   """

   h *= -0.5
   return h

def xxz_hamiltonian(n, delta, lamb):
   # First two spins
   h = .5 * (xx  + yy + delta * zz) + lamb * sparse.kron(sigma_z, sigma_0) + lamb * sparse.kron(sigma_0, sigma_z)

   # Remaining spins
   for i in range(2, n):
      h = sparse.kron(h, sigma_0) + .5 * (sparse.kron(sparse.identity(2**(i-1)), xx) + sparse.kron(sparse.identity(2**(i-1)), yy) + \
         delta * sparse.kron(sparse.identity(2**(i-1)), zz)) + lamb * sparse.kron(sparse.identity(2**i), sigma_z)

   """
   # Wrap around, uncomment for ring rather than chain
   if n > 2:
      h += .5 * (sparse.kron(sigma_x, sparse.kron(sparse.identity(2**(n-2)), sigma_x)) + \
         sparse.kron(sigma_y, sparse.kron(sparse.identity(2**(n-2)), sigma_y)) + \
         delta * sparse.kron(sigma_z, sparse.kron(sparse.identity(2**(n-2)), sigma_z)))
   """

   return h

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--number", help="number of spins")
parser.add_argument("-m", "--hamiltonian", help="Hamiltonian type, xy or xxz")
parser.add_argument("-p", "--parameters", help="colon-separated parameters. for xy, gamma:lambda. for xxz, delta:lambda.")
args = parser.parse_args()

n = int(args.number)
h_type = args.hamiltonian
h_params = args.parameters.split(":")

try:
   if n < 2:
      raise ValueError("n needs to be >= 2")
   if len(h_params) != 2:
      raise ValueError("exactly two parameters need to be specified.")
   if h_type != "xy" and h_type != "xxz":
      raise ValueError("Hamiltonian type needs to be either xy or xxz.")
except ValueError as exp:
   print("Error", exp)

if h_type =="xy":
   hamiltonian = xy_hamiltonian(n, float(h_params[0]), float(h_params[1]))
elif h_type == "xxz":
   hamiltonian = xxz_hamiltonian(n, float(h_params[0]), float(h_params[1]))
start_time = timeit.default_timer()
evals, evecs = sparse.linalg.eigsh(hamiltonian, which='SA')
tot_time = timeit.default_timer() - start_time

#np.savez(args.number + "_" + args.hamiltonian + "_" + args.parameters + ".npz", evals=evals, evecs=np.transpose(evecs))
print(tot_time)

print(evals)
print(np.transpose(evecs))
