import argparse
import numpy as np
from numpy import linalg
import scipy
from scipy import linalg
import timeit
from timeit import default_timer

sigma_0 = np.eye(2)
sigma_x = np.array([[0., 1.], [1., 0.]])
sigma_y = np.array([[0., -1j], [1j, 0.]])
sigma_z = np.array([[1., 0.], [0, -1.]])

xx = np.kron(sigma_x, sigma_x)
yy = np.kron(sigma_y, sigma_y)
zz = np.kron(sigma_z, sigma_z)

def xy_hamiltonian(n, gamma, lamb):
   coeff_x = (1. + gamma) / 2.
   coeff_y = (1. - gamma) / 2.

   # First two spins
   h = coeff_x * xx  + coeff_y * yy + lamb * np.kron(sigma_z, sigma_0) + lamb * np.kron(sigma_0, sigma_z)

   # Remaining spins
   for i in range(2, n):
      h = np.kron(h, sigma_0) + coeff_x * np.kron(np.eye(2**(i-1)), xx) + \
         coeff_y * np.kron(np.eye(2**(i-1)), yy) + lamb * np.kron(np.eye(2**i), sigma_z)

   # Wrap around
   if n > 2:
      h += coeff_x * np.kron(sigma_x, np.kron(np.eye(2**(n-2)), sigma_x)) + \
         coeff_y * np.kron(sigma_y, np.kron(np.eye(2**(n-2)), sigma_y))

   h *= -0.5
   return h

def xxz_hamiltonian(n, delta, lamb):
   # First two spins
   h = .5 * (xx  + yy + delta * zz) + lamb * np.kron(sigma_z, sigma_0) + lamb * np.kron(sigma_0, sigma_z)

   # Remaining spins
   for i in range(2, n):
      h = np.kron(h, sigma_0) + .5 * (np.kron(np.eye(2**(i-1)), xx) + np.kron(np.eye(2**(i-1)), yy) + \
         delta * np.kron(np.eye(2**(i-1)), zz)) + lamb * np.kron(np.eye(2**i), sigma_z)

   # Wrap around
   if n > 2:
      h += .5 * (np.kron(sigma_x, np.kron(np.eye(2**(n-2)), sigma_x)) + \
         np.kron(sigma_y, np.kron(np.eye(2**(n-2)), sigma_y)) + \
         delta * np.kron(sigma_z, np.kron(np.eye(2**(n-2)), sigma_z)))

   return h

def chaotic_hamiltonian(n, h_param, g_param):
   # First two spins
   h = zz + h_param * (np.kron(sigma_z, sigma_0) + np.kron(sigma_0, sigma_z)) + \
      g_param * (np.kron(sigma_x, sigma_0) + np.kron(sigma_0, sigma_x))

   # Remaining spins
   for i in range(2, n):
      h = np.kron(h, sigma_0) + np.kron(np.eye(2**(i-1)), zz) + \
         h_param * np.kron(np.eye(2**i), sigma_z) + g_param * np.kron(np.eye(2**i), sigma_x)

   # Wrap around
   if n > 2:
      h += np.kron(sigma_z, np.kron(np.eye(2**(n-2)), sigma_z))

   return h

def ising_hamiltonian(n, g):
   # First two spins
   h = -1. * zz + g * np.kron(sigma_x, sigma_0) + g * np.kron(sigma_0, sigma_x)

   # Remaining spins
   for i in range(2, n):
      h = np.kron(h, sigma_0) + -1. * np.kron(np.eye(2**(i-1)), zz) + g * np.kron(np.eye(2**i), sigma_x)

   # Wrap around
   if n > 2:
      h -= np.kron(sigma_z, np.kron(np.eye(2**(n-2)), sigma_z))

   return h

def cluster_hamiltonian(n):
   xz = np.kron(sigma_x, sigma_z)
   zx = np.kron(sigma_z, sigma_x)
   zxz = np.kron(sigma_z, xz)
   # First three
   h = 1. * zxz

   # Remaining spins
   for i in range(3, n):
      h = np.kron(h, sigma_0) + np.kron(np.eye(2**(i-2)), zxz)

   # Wrap around
   h += np.kron(sigma_z, np.kron(np.eye(2**(n-3)), zx)) + np.kron(xz, np.kron(np.eye(2**(n-3)), sigma_z))

   h *= -1.
   return h
   
def mg_hamiltonian(n, gamma):
   xi = np.kron(sigma_x, sigma_0)
   yi = np.kron(sigma_y, sigma_0)
   zi = np.kron(sigma_z, sigma_0)
   ix = np.kron(sigma_0, sigma_x)
   iy = np.kron(sigma_0, sigma_y)
   iz = np.kron(sigma_0, sigma_z)
   """
   xix = np.kron(xi, sigma_x)
   yiy = np.kron(yi, sigma_y)
   ziz = np.kron(zi, sigma_z)
   """
   xix = np.kron(sigma_x, ix)
   yiy = np.kron(sigma_y, iy)
   ziz = np.kron(sigma_z, iz)

   # First two spins
   h = gamma * (xx  + yy + zz)

   # Remaining spins
   for i in range(2, n):
      h = np.kron(h, sigma_0) + gamma * (np.kron(np.eye(2**(i-1)), xx) + np.kron(np.eye(2**(i-1)), yy) + \
         np.kron(np.eye(2**(i-1)), zz)) + gamma / 2. * (np.kron(np.eye(2**(i-2)), xix) + \
         np.kron(np.eye(2**(i-2)), yiy) + np.kron(np.eye(2**(i-2)), ziz))

   # Wrap around
   """
   h += gamma * (np.kron(sigma_x, np.kron(np.eye(2**(n-2)), sigma_x)) + np.kron(sigma_y, np.kron(np.eye(2**(n-2)), sigma_y)) + \
         np.kron(sigma_z, np.kron(np.eye(2**(n-2)), sigma_z))) + gamma / 2. * (np.kron(sigma_x, np.kron(np.eye(2**(n-3)), xi)) + \
         np.kron(ix, np.kron(np.eye(2**(n-3)), sigma_x)) + np.kron(sigma_y, np.kron(np.eye(2**(n-3)), yi)) + \
         np.kron(iy, np.kron(np.eye(2**(n-3)), sigma_y)) + np.kron(sigma_z, np.kron(np.eye(2**(n-3)), zi)) + \
         np.kron(iz, np.kron(np.eye(2**(n-3)), sigma_z)))
   """
   h += gamma * (np.kron(sigma_x, np.kron(np.eye(2**(n-2)), sigma_x)) + np.kron(sigma_y, np.kron(np.eye(2**(n-2)), sigma_y)) + \
         np.kron(sigma_z, np.kron(np.eye(2**(n-2)), sigma_z)))
   h += gamma / 2. * (np.kron(sigma_x, np.kron(np.eye(2**(n-3)), xi)) + np.kron(sigma_y, np.kron(np.eye(2**(n-3)), yi)) + \
         np.kron(sigma_z, np.kron(np.eye(2**(n-3)), zi)))
   h += gamma / 2. * (np.kron(sigma_0, np.kron(sigma_x, np.kron(np.eye(2**(n-3)), sigma_x))) + \
      np.kron(sigma_0, np.kron(sigma_y, np.kron(np.eye(2**(n-3)), sigma_y))) + \
      np.kron(sigma_0, np.kron(sigma_z, np.kron(np.eye(2**(n-3)), sigma_z))))

   return h


def chaotic_nnn_hamiltonian(n, gamma, ratio):
   zi = np.kron(sigma_z, sigma_0)
   iz = np.kron(sigma_0, sigma_z)
   ziz = np.kron(sigma_z, iz)

   # First two spins
   h = gamma * (xx  + yy + zz)

   # Remaining spins
   for i in range(2, n):
      h = np.kron(h, sigma_0) + gamma * (np.kron(np.eye(2**(i-1)), xx) + np.kron(np.eye(2**(i-1)), yy) + \
         np.kron(np.eye(2**(i-1)), zz)) + gamma * ratio * np.kron(np.eye(2**(i-2)), ziz)

   # Wrap around
   h += gamma * (np.kron(sigma_x, np.kron(np.eye(2**(n-2)), sigma_x)) + np.kron(sigma_y, np.kron(np.eye(2**(n-2)), sigma_y)) + \
         np.kron(sigma_z, np.kron(np.eye(2**(n-2)), sigma_z)))
   h += gamma * ratio * np.kron(sigma_z, np.kron(np.eye(2**(n-3)), zi))
   h += gamma * ratio * np.kron(sigma_0, np.kron(sigma_z, np.kron(np.eye(2**(n-3)), sigma_z)))

   return h

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--number", help="number of spins")
parser.add_argument("-m", "--hamiltonian", help="Hamiltonian type, xy or xxz")
parser.add_argument("-p", "--parameters", help="colon-separated parameters. for xy, gamma:lambda. for xxz, delta:lambda.")
args = parser.parse_args()

n = int(args.number)
h_type = args.hamiltonian
if args.parameters:
   h_params = args.parameters.split(":")

try:
   if n < 2:
      raise ValueError("n needs to be >= 2")
   if args.parameters:
      if len(h_params) != 2:
         raise ValueError("exactly two parameters need to be specified.")
except ValueError as exp:
   print("Error", exp)

if h_type =="xy":
   hamiltonian = xy_hamiltonian(n, float(h_params[0]), float(h_params[1]))
elif h_type == "xxz":
   hamiltonian = xxz_hamiltonian(n, float(h_params[0]), float(h_params[1]))
elif h_type=="chaotic":
   hamiltonian = chaotic_hamiltonian(n, float(h_params[0]), float(h_params[1]))
elif h_type=="ising":
   hamiltonian = ising_hamiltonian(n, float(h_params[0]))
elif h_type=="cluster":
   hamiltonian = cluster_hamiltonian(n)
elif h_type=="mg":
   hamiltonian = mg_hamiltonian(n, float(h_params[0]))
elif h_type=="chaotic_nnn":
   hamiltonian = chaotic_nnn_hamiltonian(n, float(h_params[0]), float(h_params[1]))
start_time = timeit.default_timer()
evals, evecs = np.linalg.eigh(hamiltonian)
tot_time = timeit.default_timer() - start_time

if args.parameters:
   np.savez(args.number + "_" + args.hamiltonian + "_" + args.parameters + ".npz", evals=evals, evecs=evecs)
else:
   np.savez(args.number + "_" + args.hamiltonian + ".npz", evals=evals, evecs=evecs)
print(tot_time)
