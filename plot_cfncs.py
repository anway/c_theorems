import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

#files = ["entropies_lastex_8_xxz_1:0.npz", "entropies_lastex_10_xxz_1:0.npz", "entropies_lastex_12_xxz_1:0.npz", "entropies_lastex_14_xxz_1:0.npz"]
files = ["entropies_18_xxz_1:0.npz"]
for filename in files:
   data = np.load(filename)
   entropies = np.insert(data["entropies"], 0, 0.)
   delta_entropies = [np.log2((entropies[i] - entropies[i-1]) * i) for i in range(1, len(entropies))]
   lens = np.arange(1, len(delta_entropies) + 1)
   plt.scatter(lens, delta_entropies)
   slope, intercept, r_val, p_val, stderr = linregress(lens[2:5], delta_entropies[2:5])
   print(slope)
plt.show()
      
