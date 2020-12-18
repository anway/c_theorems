import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

#files = ["entropies_lastex_8_xxz_1:0.npz", "entropies_lastex_10_xxz_1:0.npz", "entropies_lastex_12_xxz_1:0.npz", "entropies_lastex_14_xxz_1:0.npz"]
#files = ["entropies_18_xxz_1:0.npz"]
#files = ["mps/14_mg_entropies.npz"]
#files = ["mps/10_random_7-2_1_entropies.npz", "mps/10_random_7-2_2_entropies.npz", "mps/10_random_7-2_3_entropies.npz"]
#files = ["mps/10_random_2:5_entropies.npz", "mps/10_random_4:5_entropies.npz", "mps/10_random_6:5_entropies.npz", "mps/10_random_8:5_entropies.npz"]
#files = ["mps/10_random_8:5_entropies.npz", "mps/10_random_10:5_entropies.npz", "mps/10_random_12:5_entropies.npz", "mps/10_random_14:5_entropies.npz"]
files = ["mps/18_aklt_entropies.npz", "mps/18_mg_entropies.npz"]
for filename in files:
   data = np.load(filename)
   entropies = np.insert(data["entropies"], 0, 0.)
   delta_entropies = [np.log((entropies[i] - entropies[i-1]) * i) for i in range(1, len(entropies))]
   lens = np.arange(1, len(delta_entropies) + 1)
   plt.scatter(lens, delta_entropies, label=filename)
   print(delta_entropies)
   #slope, intercept, r_val, p_val, stderr = linregress(lens[2:5], delta_entropies[2:5])
   #print(slope)
plt.legend()
plt.show()
      
