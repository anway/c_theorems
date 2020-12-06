import numpy as np
import matplotlib.pyplot as plt

#files = ["entropies_lastex_8_xxz_1:0.npz", "entropies_lastex_10_xxz_1:0.npz", "entropies_lastex_12_xxz_1:0.npz", "entropies_lastex_14_xxz_1:0.npz"]
files = ["mps/14_cluster_entropies.npz", "mps/entropies_gs_14_cluster.npz"]
#files = ["mps/8_aklt_entropies.npz", "mps/8_akltcheck_entropies_1.npz", "mps/8_akltcheck_entropies_2.npz"]
for filename in files:
   data = np.load(filename)
   entropies = data["entropies"]
   print(entropies)
   lens = np.arange(1, len(entropies) + 1)
   plt.scatter(lens, entropies, label=filename)
plt.legend()
plt.show()
      
