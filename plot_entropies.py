import numpy as np
import matplotlib.pyplot as plt

#files = ["entropies_lastex_8_xxz_1:0.npz", "entropies_lastex_10_xxz_1:0.npz", "entropies_lastex_12_xxz_1:0.npz", "entropies_lastex_14_xxz_1:0.npz"]
#files = ["mps/14_mg_entropies.npz"]
#files = ["mps/8_aklt_entropies.npz", "mps/8_akltcheck_entropies_1.npz", "mps/8_akltcheck_entropies_2.npz"]
#files = ["mps/10_random_2:5_entropies.npz", "mps/10_random_4:5_entropies.npz", "mps/10_random_6:5_entropies.npz", "mps/10_random_8:5_entropies.npz"]
#files = ["mps/10_random_8:5_entropies.npz", "mps/10_random_10:5_entropies.npz", "mps/10_random_12:5_entropies.npz", "mps/10_random_14:5_entropies.npz"]
files = ["mps/18_aklt_entropies.npz", "mps/18_mg_entropies.npz"]
for filename in files:
   data = np.load(filename)
   entropies = data["entropies"]
   print(entropies)
   lens = np.arange(1, len(entropies) + 1)
   plt.scatter(lens, entropies, label=filename)
plt.legend()
plt.show()
      
