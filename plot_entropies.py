import numpy as np
import matplotlib.pyplot as plt

#files = ["entropies_lastex_8_xxz_1:0.npz", "entropies_lastex_10_xxz_1:0.npz", "entropies_lastex_12_xxz_1:0.npz", "entropies_lastex_14_xxz_1:0.npz"]
files = ["entropies_18_xxz_1:0.npz"]
for filename in files:
   data = np.load(filename)
   entropies = data["entropies"]
   print(entropies)
   lens = np.arange(1, len(entropies) + 1)
   plt.scatter(lens, entropies)
plt.show()
      
