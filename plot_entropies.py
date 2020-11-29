import numpy as np
import matplotlib.pyplot as plt

files = ["entropies_8_xxz_1:0.npz", "entropies_10_xxz_1:0.npz", "entropies_12_xxz_1:0.npz", "entropies_14_xxz_1:0.npz", "entropies_16_xxz_1:0.npz", "entropies_18_xxz_1:0.npz"]
for filename in files:
   data = np.load(filename)
   entropies = data["entropies"]
   print(entropies)
   lens = np.arange(1, len(entropies) + 1)
   plt.scatter(lens, entropies)
plt.show()
      
