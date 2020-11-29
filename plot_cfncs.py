import numpy as np
import matplotlib.pyplot as plt

files = ["entropies_8_xxz_1:0.npz", "entropies_10_xxz_1:0.npz", "entropies_12_xxz_1:0.npz", "entropies_14_xxz_1:0.npz", "entropies_16_xxz_1:0.npz", "entropies_18_xxz_1:0.npz"]
for filename in files:
   data = np.load(filename)
   entropies = [0.] + data["entropies"]
   delta_entropies = [(entropies[i] - entropies[i-1]) / i for i in range(1, len(entropies))]
   lens = np.arange(1, len(delta_entropies) + 1)
   plt.scatter(lens, delta_entropies)
plt.show()
      
