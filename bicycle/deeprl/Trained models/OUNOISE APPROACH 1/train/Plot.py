import matplotlib.pyplot as plt
import numpy as np

data = np.load ("term1.npy")
plt.plot(data)
#plt.xlabel("No of episodes")
#plt.ylabel("Reward term (-ω^2)")
plt.show()

