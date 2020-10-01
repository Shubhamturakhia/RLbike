import matplotlib.pyplot as plt
import numpy as np

data = np.load ("total_reward.npy")
plt.plot(data)
plt.show()