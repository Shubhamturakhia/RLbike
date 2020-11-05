import matplotlib.pyplot as plt
import numpy as np

data = np.load ("Roll_angle.npy", allow_pickle =True)
plt.plot(data)
plt.show()

