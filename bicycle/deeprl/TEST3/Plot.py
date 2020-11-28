import matplotlib.pyplot as plt
import numpy as np

data = np.load ("Steer_angle.npy", allow_pickle =True)
plt.plot(data)
plt.show()

