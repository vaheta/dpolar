import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as spio

pi = math.pi

# data = np.load("sphere_render/nmap_test_results.npz")

# phis = data['phis']
# thetas = data['thetas']
# ns = data['ns']

# plt.imshow(phis)
# plt.colorbar()
# plt.clim(-pi, pi)
# plt.show()

# plt.imshow(thetas)
# plt.colorbar()
# plt.clim(0,pi)
# plt.show()

# plt.imshow(ns)
# plt.colorbar()
# plt.clim(0,5)
# plt.show()

res = np.load("tempres.npy")
plt.imshow(res)
plt.colorbar()
plt.clim(0, 0.000000000001)
plt.show()
print (np.unravel_index(np.argmin(res, axis=None), res.shape))