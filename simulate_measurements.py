import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as spio

pi = math.pi

f = spio.loadmat('sphere_render/dm_and_N.mat', squeeze_me=True) 
dm = f['dm'] 

ymax, xmax = dm.shape

N = f['N']
phis = np.arctan2(N[:,:,1], N[:,:,0])
thetas = np.arccos(N[:,:,2])

env1 = np.load("sphere_render/ball_glacier.npy")
env2 = np.load("sphere_render/ball_grace_new.npy")

# setting incident light properties
Iups = env2
Ips = env1 

Iups[Iups>4] = 4
Iups = np.sqrt(np.sqrt(Iups))
Iups[np.isnan(dm)] = 0

Ips = np.sqrt(Ips)
Ips[np.isnan(dm)] = 0

psis = np.zeros((ymax, xmax))
psis = (Ips + np.sqrt(thetas))**2
psis = psis * pi / psis.max()

# setting ball properties n2 = n^2
n2 = 2.56

# simulating polarizer measurements
angles = [0, pi/6, pi/3, pi/2, 2*pi/3, 5*pi/6]
I = np.zeros((ymax, xmax, len(angles)))

xmap = np.zeros((200,200))
ymap = np.zeros((200,200))

for x in range(0, xmax):
    for y in range (0, ymax):
        Iup = Iups[y,x]
        Ip = Ips[y,x]
        ps = psis[y,x]
        sps2 = (math.sin(ps))**2
        cps2 = (math.cos(ps))**2
        tps = math.tan(ps)
        th = thetas[y,x]
        ph = phis[y,x]
        # if ph>0:
        #     Ip = 0.7
        #     ps = 0
        # else:
        #     Ip = 1.3
        #     ps = pi/2
        if (~np.isnan(dm[y,x])):
            thmap = int(math.floor(th*200/(pi/2+0.1)))
            phmap = int(math.floor((ph+pi)*200/(2*pi+0.1)))
            xmap[thmap,phmap] = x
            ymap[thmap,phmap] = y
        else:
            for i, angle in enumerate(angles):
                I[y,x,i] = 0
            continue        
        cth = math.cos(th)
        sth = math.sin(th)
        Rs = ((cth - math.sqrt(n2 - sth**2))/(cth + math.sqrt(n2 - sth**2)))**2
        Rp = ((-n2*cth + math.sqrt(n2 - sth**2))/(n2*cth + math.sqrt(n2 - sth**2)))**2

        for i, angle in enumerate(angles):
            I[y,x,i] = (Iup/2) * (Rs*(math.sin(ph - angle))**2 + Rp*(1 - (math.sin(ph - angle))**2)) + Ip*(Rp*cps2 + Rs*sps2)*(math.cos(math.atan(tps*math.sqrt(Rp/Rs)) + ph - angle))**2

np.savez ("sphere_render/meas_simulation.npz", I=I, Iups=Iups, Ips=Ips, psis=psis, phis=phis, thetas=thetas, dm=dm, xmap=xmap, ymap=ymap)

# plt.imshow(xmap)
# plt.colorbar()
# # plt.clim(0,0.5)
# plt.show()

# plt.imshow(ymap)
# plt.colorbar()
# # plt.clim(0,0.5)
# plt.show()