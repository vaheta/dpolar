import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as spio
from scipy.optimize import fsolve, root

pi = math.pi

def pied (x):
    while x>pi:
        x -= pi
    return (x)

def system (p, *args):
    angles = [0, pi/6, pi/3, pi/2, 2*pi/3, 5*pi/6]
    ph, th, n = p
    Iup, Ip, ps, I = args
    return (
        (Iup/2) * (((((math.cos(th)) - math.sqrt(n**2 - (math.sin(th))**2))/((math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2)))**2)*(math.sin(ph - angles[0]))**2 + (((-n**2*(math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2))/(n**2*(math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2)))**2)*(1 - (math.sin(ph - angles[0]))**2)) + Ip*((((-n**2*(math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2))/(n**2*(math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2)))**2)*((math.cos(ps))**2) + ((((math.cos(th)) - math.sqrt(n**2 - (math.sin(th))**2))/((math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2)))**2)*((math.sin(ps))**2))*(math.cos(math.atan((math.tan(ps))*math.sqrt((((-n**2*(math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2))/(n**2*(math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2)))**2)/((((math.cos(th)) - math.sqrt(n**2 - (math.sin(th))**2))/((math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2)))**2))) + ph - angles[0]))**2 - I[0],
        (Iup/2) * (((((math.cos(th)) - math.sqrt(n**2 - (math.sin(th))**2))/((math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2)))**2)*(math.sin(ph - angles[1]))**2 + (((-n**2*(math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2))/(n**2*(math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2)))**2)*(1 - (math.sin(ph - angles[1]))**2)) + Ip*((((-n**2*(math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2))/(n**2*(math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2)))**2)*((math.cos(ps))**2) + ((((math.cos(th)) - math.sqrt(n**2 - (math.sin(th))**2))/((math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2)))**2)*((math.sin(ps))**2))*(math.cos(math.atan((math.tan(ps))*math.sqrt((((-n**2*(math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2))/(n**2*(math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2)))**2)/((((math.cos(th)) - math.sqrt(n**2 - (math.sin(th))**2))/((math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2)))**2))) + ph - angles[1]))**2 - I[1],
        (Iup/2) * (((((math.cos(th)) - math.sqrt(n**2 - (math.sin(th))**2))/((math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2)))**2)*(math.sin(ph - angles[2]))**2 + (((-n**2*(math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2))/(n**2*(math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2)))**2)*(1 - (math.sin(ph - angles[2]))**2)) + Ip*((((-n**2*(math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2))/(n**2*(math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2)))**2)*((math.cos(ps))**2) + ((((math.cos(th)) - math.sqrt(n**2 - (math.sin(th))**2))/((math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2)))**2)*((math.sin(ps))**2))*(math.cos(math.atan((math.tan(ps))*math.sqrt((((-n**2*(math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2))/(n**2*(math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2)))**2)/((((math.cos(th)) - math.sqrt(n**2 - (math.sin(th))**2))/((math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2)))**2))) + ph - angles[2]))**2 - I[2],
        (Iup/2) * (((((math.cos(th)) - math.sqrt(n**2 - (math.sin(th))**2))/((math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2)))**2)*(math.sin(ph - angles[3]))**2 + (((-n**2*(math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2))/(n**2*(math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2)))**2)*(1 - (math.sin(ph - angles[3]))**2)) + Ip*((((-n**2*(math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2))/(n**2*(math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2)))**2)*((math.cos(ps))**2) + ((((math.cos(th)) - math.sqrt(n**2 - (math.sin(th))**2))/((math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2)))**2)*((math.sin(ps))**2))*(math.cos(math.atan((math.tan(ps))*math.sqrt((((-n**2*(math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2))/(n**2*(math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2)))**2)/((((math.cos(th)) - math.sqrt(n**2 - (math.sin(th))**2))/((math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2)))**2))) + ph - angles[3]))**2 - I[3],
        (Iup/2) * (((((math.cos(th)) - math.sqrt(n**2 - (math.sin(th))**2))/((math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2)))**2)*(math.sin(ph - angles[4]))**2 + (((-n**2*(math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2))/(n**2*(math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2)))**2)*(1 - (math.sin(ph - angles[4]))**2)) + Ip*((((-n**2*(math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2))/(n**2*(math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2)))**2)*((math.cos(ps))**2) + ((((math.cos(th)) - math.sqrt(n**2 - (math.sin(th))**2))/((math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2)))**2)*((math.sin(ps))**2))*(math.cos(math.atan((math.tan(ps))*math.sqrt((((-n**2*(math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2))/(n**2*(math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2)))**2)/((((math.cos(th)) - math.sqrt(n**2 - (math.sin(th))**2))/((math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2)))**2))) + ph - angles[4]))**2 - I[4],
        (Iup/2) * (((((math.cos(th)) - math.sqrt(n**2 - (math.sin(th))**2))/((math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2)))**2)*(math.sin(ph - angles[5]))**2 + (((-n**2*(math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2))/(n**2*(math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2)))**2)*(1 - (math.sin(ph - angles[5]))**2)) + Ip*((((-n**2*(math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2))/(n**2*(math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2)))**2)*((math.cos(ps))**2) + ((((math.cos(th)) - math.sqrt(n**2 - (math.sin(th))**2))/((math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2)))**2)*((math.sin(ps))**2))*(math.cos(math.atan((math.tan(ps))*math.sqrt((((-n**2*(math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2))/(n**2*(math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2)))**2)/((((math.cos(th)) - math.sqrt(n**2 - (math.sin(th))**2))/((math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2)))**2))) + ph - angles[5]))**2 - I[5]
    )

data = np.load("sphere_render/meas_simulation.npz")
Is = data['I']
Iups = data['Iups']
Ips = data['Ips']
psis = data['psis']
phis0 = data['phis']
thetas0 = data['thetas']
dm = data['dm']

ymax, xmax, imax = Is.shape

phis = np.zeros((ymax,xmax))
thetas = np.zeros((ymax,xmax))
ns = np.zeros((ymax,xmax))

ph = 1
th = 1
n = 1.6
kol = 0
fal = 0
for x in range(310, 311):
    if x%50 == 0:
        print ("x =", x)
    for y in range (310, 311):
        if np.isnan(dm[y,x]):
            continue
        failed = False
        Iup = Iups[y,x]
        Ip = Ips[y,x]
        ps = psis[y,x]
        I = Is[y,x,:].flatten()
        ph = phis0[y,x]
        th = thetas0[y,x]
        n = 1.6
        # print("y = ", y)
        res = np.zeros((1000, 1000, 20))
        psj = -pi
        for ii in range(0,1000):
            psj += 2*pi/1000
            thj = 0
            if ii%100 == 0:
                print ("ii =", ii)
            for jj in range(0,1000):
                thj += (pi/2)/1000
                nj = 1.4
                for kk in range (0,20):
                    nj += 0.1
                    res1 = system((psj, thj, nj), *(Iup, Ip, ps, I))
                    res1 = [abs(xx) for xx in res1]
                    res[ii,jj, kk] = sum(res1)

        # plt.imshow(res)
        # plt.colorbar()
        # # plt.clim(-pi, pi)
        # plt.show()
        print (res.min())
        np.save ("tempres.npy", res)


        # Trying GT value as input
        # try: 
        #     sol = root(system, (ph, th, n), args=(Iup, Ip, ps, I), method='lm')
        #     ph = sol.x[0]
        #     th = sol.x[1]
        #     n = sol.x[2]
        # except:
        #     failed = True

        # Stupid idea on trying different inits                    
        # try:
        #     sol = root(system, (ph, th, n), args=(Iup, Ip, ps, I), method='lm')
        #     ph = sol.x[0]
        #     th = sol.x[1]
        #     n = sol.x[2]
        # except:
        #     failed = True
        
        # if failed:
        #     failed = False
        #     try:
        #         sol = root(system, (3, 1, 1.6), args=(Iup, Ip, ps, I), method='lm')
        #         ph = sol.x[0]
        #         th = sol.x[1]
        #         n = sol.x[2]
        #     except:
        #         failed = True

        # if failed:
        #     failed = False
        #     try:
        #         sol = root(system, (2, 1.4, 1.6), args=(Iup, Ip, ps, I), method='lm')
        #         ph = sol.x[0]
        #         th = sol.x[1]
        #         n = sol.x[2]
        #     except:
        #         failed = True

        # if failed:
        #     failed = False
        #     try:
        #         sol = root(system, (0, 0, 1.6), args=(Iup, Ip, ps, I), method='lm')
        #         ph = sol.x[0]
        #         th = sol.x[1]
        #         n = sol.x[2]
        #     except:
        #         failed = True

        # if failed:
        #     failed = False
        #     phis[y,x] = 0
        #     thetas[y,x] = 0
        #     ns[y,x] = 0
        #     fal += 1
        #     continue

        # res1 = system((sol.x[0], sol.x[1], sol.x[2]), *(Iup, Ip, ps, I))
        # res1 = [abs(xx) for xx in res1]
        # summa = sum(res1)

        # if summa > 0.001:
        #     try:
        #         sol = root(system, (phis0[y,x], thetas0[y,x], 1.6), args=(Iup, Ip, ps, I), method='lm')
        #         ph = sol.x[0]
        #         th = sol.x[1]
        #         n = sol.x[2]
        #     except:
        #         pass
        #     kol += 1

        # phis[y,x] = pied(ph)
        # thetas[y,x] = pied(th)
        # ns[y,x] = n

np.savez ("sphere_render/nmap_test_results.npz", phis=phis, thetas=thetas, ns=ns)

print ("Fails =", fal)
print ("Bad estimates =", kol)


        # sol = root(system, (3, 1.4, 1.6), args=(Iup, Ip, ps, I), method='lm')
        # print (sol.x[0], sol.x[1], sol.x[2])
        # res1 = system((sol.x[0], sol.x[1], sol.x[2]), *(Iup, Ip, ps, I))
        # print (res1)
        # gt = (phis0[y,x], thetas0[y,x], 1.6)
        # print (gt)
        # res2 = system(gt, *(Iup, Ip, ps, I))
        # print (res2)