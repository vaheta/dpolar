import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as spio
from scipy.optimize import fsolve, root
import copy

pi = math.pi

def pied (x):
    while x>pi:
        x -= pi
    return (x)

def nearest_nonzero_idx(a,x,y):
    idx = np.argwhere(a)
    return idx[((idx - [x,y])**2).sum(1).argmin()]

def fill_with_neasest(ymap):
    a = copy.deepcopy(ymap)
    for y in range (0, 100):
        for x in range(0,100):
            if a[y,x] == 0:
                y1, x1 = nearest_nonzero_idx(a, y, x)
                a[y,x] = a[y1,x1]
    return a

data = np.load("sphere_render/meas_simulation.npz")
Is = data['I']
Iups = data['Iups']
Ips = data['Ips']
psis = data['psis']
xmap = data['xmap']
ymap = data['ymap']

xmap = fill_with_neasest(xmap)
ymap = fill_with_neasest(ymap)

def system (p):
    angles = [0, pi/6, pi/3, pi/2, 2*pi/3, 5*pi/6]
    ph, th, n = p
    thmap = int(math.floor(th*100/(pi/2+0.1)))
    phmap = int(math.floor((ph+pi)*100/(2*pi+0.1)))
    x = int(xmap[thmap,phmap])
    y = int(ymap[thmap,phmap])
    Iup = Iups[y,x]
    Ip = Ips[y,x]
    ps = psis[y,x]
    I = Is[y,x,:].flatten()
    return (
        (Iup/2) * (((((math.cos(th)) - math.sqrt(n**2 - (math.sin(th))**2))/((math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2)))**2)*(math.sin(ph - angles[0]))**2 + (((-n**2*(math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2))/(n**2*(math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2)))**2)*(1 - (math.sin(ph - angles[0]))**2)) + Ip*((((-n**2*(math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2))/(n**2*(math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2)))**2)*((math.cos(ps))**2) + ((((math.cos(th)) - math.sqrt(n**2 - (math.sin(th))**2))/((math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2)))**2)*((math.sin(ps))**2))*(math.cos(math.atan((math.tan(ps))*math.sqrt((((-n**2*(math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2))/(n**2*(math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2)))**2)/((((math.cos(th)) - math.sqrt(n**2 - (math.sin(th))**2))/((math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2)))**2))) + ph - angles[0]))**2 - I[0],
        (Iup/2) * (((((math.cos(th)) - math.sqrt(n**2 - (math.sin(th))**2))/((math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2)))**2)*(math.sin(ph - angles[1]))**2 + (((-n**2*(math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2))/(n**2*(math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2)))**2)*(1 - (math.sin(ph - angles[1]))**2)) + Ip*((((-n**2*(math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2))/(n**2*(math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2)))**2)*((math.cos(ps))**2) + ((((math.cos(th)) - math.sqrt(n**2 - (math.sin(th))**2))/((math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2)))**2)*((math.sin(ps))**2))*(math.cos(math.atan((math.tan(ps))*math.sqrt((((-n**2*(math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2))/(n**2*(math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2)))**2)/((((math.cos(th)) - math.sqrt(n**2 - (math.sin(th))**2))/((math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2)))**2))) + ph - angles[1]))**2 - I[1],
        (Iup/2) * (((((math.cos(th)) - math.sqrt(n**2 - (math.sin(th))**2))/((math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2)))**2)*(math.sin(ph - angles[2]))**2 + (((-n**2*(math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2))/(n**2*(math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2)))**2)*(1 - (math.sin(ph - angles[2]))**2)) + Ip*((((-n**2*(math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2))/(n**2*(math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2)))**2)*((math.cos(ps))**2) + ((((math.cos(th)) - math.sqrt(n**2 - (math.sin(th))**2))/((math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2)))**2)*((math.sin(ps))**2))*(math.cos(math.atan((math.tan(ps))*math.sqrt((((-n**2*(math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2))/(n**2*(math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2)))**2)/((((math.cos(th)) - math.sqrt(n**2 - (math.sin(th))**2))/((math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2)))**2))) + ph - angles[2]))**2 - I[2],
        (Iup/2) * (((((math.cos(th)) - math.sqrt(n**2 - (math.sin(th))**2))/((math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2)))**2)*(math.sin(ph - angles[3]))**2 + (((-n**2*(math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2))/(n**2*(math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2)))**2)*(1 - (math.sin(ph - angles[3]))**2)) + Ip*((((-n**2*(math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2))/(n**2*(math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2)))**2)*((math.cos(ps))**2) + ((((math.cos(th)) - math.sqrt(n**2 - (math.sin(th))**2))/((math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2)))**2)*((math.sin(ps))**2))*(math.cos(math.atan((math.tan(ps))*math.sqrt((((-n**2*(math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2))/(n**2*(math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2)))**2)/((((math.cos(th)) - math.sqrt(n**2 - (math.sin(th))**2))/((math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2)))**2))) + ph - angles[3]))**2 - I[3],
        (Iup/2) * (((((math.cos(th)) - math.sqrt(n**2 - (math.sin(th))**2))/((math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2)))**2)*(math.sin(ph - angles[4]))**2 + (((-n**2*(math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2))/(n**2*(math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2)))**2)*(1 - (math.sin(ph - angles[4]))**2)) + Ip*((((-n**2*(math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2))/(n**2*(math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2)))**2)*((math.cos(ps))**2) + ((((math.cos(th)) - math.sqrt(n**2 - (math.sin(th))**2))/((math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2)))**2)*((math.sin(ps))**2))*(math.cos(math.atan((math.tan(ps))*math.sqrt((((-n**2*(math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2))/(n**2*(math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2)))**2)/((((math.cos(th)) - math.sqrt(n**2 - (math.sin(th))**2))/((math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2)))**2))) + ph - angles[4]))**2 - I[4],
        (Iup/2) * (((((math.cos(th)) - math.sqrt(n**2 - (math.sin(th))**2))/((math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2)))**2)*(math.sin(ph - angles[5]))**2 + (((-n**2*(math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2))/(n**2*(math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2)))**2)*(1 - (math.sin(ph - angles[5]))**2)) + Ip*((((-n**2*(math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2))/(n**2*(math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2)))**2)*((math.cos(ps))**2) + ((((math.cos(th)) - math.sqrt(n**2 - (math.sin(th))**2))/((math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2)))**2)*((math.sin(ps))**2))*(math.cos(math.atan((math.tan(ps))*math.sqrt((((-n**2*(math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2))/(n**2*(math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2)))**2)/((((math.cos(th)) - math.sqrt(n**2 - (math.sin(th))**2))/((math.cos(th)) + math.sqrt(n**2 - (math.sin(th))**2)))**2))) + ph - angles[5]))**2 - I[5]
    )


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
        
        ph = phis0[y,x]
        th = thetas0[y,x]
        n = 1.6

        # res1 = system((27*2*pi/100-pi, (99/100)*(pi/2), n))
        # print ("x=", x, " y=", y, res1)

        res = np.zeros((100, 100))
        psj = -pi
        for ii in range(0,100):
            psj += 2*pi/100
            thj = 0
            if ii%10 == 0:
                print ("ii =", ii)
            for jj in range(0,100):
                thj += (pi/2)/100
                res1 = system((psj, thj, 1.6))
                res1 = [abs(xx) for xx in res1]
                res[ii,jj] = sum(res1)


        print (res.min())
        np.save ("tempres.npy", res)

        # try:
        #     sol = root(system, (ph, th, n), method='lm')
        #     ph = sol.x[0]
        #     th = sol.x[1]
        #     n = sol.x[2]
        # except:
        #     failed = True

        phis[y,x] = pied(ph)
        thetas[y,x] = pied(th)
        ns[y,x] = n

np.savez ("sphere_render/nmap_test_results.npz", phis=phis, thetas=thetas, ns=ns)

# print ("Fails =", fal)
# print ("Bad estimates =", kol)


        # sol = root(system, (3, 1.4, 1.6), args=(Iup, Ip, ps, I), method='lm')
        # print (sol.x[0], sol.x[1], sol.x[2])
        # res1 = system((sol.x[0], sol.x[1], sol.x[2]), *(Iup, Ip, ps, I))
        # print (res1)
        # gt = (phis0[y,x], thetas0[y,x], 1.6)
        # print (gt)
        # res2 = system(gt, *(Iup, Ip, ps, I))
        # print (res2)