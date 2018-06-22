import cv2
import numpy as np
import numpy.matlib
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy import signal
import math
import matplotlib.pyplot as plt
pi = 3.14159265

def polarimgs2norms (imgs, xvals, n, mask):
    ymax, xmax, imax = imgs.shape
    I = np.zeros((ymax,xmax))
    ro = np.zeros((ymax,xmax))
    phi = np.zeros((ymax,xmax))
    theta = np.zeros((ymax,xmax))
    B = np.zeros((imax,3))

    for y in range(0,ymax):
        for x in range(0,xmax):
            if mask[y,x]==1:
                yvals = np.zeros((1,imax))
                for i in range(0,imax):
                    yvals[0,i] = imgs[y,x,i]
                    B[i,:] = np.mat([1, math.cos(2*xvals[0,i]), math.sin(2*xvals[0,i])])
                solth = 0
                if (np.linalg.norm(yvals,1)==0):
                    ro[y,x] = 0
                    theta[y,x] = 0
                    phi[y,x] = 0
                    continue
                x1 = np.linalg.solve(B,yvals.T)
                I[y,x] = 2*x1[0]
                ro[y,x] = (2*((x1[1])**2 + (x1[2])**2)**(1/2))/I[y,x]
                r = ro[y,x]
                phi[y,x] = math.atan2(x1[2],x1[1])
                if (r>0) and (r<0.6):
                    alpha = (n-1/n)**2 + r*(n+1/n)**2
                    bb = 4*r*(n**2+1)*(alpha-4*r)
                    discr = bb**2 + 16*(r**2)*(16*(r**2)-alpha**2)*(n**2-1)**2
                    temp = ((-bb-discr**(1/2))/(2*(16*(r**2)-alpha**2)))**(1/2)
                    solth = math.asin(temp)
                theta[y,x] = solth
    return (phi, theta, ro)

def get_kdtree_normals(XYZ, objmask):
    N = np.zeros(XYZ.shape)
    NX = np.zeros(XYZ[:,:,0].shape)
    NY = np.zeros(XYZ[:,:,0].shape)
    NZ = np.zeros(XYZ[:,:,0].shape)
    ymax = XYZ[:,:,0].shape[0]
    xmax = XYZ[:,:,0].shape[1]
    thrsh = 5
    r = 6
    view_point = np.array([0,0,0])
    
    for y in range(0, ymax):
        for x in range(0, xmax):
            if objmask[y,x]==0:
                continue
            nbhood = np.array([XYZ[y,x,0], XYZ[y,x,1], XYZ[y,x,2]])
            patch = XYZ[(y-r):(y+r),(x-r):(x+r),:]
            for i in range(0, (2*r)**2):
                yy = i//(2*r)
                xx = i%(2*r)
                if 0<abs(XYZ[y,x,2]-patch[yy,xx,2])<thrsh:
                    nbhood = np.vstack([nbhood, patch[yy,xx,:]])
            if nbhood.shape[0]>2:
                shnbhood = nbhood - numpy.matlib.repmat(np.mean(nbhood, axis=0),nbhood.shape[0],1)
                u, s, vh = np.linalg.svd(shnbhood, full_matrices=True)
                normal = vh[2,:]
                normal = np.divide(normal, np.linalg.norm(normal))
                if np.dot(normal,(view_point-XYZ[y,x,:])) < 0:
                    normal = -normal
                NX[y,x] = normal[0]
                NY[y,x] = normal[1]
                NZ[y,x] = -normal[2]
    N[:,:,0] = NX
    N[:,:,1] = NY
    N[:,:,2] = NZ
    return N

def fuser(phi, objmask, azimuth):
    r = 2
    ymax = phi.shape[0]
    xmax = phi.shape[1]
    ch_mask=np.zeros(phi.shape)
    phi_corr = phi.copy()
    for y in range(0,ymax):
        for x in range(0,xmax):
            if objmask[y,x]==0:
                continue
            phixy = phi[y,x]
            phi_patch = phi[(y-r):(y+r), (x-r):(x+r)]
            azimuth_patch = azimuth[(y-r):(y+r), (x-r):(x+r)]
            mask_patch = objmask[(y-r):(y+r), (x-r):(x+r)]

            error = np.linalg.norm(np.multiply(mask_patch, (phi_patch-azimuth_patch)))
            error_pi_minus = np.linalg.norm(np.multiply(mask_patch, (phi_patch-pi-azimuth_patch)))
            error_pi_plus = np.linalg.norm(np.multiply(mask_patch, (phi_patch+pi-azimuth_patch)))
            er = np.array([error, error_pi_minus, error_pi_plus])
            if (np.min(er)==abs(error_pi_minus)) and (phixy>=0):
                ch_mask[y,x] = 1
            elif (np.min(er)==abs(error_pi_plus)) and (phixy<=0):
                ch_mask[y,x] = 1  
            elif (abs(phixy-azimuth[y,x])>=2):
                ch_mask[y,x] = 1

            if (ch_mask[y,x] == 1):
                if (phixy>=0):
                    phi_corr[y,x] = phixy - pi
                elif (phixy<0):
                    phi_corr[y,x] = phixy + pi
    return phi_corr

def normals (phi, theta):
    ymax = phi.shape[0]
    xmax = phi.shape[1]
    norms = np.zeros((ymax,xmax,3))
    grad = np.zeros((ymax,xmax,3))
    one = np.ones((ymax,xmax))

    PX = np.multiply(np.cos(phi), np.sin(theta))
    PY = np.multiply(np.sin(phi), np.sin(theta))
    PZ = np.cos(theta)

    norms[:,:,0] = PX
    norms[:,:,1] = PY
    norms[:,:,2] = PZ

    grad[:,:,0] = np.divide(-PX, PZ)
    grad[:,:,1] = np.divide(-PY, PZ)
    grad[:,:,2] = -one

    return (grad, norms)

def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def calculate_f_tensor(gx,gy,d11,d12,d21,d22):
    ymax = gx.shape[0]
    xmax = gx.shape[1] 

    gx1 = gx*d11
    gy1 = gy*d22

    gx1 = np.pad(gx1, 1, mode='constant')
    gy1 = np.pad(gy1, 1, mode='constant')

    gxx = np.zeros(gx1.shape)
    gyy = gxx

    for j in range(0,ymax+1):
        for k in range(0,xmax+1):
            gyy[j+1,k] = gy1[j+1,k] - gy1[j,k]
            gxx[j,k+1] = gx1[j,k+1] - gx1[j,k]
    
    f = gxx+gyy
    f = f[1:-1,1:-1]

    gx1 = gx*d12
    gy1 = gy*d21

    gx2 = gy*d12
    gy2 = gx*d21

    gx2[-1,:] = gx1[-1,:]
    gy2[-1,:] = gy1[-1,:]

    gx2[:,-1] = 0
    gy2[-1,:] = 0

    gx2 = np.pad(gx2, 1, mode='constant')
    gy2 = np.pad(gy2, 1, mode='constant')
    gxx = np.zeros(gx2.shape)
    gyy = gxx

    for j in range(0,ymax+1):
        for k in range(0,xmax+1):
            gyy[j+1,k] = gy2[j+1,k] - gy2[j,k]
            gxx[j,k+1] = gx2[j,k+1] - gx2[j,k]
    
    f2 = gxx+gyy
    f2 = f2[1:-1,1:-1]
    
    f = f + f2
    return f
    
def laplacian_matrix_tensor(H,W,D11,D12,D21,D22):
    D11 = np.pad(D11, 1, mode='constant')
    D12 = np.pad(D12, 1, mode='constant')
    D21 = np.pad(D21, 1, mode='constant')
    D22 = np.pad(D22, 1, mode='constant')

    N = (H+2)*(W+2)
    mask = np.zeros((H+2, W+2))
    mask[1:-1,1:-1] = 1
    idx = np.where(mask==1)
    idd = np.zeros((idx[0].shape[0],1))
    for i in range(0,idx[0].shape[0]):
        idd[i] = idx[1][i]*(H+2)+idx[0][i]


    A = csr_matrix((-D22[idx][:, np.newaxis].T[0],(idd.T[0],idd.T[0]+1)),shape=[N,N])
    A = A + csr_matrix((-D11[idx][:, np.newaxis].T[0],(idd.T[0],idd.T[0]+H+2)),shape=[N,N])
    A = A + csr_matrix((-D22[idx[0],idx[1]-1][:, np.newaxis].T[0],(idd.T[0],idd.T[0]-1)),shape=[N,N])
    A = A + csr_matrix((-D11[idx[0]-1,idx[1]][:, np.newaxis].T[0],(idd.T[0],idd.T[0]-H-2)),shape=[N,N])

    A = A + csr_matrix((-D12[idx][:, np.newaxis].T[0],(idd.T[0],idd.T[0]+1)),shape=[N,N])
    A = A + csr_matrix((-D12[idx[0]-1,idx[1]][:, np.newaxis].T[0],(idd.T[0],idd.T[0]-H-2)),shape=[N,N])
    A = A + csr_matrix((D12[idx[0]-1,idx[1]][:, np.newaxis].T[0],(idd.T[0],idd.T[0]-H-2+1)),shape=[N,N])
    A = A + csr_matrix((-D21[idx][:, np.newaxis].T[0],(idd.T[0],idd.T[0]+H+2)),shape=[N,N])
    A = A + csr_matrix((-D21[idx[0],idx[1]-1][:, np.newaxis].T[0],(idd.T[0],idd.T[0]-1)),shape=[N,N])
    A = A + csr_matrix((D21[idx[0],idx[1]-1][:, np.newaxis].T[0],(idd.T[0],idd.T[0]-1+H+2)),shape=[N,N])

    A = A[idd.T[0]][:,idd.T[0]]
    N = A.shape[0]
    dd = np.sum(A,1).A1
    idx = np.arange(N).T
    A = A + csr_matrix((-dd,(idx,idx)),shape=[N,N])
    return A

def fast_spanning_tree_integrator(nx,ny,dm,clambda,ro,calpha):
    ymax = ro.shape[0]
    xmax = ro.shape[1]
    
    ww = matlab_style_gauss2D((3,3),0.5)
    T11 = signal.convolve2d(np.square(ro),ww,mode = 'same')
    T22 = signal.convolve2d(np.square(ro),ww,mode = 'same')
    T12 = signal.convolve2d(np.square(ro),ww,mode = 'same')

    ImagPart = np.sqrt(np.square(T11-T22)+4*np.square(T12))
    EigD_1 = (T22 + T11 + ImagPart)/2.0
    EigD_2 = (T22 + T11 - ImagPart)/2.0
    THRESHOLD_SMALL = 1*np.amax(EigD_1)/100.0

    L1 = np.ones((ymax,xmax))
    idx = np.where(EigD_1 > THRESHOLD_SMALL)
    L1[idx] = calpha + 1 - np.exp(-3.315/EigD_1[idx]**4)
    L2 = np.ones((ymax,xmax))

    D11 = np.zeros((ymax,xmax))
    D12 = np.zeros((ymax,xmax))
    D22 = np.zeros((ymax,xmax))

    for y in range(0,ymax):
        for x in range(0,xmax):
            Wmat = np.array([[T11[y,x], T12[y,x]], [T12[y,x], T22[y,x]]])
            d, v = np.linalg.eig(Wmat)
            if d[0]>d[1]:
                d1 = d.copy()
                d1[0] = d[1]
                d1[1] = d[0]
                d = d1
                v1 = v.copy()
                v1[:,0] = v[:,1]
                v1[:,1] = v[:,0]
                v = v1
            
            d[0] = L2[y,x]
            d[1] = L1[y,x]
            dd = np.diag(d)
            Wmat = v.dot(dd.dot(v.T))

            D11[y,x] = Wmat[0,0]
            D22[y,x] = Wmat[1,1]
            D12[y,x] = Wmat[0,1]
    
    A = laplacian_matrix_tensor(ymax,xmax,D11,D12,D12,D22)
    f = calculate_f_tensor(nx,ny,D11,D12,D12,D22)

    # Операции со sparse матрицами

    A = A[:,1:]
    f = f.flatten()

    A_bot = csr_matrix((ymax*xmax, ymax*xmax))
        
    A_bot_lil = A_bot.tolil()
    
    for ii in range(0, ymax*xmax):
        A_bot_lil[ii,ii] = clambda
    
    A_bot = A_bot_lil.tocsr()
    A_bot = A_bot[:,1:]
    f_bot = clambda * (-dm).flatten()

    AA = vstack([A, A_bot])
    ff = np.vstack([f[:, np.newaxis], f_bot[:, np.newaxis]])

    surface0 = lsqr(AA,ff.T)
    surface = np.zeros((ymax*xmax,1))
    surface[0,0] = 0
    surface[1:,0] = surface0[0]

    surface = surface.reshape((ymax,xmax))
    surface = surface - np.amin(surface)
    return surface