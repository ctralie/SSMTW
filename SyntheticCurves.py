"""
Programmer: Chris Tralie
Purpose: To create a collection of functions for making families of curves and applying
random rotations/translations/deformations/reparameterizations to existing curves
to test out the isometry blind time warping algorithms
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.interpolate as interp
import sys
from AlignmentTools import *

###################################################
#           TOPC Utility Functions                #
###################################################

def applyRandomRigidTransformation(X):
    """
    Randomly rigidly rotate and translate a time-ordered point cloud
    :param X: Nxd matrix representing a time-ordered point cloud
    :return Y: Nxd matrix representing transformed version of X
    """
    dim = X.shape[1]
    CM = np.mean(X, 0)
    X = X - CM
    #Make a random rotation matrix
    R = np.random.randn(dim, dim)
    R, S, V = np.linalg.svd(R)
    T = np.std(X)*np.random.randn(dim)
    return CM[None, :] + np.dot(X, R) + T[None, :]

def getMeanDistNeighbs(X, Kappa):
    N = X.shape[0]
    K = int(Kappa*N)
    (D, _) = getSSM(X, N)
    Neighbs = np.partition(D, K+1, 1)[:, 0:K+1]
    return np.mean(Neighbs, 1)*float(K+1)/float(K)

def addGaussianNoise(X, Kappa, NRelMag):
    N = X.shape[0]
    MeanDist = getMeanDistNeighbs(X, Kappa)
    return X + NRelMag*MeanDist[:, None]*np.random.randn(N, X.shape[1])

def addRandomBumps(X, Kappa, NRelMag, NBumps):
    N = X.shape[0]
    Y = np.array(X)
    MeanDist = getMeanDistNeighbs(X, Kappa)
    Bumps = np.zeros((NBumps, X.shape[1]))
    for i in range(NBumps):
        idx = np.random.randint(N)
        u = np.random.randn(1, X.shape[1])
        u = u/np.sqrt(np.sum(u**2))
        x = Y[idx, :] + MeanDist[idx]*NRelMag*u
        Bumps[i, :] = x
        diff = -Y+x
        distSqr = np.sum(diff**2, 1)
        idx = np.argmin(distSqr)
        t = (idx - np.arange(N))/(N*Kappa)
        sigma = np.sqrt(distSqr[idx])/np.sqrt(-np.log(0.9))
        Y += diff*np.exp(-t**2)[:, None]*np.exp(-distSqr/(sigma**2))[:, None]
    return (Y, Bumps)

def smoothCurve(X, Fac):
    """
    Use splines to smooth the curve
    :param X: Nxd matrix representing a time-ordered point cloud
    :param Fac: Smoothing factor
    :return Y: An (NxFac)xd matrix of a smoothed, upsampled point cloud
    """
    NPoints = X.shape[0]
    dim = X.shape[1]
    idx = range(NPoints)
    idxx = np.linspace(0, NPoints, NPoints*Fac)
    Y = np.zeros((NPoints*Fac, dim))
    NPointsOut = 0
    for ii in range(dim):
        Y[:, ii] = interp.spline(idx, X[:, ii], idxx)
        #Smooth with box filter
        y = (0.5/Fac)*np.convolve(Y[:, ii], np.ones(Fac*2), mode='same')
        Y[0:len(y), ii] = y
        NPointsOut = len(y)
    Y = Y[0:NPointsOut-1, :]
    Y = Y[2*Fac:-2*Fac, :]
    return Y

def makeRandomWalkCurve(res, NPoints, dim):
    """
    Make a random walk curve with "NPoints" in dimension "dim"
    :param res: An integer specifying the resolution of the random walk grid
    :param NPoints: Number of points in the curve
    :param dim: Dimension of the ambient Euclidean space of the curve
    :return X
    """
    #Enumerate all neighbors in hypercube via base 3 counting between [-1, 0, 1]
    Neighbs = np.zeros((3**dim, dim))
    Neighbs[0, :] = -np.ones((1, dim))
    idx = 1
    for ii in range(1, 3**dim):
        N = np.copy(Neighbs[idx-1, :])
        N[0] += 1
        for kk in range(dim):
            if N[kk] > 1:
                N[kk] = -1
                N[kk+1] += 1
        Neighbs[idx, :] = N
        idx += 1
    #Exclude the neighbor that's in the same place
    Neighbs = Neighbs[np.sum(np.abs(Neighbs), 1) > 0, :]

    #Pick a random starting point
    X = np.zeros((NPoints, dim))
    X[0, :] = np.random.choice(res, dim)

    #Trace out a random path
    for ii in range(1, NPoints):
        prev = np.copy(X[ii-1, :])
        N = np.tile(prev, (Neighbs.shape[0], 1)) + Neighbs
        #Pick a random next point that is in bounds
        idx = np.sum(N > 0, 1) + np.sum(N < res, 1)
        N = N[idx == 2*dim, :]
        X[ii, :] = N[np.random.choice(N.shape[0], 1), :]
    return X

###################################################
#               Curve Families                    #
###################################################

#Note: All function assume the parameterization is given
#in the interval [0, 1]


#######2D Curves
def getLissajousCurve(A, B, a, b, delta, pt):
    """
    Return the curve with
    x = Asin(at + delta)
    y = Bsin(bt)
    """
    N = len(pt)
    t = 2*np.pi*pt
    X = np.zeros((N, 2))
    X[:, 0] = A*np.sin(a*t + delta)
    X[:, 1] = B*np.sin(b*t)
    return X

def get2DFigure8(pt):
    """Return a figure 8 curve parameterized on [0, 1]"""
    return getLissajousCurve(1, 1, 1, 2, 0, pt)


def getPinchedCircle(pt):
    """Return a pinched circle paramterized on [0, 1]"""
    N = len(pt)
    t = 2*np.pi*pt
    X = np.zeros((N, 2))
    X[:, 0] = (1.5 + np.cos(2*t))*np.cos(t)
    X[:, 1] = (1.5 + np.cos(2*t))*np.sin(t)
    return X

def getEpicycloid(R, r, pt):
    N = len(pt)
    t = 2*np.pi*pt
    X = np.zeros((N, 2))
    X[:, 0] = (R+r)*np.cos(t) - r*np.cos(t*(R+r)/r)
    X[:, 1] = (R+r)*np.sin(t) - r*np.sin(t*(R+r)/r)
    return X

def getTschirnhausenCubic(a, pt):
    """
    Return the plane curve defined by the polar equation
    r = asec^3(theta/3)
    """
    N = len(pt)
    t = 5*(pt-0.5)
    X = np.zeros((N, 2))
    X[:, 0] = a*(1-3*t**2)
    X[:, 1] = a*t*(3-t**2)
    X = 2*X/np.max(np.abs(X))
    return X

#######3D Curves
def getVivianiFigure8(a, pt):
    """
    Return the curve that results from the intersection of
    a sphere of radius 2a centered at the origin and a cylinder
    centered at (a, 0, 0) of radius a (the figure 8 I have is
    a 2D projection of this)
    """
    N = len(pt)
    t = 4*np.pi*pt - np.pi
    X = np.zeros((N, 3))
    X[:, 0] = a*(1+np.cos(t))
    X[:, 1] = a*np.sin(t)
    X[:, 2] = 2*a*np.sin(t/2)
    return X


def getTorusKnot(p, q, pt):
    """Return a p-q torus not parameterized on [0, 1]"""
    N = len(pt)
    t = 2*np.pi*pt
    X = np.zeros((N, 3))
    r = np.cos(q*t) + 2
    X[:, 0] = r*np.cos(p*t)
    X[:, 1] = r*np.sin(p*t)
    X[:, 2] = -np.sin(q*t)
    return X

def getConeHelix(c, NPeriods, pt):
    """Return a helix wrapped around a double ended cone"""
    N = len(pt)
    t = NPeriods*2*np.pi*pt
    zt = c*(pt-0.5)
    r = zt
    X = np.zeros((N, 3))
    X[:, 0] = r*np.cos(t)
    X[:, 1] = r*np.sin(t)
    X[:, 2] = zt
    return X


if __name__ == '__main__2':
    np.random.seed(20)
    D = getWarpDictionary(200)
    plt.figure(figsize=(16, 10))
#    plt.subplot(2, 4, 1)
#    t = np.linspace(0, 1, 200)
#    plt.plot(t)
#    plt.title("Original Parameterization")
#    plt.subplot(2, 4, 5)
#    X = getTorusKnot(3, 5, t)
#    SSM = getCSM(X, X)
#    plt.imshow(SSM, cmap = 'afmhot', interpolation = 'nearest')
#    plt.title("Original SSM")
    for i in range(3):
        plt.subplot(2, 3, i+1)
        t = getWarpingPath(D, 3, True)
        #plt.plot(t)
        plt.title("Warping Path %i"%(i+1))
        X = getTorusKnot(3, 5, t)
        SSM = getCSM(X, X)
        plt.subplot(2, 3, 4+i)
        plt.imshow(SSM, cmap = 'afmhot', interpolation = 'nearest')
        plt.title("SSM %i"%(i+1))
    plt.savefig("RandomWarpingPaths.svg", bbox_inches = 'tight')

if __name__ == "__main__2":
    plt.figure(figsize=(7, 3))
    for i in [7, 8]:
        plt.clf()
        np.random.seed(i)
        N = 400
        t = np.linspace(0, 1, N+1)[0:N]
        Kappa = 0.1
        NRelMag = 2
        NBumps = 4
        X = np.zeros((N, 2))
        X[:, 0] = np.cos(2*np.pi*t)
        X[:, 1] = np.sin(4*np.pi*t)
        (Y, Bumps) = addRandomBumps(X, Kappa, NRelMag, NBumps)
        
        diff = np.sqrt(np.sum((X-Y)**2, 1))
        maxdistance = np.max(diff)
        #Do critical point time warping
        D = getCSM(Y, Y)
        diam = np.max(D)
        
        plt.subplot(121)
        plt.scatter(Y[:, 0], Y[:, 1], 20, np.arange(Y.shape[0]), cmap='Spectral', edgecolor='none')
        plt.hold(True)
        plt.scatter(Bumps[:, 0], Bumps[:, 1], 100, 'r')
        plt.axis('equal')
        ax = plt.gca()
        ax.set_axis_bgcolor((0.15, 0.15, 0.15))
        ax.set_xticks([])
        ax.set_yticks([])
        (D, _) = getSSM(Y, Y.shape[0])
        plt.title("r = %.3g"%(maxdistance/diam))
        plt.subplot(122)
        plt.imshow(D, cmap = 'afmhot', interpolation = 'nearest')
        plt.axis('off')
        plt.title("SSM")
        plt.savefig("Figure8Distorted%i.svg"%i, bbox_inches = 'tight')


if __name__ == "__main__":
    N = 400
    t = np.linspace(0, 1, N)
    Xs = {}
    Xs['VivianiFigure8'] = getVivianiFigure8(0.5, t)
    Xs['TSCubic'] = getTschirnhausenCubic(1, t)
    Xs['TorusKnot23'] = getTorusKnot(2, 3, t)
    Xs['TorusKnot35'] = getTorusKnot(3, 5, t)
    Xs['PinchedCircle'] = getPinchedCircle(t)
    Xs['Lissajous32'] = getLissajousCurve(1, 1, 3, 2, 0, t)
    Xs['Lissajous54'] = getLissajousCurve(1, 1, 5, 4, 0, t)
    Xs['ConeHelix'] = getConeHelix(1, 16, t)
    Xs['Epicycloid1_3'] = getEpicycloid(1.5, 0.5, t)
    Xs['Epicycloid1_4'] = getEpicycloid(2, 0.5, t)
    Xs['Epicycloid1_20'] = getEpicycloid(2, 0.1, t)

    fig = plt.figure(figsize=(15, 4))
    i = 1
    for Curve in Xs:
        plt.clf()
        X = Xs[Curve]
        (SSM, _) = getSSM(X, N)
        if X.shape[1] > 2:
            ax = fig.add_subplot(131, projection='3d')
            ax.plot(X[:, 0], X[:, 1], X[:, 2])
        else:
            plt.subplot(131)
            plt.plot(X[:, 0], X[:, 1])
        plt.title(Curve)
        plt.subplot(132)
        plt.imshow(SSM, interpolation = 'none', cmap='afmhot')
        #SSMD = np.sign(SSM[:, 1::] - SSM[:, 0:-1])
        idx = np.argsort(SSM, 1)
        SSMD = np.zeros(idx.shape)
        for k in range(SSMD.shape[0]):
            SSMD[k, idx[k, :]] = np.linspace(0, 1, SSMD.shape[1])
        plt.subplot(133)
        plt.imshow(SSMD, interpolation = 'none', cmap = 'gray')
        plt.savefig("%s.png"%Curve, bbox_inches = 'tight')
        i += 1


if __name__ == "__main__2":
    np.random.seed(10)
    X = makeRandomWalkCurve(4, 20, 2)
    Y = smoothCurve(X, 20)
    plt.subplot(1, 2, 1)
    plt.scatter(Y[:, 0], Y[:, 1], 10, 'b')
    plt.plot(Y[:, 0], Y[:, 1], 'r')
    plt.show()
