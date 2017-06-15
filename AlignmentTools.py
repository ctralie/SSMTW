import numpy as np
import scipy.interpolate as interp
import matplotlib.pyplot as plt

def getCSM(X, Y):
    XSqr = np.sum(X**2, 1)
    YSqr = np.sum(Y**2, 1)
    C = XSqr[:, None] + YSqr[None, :] - 2*X.dot(Y.T)
    C[C < 0] = 0
    return np.sqrt(C)

def getSSM(X, DPixels, doPlot = False):
    """
    Compute a Euclidean self-similarity image between a set of points
    :param X: An Nxd matrix holding the d coordinates of N points
    :param DPixels: The image will be resized to this dimensions
    :param doPlot: If true, show a plot comparing the original/resized images
    :return: A tuple (D, DResized)
    """
    D = np.sum(X**2, 1)[:, None]
    D = D + D.T - 2*X.dot(X.T)
    D[D < 0] = 0
    D = 0.5*(D + D.T)
    D = np.sqrt(D)
    if doPlot:
        plt.subplot(121)
        plt.imshow(D, interpolation = 'none')
        plt.subplot(122)
        plt.imshow(scipy.misc.imresize(D, (DPixels, DPixels)), interpolation = 'none')
        plt.show()
    if not (D.shape[0] == DPixels):
        return (D, scipy.misc.imresize(D, (DPixels, DPixels)))
    return (D, D)

def getRankSSM(SSM):
    idx = np.argsort(SSM, 1)
    SSMD = np.zeros(idx.shape)
    for k in range(SSMD.shape[0]):
        SSMD[k, idx[k, :]] = np.linspace(0, 1, SSMD.shape[1])
    return SSMD

def get2DRankSSM(SSM):
    N = SSM.shape[0]
    [I, J] = np.meshgrid(np.arange(N), np.arange(N))
    d = SSM[I < J]
    idx = np.argsort(d)
    d[idx] = np.linspace(0, 1, len(d))
    D = 0*SSM
    D[I < J] = d
    D = D + D.T
    return D

def getZNormSSM(SSM):
    std = np.std(SSM, 1)
    return SSM/std[:, None]

###################################################
#                Warping Paths                    #
###################################################
def getInverseFnEquallySampled(t, x):
    N = len(t)
    t2 = np.linspace(np.min(x), np.max(x), N)
    return interp.spline(x, t, t2)

def getWarpDictionary(N, plotPaths = False):
    t = np.linspace(0, 1, N)
    D = []
    #Polynomial
    if plotPaths:
        plt.subplot(131)
        plt.title('Polynomial')
        plt.hold(True)
    for p in range(-4, 6):
        tp = p
        if tp < 0:
            tp = -1.0/tp
        x = t**(tp**1)
        D.append(x)
        if plotPaths:
            plt.plot(x)
    #Exponential / Logarithmic
    if plotPaths:
        plt.subplot(132)
        plt.title('Exponential / Logarithmic')
        plt.hold(True)
    for p in range(2, 6):
        t = np.linspace(1, p**p, N)
        x = np.log(t)
        x = x - np.min(x)
        x = x/np.max(x)
        t = t/np.max(t)
        x2 = getInverseFnEquallySampled(t, x)
        x2 = x2 - np.min(x2)
        x2 = x2/np.max(x2)
        #D.append(x)
        #D.append(x2)
        if plotPaths:
            plt.plot(x)
            plt.plot(x2)
    #Hyperbolic Tangent
    if plotPaths:
        plt.subplot(133)
        plt.title('Hyperbolic Tangent')
        plt.hold(True)
    for p in range(2, 5):
        t = np.linspace(-2, p, N)
        x = np.tanh(t)
        x = x - np.min(x)
        x = x/np.max(x)
        t = t/np.max(t)
        x2 = getInverseFnEquallySampled(t, x)
        x2 = x2 - np.min(x2)
        x2 = x2/np.max(x2)
        D.append(x)
        D.append(x2)
        if plotPaths:
            plt.plot(x)
            plt.plot(x2)
    D = np.array(D)
    return D

def getWarpingPath(D, k, doPlot = False):
    """
    Return a warping path made up of k elements
    drawn from dictionary D
    """
    N = D.shape[0]
    dim = D.shape[1]
    ret = np.zeros(dim)
    idxs = np.random.permutation(N)[0:k]
    weights = np.zeros(N)
    weights[idxs] = np.random.rand(k)
    weights = weights / np.sum(weights)
    res = weights.dot(D)
    res = res - np.min(res)
    res = res/np.max(res)
    if doPlot:
        plt.plot(res)
        plt.hold(True)
        for idx in idxs:
            plt.plot(np.arange(dim), D[idx, :], linestyle='--')
        plt.title('Constructed Warping Path')
    return res

def getInterpolatedEuclideanTimeSeries(X, t):
    M = X.shape[0]
    d = X.shape[1]
    t0 = np.linspace(0, 1, M)
    dix = np.arange(d)
    f = interp.interp2d(dix, t0, X, kind='linear')
    Y = f(dix, t)
    return Y

def projectPath(path):
    """
    Choose an index along the column to go with every
    row index
    """
    M = path[-1, 0] + 1
    N = path[-1, 1] + 1
    involved = np.zeros((M, N))
    involved[path[:, 0], path[:, 1]] = 1
    return np.argsort(-involved, 0)[0, :]

def computeAlignmentError(P1, P2, M, N):
    Area = np.zeros((M, N))
