import numpy as np
import scipy.interpolate as interp
import matplotlib.pyplot as plt

def getCSM(X, Y):
    XSqr = np.sum(X**2, 1)
    YSqr = np.sum(Y**2, 1)
    C = XSqr[:, None] + YSqr[None, :] - 2*X.dot(Y.T)
    C[C < 0] = 0
    return np.sqrt(C)

def getSSM(X):
    return getCSM(X, X)

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

def get1DZNormSSM(SSM):
    std = np.std(SSM, 1)
    return SSM/std[:, None]

def get2DZNormSSM(SSM):
    std = np.std(SSM.flatten())
    return SSM/std

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

def rasterizeWarpingPath(P):
    if np.sum(np.abs(P - np.round(P))) == 0:
        #No effect if already integer
        return P
    P2 = np.round(P)
    P2 = np.array(P2, dtype = np.int32)
    ret = []
    for i in range(P2.shape[0]-1):
        [i1, j1] = [P2[i, 0], P2[i, 1]]
        [i2, j2] = [P2[i+1, 0], P2[i+1, 1]]
        ret.append([i1, j1])
        for k in range(1, i2-i1):
            ret.append([i1+k, j1])
        ret.append([i2, j2])
    return np.array(ret)

def computeAlignmentError(pP1, pP2, doPlot = False):
    """
    Compute area-based alignment error.  Assume that P1
    and P2 are on the same grid
    """
    P1 = rasterizeWarpingPath(pP1)
    P2 = rasterizeWarpingPath(pP2)
    M = np.max(P1[:, 0])
    N = np.max(P1[:, 1])
    A1 = np.zeros((M, N))
    A2 = np.zeros((M, N))
    for i in range(P1.shape[0]):
        [ii, jj] = [P1[i, 0], P1[i, 1]]
        [ii, jj] = [min(ii, M-1), min(jj, N-1)]
        A1[ii, jj::] = 1.0
    for i in range(P2.shape[0]):
        [ii, jj] = [P2[i, 0], P2[i, 1]]
        [ii, jj] = [min(ii, M-1), min(jj, N-1)]
        A2[ii, jj::] = 1.0
    A = np.abs(A1 - A2)
    score = np.sum(A)/(float(M)*float(N))
    if doPlot:
        plt.imshow(A)
        plt.hold(True)
        plt.scatter(pP1[:, 1], pP1[:, 0], 5, 'c', edgecolor = 'none')
        plt.scatter(pP2[:, 1], pP2[:, 0], 5, 'r', edgecolor = 'none')
        plt.title("Score = %g"%score)
    return score
