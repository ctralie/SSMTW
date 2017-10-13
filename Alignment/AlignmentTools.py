import numpy as np
import scipy.interpolate as interp
import matplotlib.pyplot as plt
import scipy.sparse as sparse

def getCSM(X, Y):
    XSqr = np.sum(X**2, 1)
    YSqr = np.sum(Y**2, 1)
    C = XSqr[:, None] + YSqr[None, :] - 2*X.dot(Y.T)
    C[C < 0] = 0
    return np.sqrt(C)

def CSMToBinary(D, Kappa):
    """
    Turn a cross-similarity matrix into a binary cross-simlarity matrix
    If Kappa = 0, take all neighbors
    If Kappa < 1 it is the fraction of mutual neighbors to consider
    Otherwise Kappa is the number of mutual neighbors to consider
    """
    N = D.shape[0]
    M = D.shape[1]
    if Kappa == 0:
        return np.ones((N, M))
    elif Kappa < 1:
        NNeighbs = int(np.round(Kappa*M))
    else:
        NNeighbs = Kappa
    J = np.argpartition(D, NNeighbs, 1)[:, 0:NNeighbs]
    I = np.tile(np.arange(N)[:, None], (1, NNeighbs))
    V = np.ones(I.size)
    [I, J] = [I.flatten(), J.flatten()]
    ret = sparse.coo_matrix((V, (I, J)), shape=(N, M))
    return ret.toarray()

def CSMToBinaryMutual(D, Kappa):
    """
    Take the binary AND between the nearest neighbors in one direction
    and the other
    """
    B1 = CSMToBinary(D, Kappa)
    B2 = CSMToBinary(D.T, Kappa).T
    return B1*B2

def getSSM(X):
    """
    Return the SSM between all rows of a time-ordered Euclidean point cloud X
    """
    return getCSM(X, X)

def getRankSSM(SSM):
    idx = np.argsort(SSM, 1)
    SSMD = np.zeros(idx.shape)
    for k in range(SSMD.shape[0]):
        SSMD[k, idx[k, :]] = np.linspace(0, 1, SSMD.shape[1])
    return SSMD

def get2DRankSSM(SSM):
    """
    Return the SSM so that distances are normalized to [0, 1] by their rank;
    i.e. a pixel of rank k in an NxN distance matrix has value k/N*N
    """
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
    """
    Return the SSM normalized row by row by the standard deviation of
    that row
    """
    std = np.std(SSM, 1)
    return SSM/std[:, None]

def get2DZNormSSM(SSM):
    """
    Return the SSM normalized by the standard deviation over all pixels
    """
    std = np.std(SSM.flatten())
    return SSM/std

def matchSSMDist(A, B, L = 100):
    """
    Normalize the distance matrices A and B to the range [0, 1],
    discretize the levels to L bins in that range, and then
    remap the levels in A to the levels in B so that the distribution
    of A matches the distribution of B (ref. Gonzalez&Woods 3.3)
    with help from https://stackoverflow.com/questions/32655686/
    :param A: An MxM distance matrix
    :param B: An NxN distance matrix
    :param L: Number of discretization levels
    """
    #Discretize to the range [0, L-1]
    AFlat = np.round((L-1)*A.flatten()/np.max(A))
    BFlat = np.round((L-1)*B.flatten()/np.max(B))
    #Compute normalized CDFs
    valsA, idxA, countsA = np.unique(AFlat, return_inverse = True, return_counts = True)
    PA = np.cumsum(countsA).astype(np.float64)
    PA /= PA[-1]
    valsB, countsB = np.unique(BFlat, return_counts = True)
    PB = np.cumsum(countsB).astype(np.float64)
    PB /= PB[-1]
    valsInterp = np.interp(PA, PB, valsB)
    return (valsInterp[idxA].reshape(A.shape)/L, BFlat.reshape(B.shape)/L)


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

def projectPath(path, M, N, direction = 0):
    """
    Project the path onto one of the axes of the CSWM so that the
    correspondence is a bijection
    :param path: An NEntries x 2 array of coordinates in a warping path
    :param M: Number of rows in the CSM
    :param N: Number of columns in the CSM
    :param direction.
        0 - Choose an index along the rows to go with every
            column index
        1 - Choose an index along the columns to go with every
            row index
    :returns retpath: An NProjectedx2 array representing the projected path
    """
    involved = np.zeros((M, N))
    involved[path[:, 0], path[:, 1]] = 1
    retpath = np.array([[0, 0]], dtype =  np.int64)
    if direction == 0:
        retpath = np.zeros((N, 2), dtype = np.int64)
        retpath[:, 1] = np.arange(N)
        #Choose an index along the rows to go with every column index
        retpath[:, 0] = np.argsort(-involved, 0)[0, :]
        #Prune to the column indices that are actually used
        #(for partial matches)
        colmin = np.min(path[:, 1])
        colmax = np.max(path[:, 1])
        retpath = retpath[(retpath[:,1]>=colmin)*(retpath[:,1]<=colmax), :]
    elif direction == 1:
        retpath = np.zeros((M, 2), dtype = np.int64)
        retpath[:, 0] = np.arange(M)
        #Choose an index along the columns to go with every row index
        retpath[:, 1] = np.argsort(-involved, 1)[:, 0]
        #Prune to the row indices that are actually used
        rowmin = np.min(path[:, 0])
        rowmax = np.max(path[:, 1])
        retpath = retpath[(retpath[:,0]>=rowmin)*(retpath[:,0]<=rowmax), :]
    return retpath

def getProjectedPathParam(path, direction = 0, strcmap = 'Spectral'):
    """
    Given a projected path, return the arrays [t1, t2]
    in [0, 1] which synchronize the first curve to the
    second curve
    :param path: Nx2 array representing projected warping path
    :param direction.
        0 - There is an index along the rows to go with every
            column index
        1 - There is an index along the columns to go with every
            row index
    """
    #Figure out bounds of path
    M = path[-1, 0] - path[0, 0] + 1
    N = path[-1, 1] - path[0, 1] + 1
    if direction == 0:
        t1 = np.linspace(0, 1, M)
        t2 = (path[:, 0] - path[0, 0])/float(M)
    elif direction == 1:
        t2 = np.linspace(0, 1, N)
        t1 = float(path[:, 1] - path[0, 1])/float(N)
    else:
        print("Unknown direction for parameterizing projected paths")
        return None

    c = plt.get_cmap(strcmap)
    C1 = c(np.array(np.round(255*t1), dtype=np.int32))
    C1 = C1[:, 0:3]
    C2 = c(np.array(np.round(255*t2), dtype=np.int32))
    C2 = C2[:, 0:3]
    return {'t1':t1, 't2':t2, 'C1':C1, 'C2':C2}


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
