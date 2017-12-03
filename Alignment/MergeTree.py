import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

def UFFind(UFP, u):
    """
    Union find "find" with path-compression
    :param UFP: A list of pointers to reprsentative nodes
    :param u: Index of the node to find
    :return: Index of the representative of the component of u
    """
    if not (UFP[u] == u):
        UFP[u] = UFFind(UFP, UFP[u])
    return UFP[u]

def UFUnion(UFP, u, v, idxorder):
    """
    Union find "union" with early birth-based merging
    (similar to rank-based merging...not sure if exactly the
    same theoretical running time)
    """
    u = UFFind(UFP, u)
    v = UFFind(UFP, v)
    if u == v:
        return #Already in union
    [ufirst, usecond] = [u, v]
    if idxorder[v] < idxorder[u]:
        [ufirst, usecond] = [v, u]
    UFP[usecond] = ufirst

def mergeTreeFrom1DTimeSeries(x):
    """
    Uses union find to make a merge tree object from the time series x
    (NOTE: This code is pretty general and could work to create merge trees
    on any domain if the neighbor set was updated)
    :param x: 1D array representing the time series
    :return: (Merge Tree dictionary, Persistences dictionary, Persistence diagram)
    """
    #Add points from the bottom up
    N = len(x)
    idx = np.argsort(x)
    idxorder = np.zeros(N)
    idxorder[idx] = np.arange(N)
    UFP = np.arange(N) #Pointer to oldest indices
    UFR = np.arange(N) #Representatives of classes
    I = [] #Persistence diagram
    PS = {} #Persistences for merge tree nodes
    MT = {} #Merge tree
    for i in idx:
        neighbs = set([])
        #Find the oldest representatives of the neighbors that
        #are already alive
        for di in [-1, 1]: #Neighbor set is simply left/right
            if i+di >= 0 and i+di < N:
                if idxorder[i+di] < idxorder[i]:
                    neighbs.add(UFFind(UFP, i+di))
        #If none of this point's neighbors are alive yet, this
        #point will become alive with its own class
        if len(neighbs) == 0:
            continue
        neighbs = [n for n in neighbs]
        #Find the oldest class, merge earlier classes with this class,
        #and record the merge events and birth/death times
        oldestNeighb = neighbs[np.argmin([idxorder[n] for n in neighbs])]
        #No matter, what, the current node becomes part of the
        #oldest class to which it is connected
        UFUnion(UFP, oldestNeighb, i, idxorder)
        if len(neighbs) > 1: #A nontrivial merge
            MT[i] = [UFR[n] for n in neighbs] #Add merge tree children
            for n in neighbs:
                if not (n == oldestNeighb):
                    #Record persistence event
                    I.append([x[n], x[i]])
                    pers = x[i] - x[n]
                    PS[i] = pers
                    PS[n] = pers
                UFUnion(UFP, oldestNeighb, n, idxorder)
            #Change the representative for this class to be the
            #saddle point
            UFR[oldestNeighb] = i
    #Add the essential class
    idx1 = np.argmin(x)
    idx2 = np.argmax(x)
    [b, d] = [x[idx1], x[idx2]]
    I.append([b, d])
    I = np.array(I)
    PS[idx1] = d-b
    PS[idx2] = d-b
    return (MT, PS, I)

def getPersistenceImage(dgm, plims, res, weightfn = lambda b, l: l, psigma = None):
    """
    Return a persistence image (Adams et al.)
    :param dgm: Nx2 array holding persistence diagram
    :param plims: An array [birthleft, birthright, lifebottom, lifetop] \
        limits of the actual grid will be rounded based on res
    :param res: Width of each pixel
    :param weightfn(b, l): A weight function as a function of birth time\
        and life time
    :param psigma: Standard deviation of each Gaussian.  By default\
        None, which indicates it should be res/2.0
    """
    #Convert to birth time/lifetime
    I = np.array(dgm)
    I[:, 1] = I[:, 1] - I[:, 0]
    
    #Create grid
    lims = np.array([np.floor(plims[0]/res), np.ceil(plims[1]/res), np.floor(plims[2]/res), np.ceil(plims[3]/res)])
    xr = np.arange(int(lims[0]), int(lims[1])+2)*res
    yr = np.arange(int(lims[2]), int(lims[3])+2)*res
    sigma = res/2.0
    if psigma:
        sigma = psigma        
            
    #Add each integrated Gaussian
    PI = np.zeros((len(yr)-1, len(xr)-1))
    for i in range(I.shape[0]):
        [x, y] = I[i, :]
        w = weightfn(x, y)
        if w == 0:
            continue
        #CDF of 2D isotropic Gaussian is separable
        xcdf = scipy.stats.norm.cdf((xr - x)/sigma)
        ycdf = scipy.stats.norm.cdf((yr - y)/sigma)
        X = ycdf[:, None]*xcdf[None, :]
        #Integral image
        PI += weightfn(x, y)*(X[1::, 1::] - X[0:-1, 1::] - X[1::, 0:-1] + X[0:-1, 0:-1])
    return {'PI':PI, 'xr':xr[0:-1], 'yr':yr[0:-1]}

def plotDGM(dgm, color = 'b', sz = 20, label = 'dgm', axcolor = np.array([0.0, 0.0, 0.0]), marker = None):
    if dgm.size == 0:
        return
    # Create Lists
    # set axis values
    axMin = np.min(dgm)
    axMax = np.max(dgm)
    axRange = axMax-axMin
    a = max(axMin - axRange/5, 0)
    b = axMax+axRange/5
    # plot line
    plt.plot([a, b], [a, b], c = axcolor, label = 'none')
    plt.hold(True)
    # plot points
    if marker:
        H = plt.scatter(dgm[:, 0], dgm[:, 1], sz, color, marker, label=label, edgecolor = 'none')
    else:
        H = plt.scatter(dgm[:, 0], dgm[:, 1], sz, color, label=label, edgecolor = 'none')
    # add labels
    plt.xlabel('Time of Birth')
    plt.ylabel('Time of Death')
    return H

if __name__ == '__main__':
    t = np.linspace(0, 1, 400)
    t = np.sqrt(t)
    t = t*16*np.pi 
    x = t*np.cos(t)
    x = x - np.min(x)
    x = x/np.max(x)
    (MT, PS, I) = mergeTreeFrom1DTimeSeries(x)
    res = getPersistenceImage(I, [0, 1, 0, 1], 0.01, psigma = 0.02)
    PI = res['PI']
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.plot(x)
    plt.title("Time Series")
    plt.subplot(132)
    plotDGM(I)
    plt.title("Persistence Diagram")
    plt.subplot(133)
    plt.imshow(PI, interpolation = 'none', cmap = 'afmhot',\
        extent = (res['xr'][0], res['xr'][-1], res['yr'][-1], res['yr'][0]))
    plt.xlim([res['xr'][0], res['xr'][-1]])
    plt.ylim([res['yr'][0], res['yr'][-1]])
    plt.title("Persistence Image")
    plt.show()