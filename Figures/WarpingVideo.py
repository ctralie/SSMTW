import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.interpolate as interp
import sys
sys.path.append('..')
from SyntheticCurves import *
from AlignmentTools import *

if __name__ == '__main__':
    N = 600
    NParams = 4
    NPerParam = 30
    Dict = getWarpDictionary(N)
    ts = np.linspace(0, 1, NPerParam)  
    
    c = plt.get_cmap('Spectral')
    C = c(np.array(np.round(np.linspace(0, 255, N)), dtype=np.int32))
    C = C[:, 0:3]
    
    count = 0
    t1 = np.linspace(0, 1, N)
    t2 = np.array(t1)
    fig = plt.figure(figsize=(18, 5))
    for i in range(NParams):
        t1 = np.array(t2)
        t2 = getWarpingPath(Dict, 3, doPlot = False)
        for k in range(NPerParam):
            plt.clf()
            t = ts[k]*t2 + (1.0-ts[k])*t1
            t = t - np.min(t)
            t = t/np.max(t)
            plt.subplot(131)
            plt.plot(t, 'b')
            plt.xlabel("Time Index")
            plt.ylabel("Parameter")
            X = getTorusKnot(3, 5, t)
            ax = fig.add_subplot(132, projection='3d')
            ax.scatter(X[:, 0], X[:, 1], X[:, 2], c = C)
            
            
            plt.subplot(133)
            D = getCSM(X, X)
            plt.imshow(D, interpolation = 'none', cmap = 'afmhot')
            
            if k == 0:
                plt.subplot(131)
                plt.plot(t, 'r', LineWidth=4)
                for kk in range(10):
                    plt.savefig("%i.png"%count, bbox_inches = 'tight')
                    count += 1
            else:        
                plt.savefig("%i.png"%count, bbox_inches = 'tight')
                count += 1
