"""
Code that wraps around Matlab to execute Feng Zhou's CTW/GTW code
http://www.f-zhou.com/ta_code.html
"""
import numpy as np
import scipy.io as sio
import subprocess
import matplotlib.pyplot as plt
import os
import sys
import inspect

def getPrefix():
    s = ''
    if 'Alignment.ctw' in sys.modules:
        #raise Exception('Must import Alignment.ctw before running Matlab code')
        s = "%s/"%os.path.dirname(inspect.getfile(sys.modules['Alignment.ctw']))
    return s

def getCTWAlignments(eng, X1, X2, doPCA = 1):
    """
    Wrap around the CTW library to compute warping paths between
    X1 and X2 using DTW, DDTW, IMW, CTW, and GTW
    :param eng: Matlab engine
    :param X1: N x d Euclidean vector
    :param X2: M x d Euclidean vector
    :param prefix: Relative path to help locate Matlab files
    :returns: dictionary of warping results
    """
    prefix = getPrefix()
    sio.savemat("%sXs.mat"%prefix, {"X1":X1, "X2":X2, "doPCA":doPCA})
    eng.extractAlignments(nargout=0)
    res = sio.loadmat("%smatlabResults.mat"%prefix)
    os.remove("%sXs.mat"%prefix)
    os.remove("%smatlabResults.mat"%prefix)
    toRemove = []
    for k in res.keys():
        if not k[0] == 'P':
            toRemove.append(k)
        else:
            res[k] = res[k] - 1 #Matlab is 1-indexed
    for k in toRemove:
        del res[k]
    return res

def initMatlabEngine():
    prefix = getPrefix()
    import matlab.engine
    beforePath = os.path.realpath(".")
    if len(prefix) > 0:
        os.chdir('%s'%prefix)
    eng = matlab.engine.start_matlab()
    os.chdir(beforePath)
    return eng

def testCTWAlignments(eng):
    N = 200
    t1 = np.linspace(0, 1, N)
    t2 = t1**2
    X1 = np.zeros((N, 2))
    X1[:, 0] = np.cos(2*np.pi*t1)
    X1[:, 1] = np.sin(4*np.pi*t1)
    X2 = np.zeros((N, 2))
    X2[:, 0] = np.cos(2*np.pi*t2)
    X2[:, 1] = np.sin(4*np.pi*t2)

    Ps = getCTWAlignments(eng, X1, X2)
    types = Ps.keys()
    plt.hold(True)
    for i in range(len(types)):
        print(types[i])
        P = Ps[types[i]]
        plt.plot(P[:, 0], P[:, 1])
    plt.legend([t[1::] for t in types])
    plt.show()

if __name__ == '__main__':
    eng = initMatlabEngine()
    testCTWAlignments(eng)
