"""
Some test to make sure the GPU algorithms return the same answers
as the CPU algorithms
"""
import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.io as sio
import pkg_resources
import sys
from Alignment.Alignments import *
from Alignment.AlignmentTools import *
from Alignment.DTWGPU import *
import Alignment._SequenceAlignment as SAC

def DTWGPUExample():
    initParallelAlgorithms()
    t1 = np.linspace(0, 1, 500)
    t1 = np.sqrt(t1)
    t2 = np.linspace(0, 1, 500)
    ci = 100
    cj = 20
    #t2 = np.sqrt(t2)
    #t1 = t1**2

    X = np.zeros((len(t1), 2))
    X[:, 0] = t1
    X[:, 1] = np.cos(4*np.pi*t1) + t1
    Y = np.zeros((len(t2), 2))
    Y[:, 0] = t2
    Y[:, 1] = np.cos(4*np.pi*t2) + t2 + 0.5

    tic = time.time()
    (DCPU, CSM, backpointers, involved) = DTWConstrained(X, Y, lambda x,y: np.sqrt(np.sum((np.array(x, dtype=np.float32)-np.array(y, dtype=np.float32))**2)), ci, cj)
    # print(Elapsed Time Python: %g"%(time.time() - tic))
    # DCPU = DCPU[1::, 1::]
    resPython = DCPU[-1, -1]
    CSM = getCSM(X, Y)

    tic = time.time()
    resC = SAC.DTWConstrained(CSM, ci, cj)
    print("Elapsed Time C: ", time.time() - tic)

    CSM = np.array(CSM, dtype = np.float32)
    CSM = gpuarray.to_gpu(CSM)
    resGPU = doDTWGPU(CSM, ci, cj)


    print("Python Result: %g"%resPython)
    print("C Result: %g"%resC)
    print("GPU Result: %g"%resGPU)

def IBDTWGPUExample():
    initParallelAlgorithms()

    np.random.seed(100)
    t1 = np.linspace(0, 1, 150)
    X1 = np.zeros((len(t1), 2))
    X1[:, 0] = np.cos(2*np.pi*t1)
    X1[:, 1] = np.sin(4*np.pi*t1)
    t2 = t1**2
    X2 = 0*X1
    X2[:, 0] = np.cos(2*np.pi*t2)
    X2[:, 1] = np.sin(4*np.pi*t2)

    SSMA = get2DRankSSM(getSSM(X1))
    SSMB = get2DRankSSM(getSSM(X2))
    M = SSMA.shape[0]
    N = SSMB.shape[0]

    tic = time.time()
    CSMG = doIBDTWGPU(SSMA, SSMB, True)
    print("Elapsed Time GPU: %g"%(time.time() - tic))

    plt.imshow(CSMG, aspect = 'auto', cmap = 'afmhot', interpolation = 'none')
    plt.title("GPU")
    plt.show()

def IBSMWatGPUExample():
    initParallelAlgorithms()

    np.random.seed(100)

    """
    #Shorter example
    t = np.linspace(0, 1, 30)
    t1 = t
    X1 = 0.3*np.random.randn(40, 2)
    X1[5:5+len(t1), 0] = np.cos(2*np.pi*t1)
    X1[5:5+len(t1), 1] = np.sin(4*np.pi*t1)
    t2 = t**2
    X2 = 0.3*np.random.randn(35, 2)
    X2[0:len(t2), 0] = np.cos(2*np.pi*t2)
    X2[0:len(t2), 1] = np.sin(4*np.pi*t2)
    """

    #"""
    #Longer example
    t = np.linspace(0, 1, 150)
    t1 = t
    X1 = 0.3*np.random.randn(200, 2)
    X1[50:50+len(t1), 0] = np.cos(2*np.pi*t1)
    X1[50:50+len(t1), 1] = np.sin(4*np.pi*t1)
    t2 = t**2
    X2 = 0.3*np.random.randn(180, 2)
    X2[0:len(t2), 0] = np.cos(2*np.pi*t2)
    X2[0:len(t2), 1] = np.sin(4*np.pi*t2)
    #"""

    SSMA = get2DRankSSM(getSSM(X1))
    SSMB = get2DRankSSM(getSSM(X2))
    M = SSMA.shape[0]
    N = SSMB.shape[0]

    matchfn = lambda x: np.exp(-x/(0.3**2))-0.6
    hvPenalty = -0.3

    tic = time.time()
    D1 = doIBSMWat(SSMA, SSMB, matchfn, hvPenalty)
    print("Elapsed Time CPU: %g"%(time.time() - tic))

    tic = time.time()
    D2 = doIBSMWatGPU(SSMA, SSMB, hvPenalty)
    print("Elapsed Time GPU: %g"%(time.time() - tic))

    plt.subplot(131)
    plt.imshow(D1, cmap = 'afmhot', interpolation = 'none')
    plt.title("CPU")
    plt.subplot(132)
    plt.imshow(D2, cmap = 'afmhot', interpolation = 'none')
    plt.title("GPU")
    plt.subplot(133)
    plt.imshow(D1 - D2, cmap = 'afmhot', interpolation = 'none')
    plt.title("Difference")
    plt.show()

    plt.show()


if __name__ == '__main__':
    #DTWGPUExample()
    #IBDTWGPUExample()
    IBSMWatGPUExample()
