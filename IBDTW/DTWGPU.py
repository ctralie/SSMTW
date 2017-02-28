import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
import pycuda.cumath
import skcuda.misc
import skcuda.linalg as linalg
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.io as sio
from Alignments import *

from pycuda.compiler import SourceModule

DTW_ = None
getSumSquares_ = None
finishCSM_ = None

def initParallelAlgorithms():
    global DTW_
    fin = open("DTWGPU.cu")
    mod = SourceModule(fin.read())
    fin.close()
    DTW_ = mod.get_function("DTW")

    #Run each of the algorithms on dummy data so that they're pre-compiled
    linalg.init()

def roundUpPow2(x):
    return np.array(int(2**np.ceil(np.log2(float(x)))), dtype=np.int32)

def getCSM(X, Y):
    XSqr = np.sum(X**2, 1)
    YSqr = np.sum(Y**2, 1)
    C = XSqr[:, None] + YSqr[None, :] - 2*X.dot(Y.T)
    C[C < 0] = 0
    return np.sqrt(C)

def doDTW(CSM):
    #Minimum dimension of array can be at max size 1024
    #for this scheme to fit in memory
    M = CSM.shape[0]
    N = CSM.shape[1]
    D = np.zeros((M, N), dtype=np.float32)
    D = gpuarray.to_gpu(D)
    diagLen = np.array(min(M, N), dtype = np.int32)
    diagLenPow2 = roundUpPow2(diagLen)
    NThreads = min(diagLen, 512)
    res = gpuarray.to_gpu(np.array([0.0], dtype=np.float32))
    M = np.array(M, dtype=np.int32)
    N = np.array(N, dtype=np.int32)
    tic = time.time()
    DTW_(CSM, D, M, N, diagLen, diagLenPow2, res, block=(int(NThreads), 1, 1), grid=(1, 1), shared=12*diagLen)
    print "Elapsed Time: ", time.time() - tic
    return (D.get(), res.get()[0])


if __name__ == '__main__':
    initParallelAlgorithms()
    np.random.seed(100)
    t1 = np.linspace(0, 1, 50)
    t1 = t1
    t2 = np.linspace(0, 1, 60)
    #t2 = np.sqrt(t2)
    #t1 = t1**2

    X = np.zeros((len(t1), 2))
    X[:, 0] = t1
    X[:, 1] = np.cos(4*np.pi*t1) + t1
    Y = np.zeros((len(t2), 2))
    Y[:, 0] = t2
    Y[:, 1] = np.cos(4*np.pi*t2) + t2 + 0.5

    (DCPU, CSM, backpointers, involved) = DTW(X, Y, lambda x,y: np.sqrt(np.sum((x-y)**2)))
    DCPU = DCPU[1::, 1::]
    resCPU = DCPU[-1, -1]
    CSM = np.array(CSM, dtype = np.float32)
    CSM = gpuarray.to_gpu(CSM)
    (DGPU, resGPU) = doDTW(CSM)

    plt.subplot(221)
    plt.imshow(DCPU, cmap='afmhot')
    plt.subplot(222)
    plt.imshow(DGPU, cmap='afmhot')
    plt.subplot(223)
    diff = DCPU - DGPU
    plt.imshow(diff, cmap='afmhot')
    plt.subplot(224)
    plt.imshow(np.abs(diff) < 1e-5)
    plt.show()

    sio.savemat("D.mat", {"D":DGPU})

    print "CPU Result: %g"%resCPU
    print "GPU Result: %g"%resGPU
