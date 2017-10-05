"""
Provides an interface to CUDA for running the parallel IBDTW and 
partial IBDTW algorithms
"""
import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
import pycuda.cumath
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.io as sio
import pkg_resources
import sys
import Alignment
from Alignment.Alignments import *
from Alignment.AlignmentTools import *
import Alignment._SequenceAlignment as SAC

from pycuda.compiler import SourceModule

Alignment.DTW_ = None
Alignment.DTWSSM_ = None
Alignment.SMWat_ = None
Alignment.SMWatSSM_ = None

def getResourceString(filename):
    s = ''
    if 'Alignment' in sys.modules:
        s = pkg_resources.resource_string('Alignment', '/%s'%filename)
    elif 'SSMTW.Alignment' in sys.modules:
        s = pkg_resources.resource_string('SSMTW.Alignment', '/%s'%filename)
    else:
        #If calling from within this directory
        fin = open(filename)
        s = fin.read()
        fin.close()
    return s.decode('utf8')

def initParallelAlgorithms():
    s = getResourceString("DTWGPU.cu")
    mod = SourceModule(s)
    Alignment.DTW_ = mod.get_function("DTW")

    s = getResourceString("DTWSSMGPU.cu")
    mod = SourceModule(s)
    Alignment.DTWSSM_ = mod.get_function("DTWSSM")

    s = getResourceString("SMWatGPU.cu")
    mod = SourceModule(s)
    Alignment.SMWat_ = mod.get_function("SMWat")

    s = getResourceString("SMWatSSMGPU.cu")
    mod = SourceModule(s)
    Alignment.SMWatSSM_ = mod.get_function("SMWatSSM")

def roundUpPow2(x):
    return np.array(int(2**np.ceil(np.log2(float(x)))), dtype=np.int32)

def doDTWGPU(CSM, ci, cj):
    #Minimum dimension of array can be at max size 1024
    #for this scheme to fit in memory
    M = CSM.shape[0]
    N = CSM.shape[1]

    diagLen = np.array(min(M, N), dtype = np.int32)
    diagLenPow2 = roundUpPow2(diagLen)
    NThreads = min(diagLen, 512)
    res = gpuarray.to_gpu(np.array([0.0], dtype=np.float32))
    M = np.array(M, dtype=np.int32)
    N = np.array(N, dtype=np.int32)
    ci = np.array(ci, dtype = np.int32)
    cj = np.array(cj, dtype = np.int32)
    Alignment.DTW_(CSM, M, N, ci, cj, diagLen, diagLenPow2, res, block=(int(NThreads), 1, 1), grid=(1, 1), shared=12*diagLen)
    ret = res.get()[0]
    return ret

def doIBDTWGPU(SSMA, SSMB, returnCSM = False, printElapsedTime = False):
    """
    :param SSMA: MxM self-similarity matrix of first curve (gpuarray)
    :param SSMB: NxN self-similarity matrix of second curve (gpuarray)
    :param returnCSM: If True, return the CSM.  If false, just return the final cost
    :param printElapsedTime: Print the elapsed time
    """
    M = SSMA.shape[0]
    N = SSMB.shape[0]
    if not type(SSMA) == gpuarray:
        SSMA = gpuarray.to_gpu(np.array(SSMA, dtype = np.float32))
    if not type(SSMB) == gpuarray:
        SSMB = gpuarray.to_gpu(np.array(SSMB, dtype = np.float32))

    CSM = np.zeros((M, N), dtype=np.float32)
    CSM = gpuarray.to_gpu(CSM)

    diagLen = np.array(min(M, N), dtype = np.int32)
    diagLenPow2 = roundUpPow2(diagLen)
    NThreads = min(diagLen, 512)

    M = np.array(M, dtype=np.int32)
    N = np.array(N, dtype=np.int32)
    tic = time.time()

    Alignment.DTWSSM_(SSMA, SSMB, CSM, M, N, diagLen, diagLenPow2, block=(int(NThreads), 1, 1), grid=(int(M), int(N)), shared=12*diagLen)
    if returnCSM:
        return CSM.get()
    else:
        res = doDTWGPU(CSM, 0, 0)
        if printElapsedTime:
            print("Elapsed Time GPU: ", time.time() - tic)
        return res

def doSMWatGPU(CSM, hvPenalty):
    #Minimum dimension of array can be at max size 1024
    #for this scheme to fit in memory
    M = CSM.shape[0]
    N = CSM.shape[1]

    D = np.zeros((M, N), dtype=np.float32)
    D = gpuarray.to_gpu(D)

    U = np.zeros((M, N), dtype=np.float32)
    U = gpuarray.to_gpu(U)

    L = np.zeros((M, N), dtype=np.float32)
    L = gpuarray.to_gpu(L)

    UL = np.zeros((M, N), dtype=np.float32)
    UL = gpuarray.to_gpu(UL)

    phvPenalty = np.array(hvPenalty, dtype = np.float32)

    diagLen = np.array(min(M, N), dtype = np.int32)
    diagLenPow2 = roundUpPow2(diagLen)
    NThreads = min(diagLen, 512)
    M = np.array(M, dtype=np.int32)
    N = np.array(N, dtype=np.int32)
    Alignment.SMWat_(CSM, D, U, L, UL, M, N, diagLen, diagLenPow2, phvPenalty, block=(int(NThreads), 1, 1), grid=(1, 1), shared=12*diagLen)
    return {'D':D.get(), 'U':U.get(), 'L':L.get(), 'UL':UL.get()}

def doIBSMWatGPUHelper(SSMA, SSMB, hvPenalty, flip = False):
    """
    :param SSMA: MxM self-similarity matrix of first curve (gpuarray)
    :param SSMB: NxN self-similarity matrix of second curve (gpuarray)
    """
    M = SSMA.shape[0]
    N = SSMB.shape[0]

    CSM = np.zeros((M, N), dtype=np.float32)
    CSM = gpuarray.to_gpu(CSM)

    diagLen = np.array(min(M, N), dtype = np.int32)
    diagLenPow2 = roundUpPow2(diagLen)
    NThreads = min(diagLen, 512)

    M = np.array(M, dtype=np.int32)
    N = np.array(N, dtype=np.int32)
    pflip = np.array(0, dtype=np.int32)
    if flip:
        pflip = np.array(1, dtype=np.int32)
    phvPenalty = np.array(hvPenalty, dtype = np.float32)

    Alignment.SMWatSSM_(SSMA, SSMB, CSM, M, N, diagLen, diagLenPow2, phvPenalty, pflip, block=(int(NThreads), 1, 1), grid=(int(M), int(N)), shared=12*diagLen)
    CSM = CSM.get()
    return CSM

def flrud(A):
    return np.fliplr(np.flipud(A))

def doIBSMWatGPU(SSMA, SSMB, hvPenalty, printElapsedTime = False):
    tic = time.time()
    if not type(SSMA) == gpuarray:
        SSMA = gpuarray.to_gpu(np.array(SSMA, dtype = np.float32))
    if not type(SSMB) == gpuarray:
        SSMB = gpuarray.to_gpu(np.array(SSMB, dtype = np.float32))
    CSM = doIBSMWatGPUHelper(SSMA, SSMB, hvPenalty, False)
    CSM = CSM + flrud(doIBSMWatGPUHelper(SSMA, SSMB, hvPenalty, True))
    if printElapsedTime:
        print("Elapsed Time Smith Waterman GPU: %g"%(time.time() - tic))
    return CSM


