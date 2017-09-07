import numpy as np
import matplotlib.pyplot as plt
from Alignment.AlignmentTools import *
from Alignment.Alignments import *
from Alignment.DTWGPU import *
from CTWLib import *
import time

def getIBDTWAlignment(X1, X2, useGPU = True, normFn = get2DRankSSM, Verbose = False, doPlot = False):
    """
    Get alignment path for SSMs and ranked SSMs
    """
    tic = time.time()
    SSM1 = np.array(getSSM(X1), dtype = np.float32)
    SSM2 = np.array(getSSM(X2), dtype = np.float32)
    SSM1Norm = np.array(normFn(SSM1), dtype = np.float32)
    SSM2Norm = np.array(normFn(SSM2), dtype = np.float32)
    if useGPU:
        import pycuda.gpuarray as gpuarray
        D = doIBDTWGPU(gpuarray.to_gpu(SSM1), gpuarray.to_gpu(SSM2), returnCSM = True)
        DNorm = doIBDTWGPU(gpuarray.to_gpu(SSM1Norm), gpuarray.to_gpu(SSM2Norm), returnCSM = True)
    else:
        D = doIBDTW(SSM1, SSM2)
        DNorm = doIBDTW(SSM1Norm, SSM2Norm)
    if Verbose:
        print "Elapsed Time GPU: ", time.time() - tic
    (DAll, CSM, backpointers, path) = DTWCSM(D)
    (DAllN, CSMN, backpointersN, pathN) = DTWCSM(DNorm)
    if Verbose:
        print "Elapsed Time Total: ", time.time() - tic

    if doPlot:
        plt.figure(figsize=(10, 10))
        plt.subplot(321)
        plt.imshow(SSM1, cmap = 'afmhot', interpolation = 'nearest')
        plt.axis('off')
        plt.title('SSM 1')
        plt.subplot(322)
        plt.imshow(SSM2, cmap = 'afmhot', interpolation = 'nearest')
        plt.axis('off')
        plt.title('SSM 2')

        plt.subplot(323)
        plt.imshow(SSM1Norm, cmap = 'afmhot', interpolation = 'nearest')
        plt.axis('off')
        plt.title('SSM 1 Norm')
        plt.subplot(324)
        plt.imshow(SSM2Norm, cmap = 'afmhot', interpolation = 'nearest')
        plt.axis('off')
        plt.title('SSM 2 Norm')

        plt.subplot(325)
        plt.imshow(D, cmap = 'afmhot', interpolation = 'nearest')
        plt.hold(True)
        plt.scatter(path[:, 1], path[:, 0], 5, 'r', edgecolor = 'none')
        plt.xlim([0, CSM.shape[1]])
        plt.ylim([CSM.shape[0], 0])
        plt.axis('off')
        plt.title("CSWM Orig")

        plt.subplot(326)
        plt.imshow(DNorm, cmap = 'afmhot', interpolation = 'nearest')
        plt.hold(True)
        plt.scatter(pathN[:, 1], pathN[:, 0], 5, 'c', edgecolor = 'none')
        plt.xlim([0, CSMN.shape[1]])
        plt.ylim([CSMN.shape[0], 0])
        plt.axis('off')
        plt.title("CSWM Normed")
    return (path, pathN)

def doAllAlignments(eng, X1, X2, t2, useGPU = True, drawPaths = False, drawAlignmentScores = False):
    tic = time.time()
    (PIBDTW, PIBDTWN) = getIBDTWAlignment(X1, X2, useGPU = useGPU)
    timeIBDTW = time.time() - tic
    tic = time.time()
    Ps = getCTWAlignments(eng, X1, X2)
    timeOthers = time.time() - tic
    print "IBDTW Time: %g, Others Time: %g"%(timeIBDTW, timeOthers)
    Ps['PIBDTW'] = PIBDTW
    Ps['PIBDTWN'] = PIBDTWN

    #Ground truth path
    t2 = t2*(X1.shape[0]-1)
    t1 = X2.shape[0]*np.linspace(0, 1, len(t2))
    PGT = np.zeros((len(t1), 2))
    PGT[:, 0] = t2
    PGT[:, 1] = t1

    types = Ps.keys()
    errors = {}
    for ptype in types:
        err = computeAlignmentError(Ps[ptype], PGT, doPlot = drawAlignmentScores)
        if drawAlignmentScores:
            plt.show()
        errors[ptype] = err
    
    types = ['PGTW', 'PIBDTWN']
    if drawPaths:
        plt.hold(True)
        for i in range(len(types)):
            print types[i]
            P = Ps[types[i]]
            plt.plot(P[:, 0], P[:, 1])
        plt.plot(PGT[:, 0], PGT[:, 1], 'k', lineWidth = 4, lineStyle = '--')
        plt.legend(["%s %.3g"%(t[1::], errors[t]) for t in types] + ["GT"])

    return (errors, Ps)
