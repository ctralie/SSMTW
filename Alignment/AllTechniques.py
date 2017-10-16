import numpy as np
import matplotlib.pyplot as plt
from Alignment.AlignmentTools import *
from Alignment.Alignments import *
from Alignment.ctw.CTWLib import *
import time

def getIBDTWAlignment(X1, X2, L = 100, useGPU = True, Verbose = False, doPlot = False):
    """
    Get alignment path for SSMs and ranked SSMs
    :param X1: Euclidean point cloud 1
    :param X2: Euclidean point cloud 2
    :param L: Number of levels to use in histogram matching
    :param useGPU: Whether to use the GPU
    :param Verbose: Whether to print timing information
    :param doPlot: Whether to plot the SSMs, CSWMs, and alignment results
    """
    tic = time.time()
    D1 = getSSM(X1)
    D2 = getSSM(X2)
    (D1N1, D2N1) = matchSSMDist(D1, D2, L)
    (D2N2, D1N2) = matchSSMDist(D2, D1, L)
    if useGPU:
        from Alignment.DTWGPU import doIBDTWGPU
        D = doIBDTWGPU(D1, D2, returnCSM = True)
        DNorm1 = doIBDTWGPU(D1N1, D2N1, returnCSM = True)
        DNorm2 = doIBDTWGPU(D1N2, D2N2, returnCSM = True)
    else:
        D = doIBDTW(D1, D2)
        DNorm1 = doIBDTW(D1N1, D2N1)
        DNorm2 = doIBDTW(D1N2, D2N2)
    if Verbose:
        print("Elapsed Time GPU: %g"%(time.time() - tic))

    (DAll, CSSM, backpointers, path) = DTWCSM(D)
    (DAllN, CSSM1, backpointersN, pathN12) = DTWCSM(DNorm1)
    (DAllN, CSSM2, backpointersN, pathN21) = DTWCSM(DNorm2)

    if Verbose:
        print("Elapsed Time Total: %g"%(time.time() - tic))

    if doPlot:
        plt.figure(figsize=(10, 10))
        plt.subplot(331)
        plt.imshow(D1, cmap = 'afmhot', interpolation = 'nearest')
        plt.axis('off')
        plt.title('SSM 1')
        plt.subplot(334)
        plt.imshow(D2, cmap = 'afmhot', interpolation = 'nearest')
        plt.axis('off')
        plt.title('SSM 2')
        plt.subplot(337)
        plt.imshow(D, cmap = 'afmhot', interpolation = 'nearest')
        plt.hold(True)
        plt.scatter(path[:, 1], path[:, 0], 5, 'r', edgecolor = 'none')
        plt.xlim([0, D.shape[1]])
        plt.ylim([D.shape[0], 0])
        plt.axis('off')
        plt.title("CSWM Orig\nScore = %.3g"%D[-1, -1])

        plt.subplot(332)
        plt.imshow(D1N1, cmap = 'afmhot', interpolation = 'nearest')
        plt.axis('off')
        plt.title('SSM 1 Norm To 2')
        plt.subplot(335)
        plt.imshow(D2N1, cmap = 'afmhot', interpolation = 'nearest')
        plt.axis('off')
        plt.title('SSM 2')
        plt.subplot(338)
        plt.imshow(DNorm1, cmap = 'afmhot', interpolation = 'nearest')
        plt.hold(True)
        plt.scatter(pathN12[:, 1], pathN12[:, 0], 5, 'c', edgecolor = 'none')
        plt.xlim([0, DNorm1.shape[1]])
        plt.ylim([DNorm1.shape[0], 0])
        plt.axis('off')
        plt.title("CSWM 1 Norm To 2\nScore = %.3g"%CSSM1[-1, -1])

        plt.subplot(333)
        plt.imshow(D1N2, cmap = 'afmhot', interpolation = 'nearest')
        plt.axis('off')
        plt.title('SSM 1')
        plt.subplot(336)
        plt.imshow(D2N2, cmap = 'afmhot', interpolation = 'nearest')
        plt.axis('off')
        plt.title('SSM 2 Norm To 1')
        plt.subplot(339)
        plt.imshow(DNorm2, cmap = 'afmhot', interpolation = 'nearest')
        plt.hold(True)
        plt.scatter(pathN21[:, 1], pathN21[:, 0], 5, 'm', edgecolor = 'none')
        plt.xlim([0, DNorm2.shape[1]])
        plt.ylim([DNorm2.shape[0], 0])
        plt.axis('off')
        plt.title("CSWM 2 Norm To 1\nScore = %.3g"%CSSM2[-1, -1])

    pathN = pathN12
    #Return the normalized path with the best alignment score
    if CSSM2[-1, -1] < CSSM1[-1, -1]:
        pathN = pathN21
        if Verbose:
            print("\n\n\n-----------------\nPATH 2\n\n\n-----------------\n")
    elif Verbose:
        print("\n\n\n-----------------\nPATH 1\n\n\n-----------------\n")
    return (path, pathN)

def doAllAlignments(eng, X1, X2, t2, doPCA = 1, useGPU = True, drawPaths = False, drawAlignmentScores = False):
    tic = time.time()
    (PIBDTW, PIBDTWN) = getIBDTWAlignment(X1, X2, useGPU = useGPU)
    timeIBDTW = time.time() - tic
    tic = time.time()
    Ps = getCTWAlignments(eng, X1, X2, doPCA = doPCA)
    timeOthers = time.time() - tic
    print("IBDTW Time: %g, Others Time: %g"%(timeIBDTW, timeOthers))
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

    types = ['PGTW', 'PIBDTW', 'PIBDTWN', 'PCTW', 'PDTW', 'PDDTW', 'PIMW']
    if drawPaths:
        plt.hold(True)
        for i in range(len(types)):
            print(types[i])
            P = Ps[types[i]]
            plt.plot(P[:, 0], P[:, 1])
        plt.plot(PGT[:, 0], PGT[:, 1], 'k', lineWidth = 4, lineStyle = '--')
        plt.legend(["%s %.3g"%(t[1::], errors[t]) for t in types] + ["GroundTruth"], bbox_to_anchor=(0, 1), loc=2, fontsize=12)
        plt.xlim([0, np.max(PGT[:, 0])])
        plt.ylim([0, np.max(PGT[:, 1])])

    return (errors, Ps)
