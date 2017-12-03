import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from SlidingWindowVideoTDA.VideoTools import *
from Alignment.AllTechniques import *
from Alignment.AlignmentTools import *
from Alignment.Alignments import *
from Alignment.DTWGPU import *
from Alignment.ctw.CTWLib import *
from Alignment.SyntheticCurves import *
from PaperFigures import makeColorbar

def LoopExperiments(SamplesPerCurve, Kappa = 0.1, NRelMag = 2, NBumps = 2, doPlot = False):
    np.random.seed(SamplesPerCurve)
    if doPlot:
        plt.figure(figsize=(15, 5))
    NClasses = 7
    CurvesPerClass = 20
    Scores = np.zeros((NClasses, CurvesPerClass, SamplesPerCurve))
    distortionRatios = []
    for i in range(1, NClasses+1):
        for j in range(1, CurvesPerClass+1):
            for k in range(SamplesPerCurve):
                x = sio.loadmat('bicego_data/Class%i_Sample%i.mat'%(i, j))['x']
                x = np.array(x, dtype = np.float64)
                x = x[0::3, :]
                N = x.shape[0]
                circshift = np.random.randint(N)
                yo = np.roll(x, circshift, 0)
                
                (y, Bumps) = addRandomBumps(yo, Kappa, NRelMag, NBumps)
                diff = np.sqrt(np.sum((yo-y)**2, 1))
                GHDist = np.max(diff)
                D = getCSM(y, y)
                distortionRatios.append(GHDist/np.max(D))

                WarpDict = getWarpDictionary(N)
                t2 = getWarpingPath(WarpDict, 4, False)
                y = getInterpolatedEuclideanTimeSeries(y, t2)
                y = applyRandomRigidTransformation(y, True)
                y = y + np.array([[30, -220]])

                #Ground truth path
                t2 = t2*N
                t1 = N*np.linspace(0, 1, N)
                PGT = np.zeros((len(t1), 2))
                PGT[:, 0] = np.mod(t2-circshift, N)
                PGT[:, 1] = t1

                D1 = getSSM(np.concatenate((x, x), 0))
                D2 = getSSM(np.concatenate((y, y), 0))

                D1 = D1/np.max(D1)
                D2 = D2/np.max(D2)
                PCSWM = doIBSMWatGPU(D1, D2, 0.3, True)
                matchfn = lambda x: x
                hvPenalty = -0.5
                PCSWM = PCSWM - np.median(PCSWM)
                PCSWM = PCSWM/np.max(np.abs(PCSWM))
                res = SMWat(PCSWM, matchfn, hvPenalty, backtrace = True)
                path = res['path']
                path[:, 0] = np.mod(path[:, 0], len(x))
                path[:, 1] = np.mod(path[:, 1], len(y))
                score = computeAlignmentError(PGT, path, etype = 2)
                Scores[i-1, j-1, k] = score
                sio.savemat("ClosedLoops.mat", {"Scores":Scores, "distortionRatios":np.array(distortionRatios)})
                if doPlot:
                    pathProj = projectPath(path, PCSWM.shape[0], PCSWM.shape[1], 1)
                    #Walk along projected path until we've gone N samples along
                    pathProj = pathProj[0:N, :]
                    
                    plt.clf()
                    plt.subplot(131)
                    plt.scatter(x[:, 0], x[:, 1], 20, np.arange(x.shape[0]), cmap = 'Spectral', edgecolor = 'none')
                    plt.scatter(y[:, 0], y[:, 1], 20, np.arange(x.shape[0]), cmap = 'Spectral', edgecolor = 'none')
                    plt.axis('equal')
                    plt.axis('off')
                    plt.subplot(132)
                    plt.imshow(PCSWM, cmap = 'gray')
                    plt.scatter(pathProj[:, 1], pathProj[:, 0], 20, 'c', edgecolor = 'none')
                    plt.scatter(PGT[:, 1], PGT[:, 0], 20, 'm', edgecolor = 'none')
                    plt.title("Score = %g"%score)
                    plt.axis('off')
                    plt.scatter(np.zeros(N), np.arange(N), 20, np.arange(N), cmap = 'Spectral', edgecolor = 'none')
                    plt.scatter(np.zeros(N), N+np.arange(N), 20, np.arange(N), cmap = 'Spectral', edgecolor = 'none')
                    plt.scatter(np.arange(N), np.zeros(N), 20, np.arange(N), cmap = 'Spectral', edgecolor = 'none')
                    plt.scatter(N+np.arange(N), np.zeros(N), 20, np.arange(N), cmap = 'Spectral', edgecolor = 'none')

                    x = np.concatenate((x, x), 0)
                    x = x[pathProj[:, 0], :]
                    y = np.concatenate((y, y), 0)
                    y = y[pathProj[:, 1], :]
                    plt.subplot(133)
                    plt.scatter(x[:, 0], x[:, 1], 20, np.arange(N), cmap = 'Spectral', edgecolor = 'none')
                    plt.scatter(y[:, 0], y[:, 1], 20, np.arange(N), cmap = 'Spectral', edgecolor = 'none')
                    plt.axis('equal')
                    plt.axis('off')
                    plt.savefig("LoopPCSWM%i_%i_%i.svg"%(i, j, k), bbox_inches = 'tight')

if __name__ == '__main__':
    initParallelAlgorithms()
    LoopExperiments(30, doPlot = False)
