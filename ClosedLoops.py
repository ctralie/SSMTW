import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from SlidingWindowVideoTDA.VideoTools import *
from Alignment.AllTechniques import *
from Alignment.AlignmentTools import *
from Alignment.Alignments import *
from Alignment.DTWGPU import *
from Alignment.ctw.CTWLib import *
from Alignment.SyntheticCurves import applyRandomRigidTransformation
from PaperFigures import makeColorbar

def LoopExample():
    np.random.seed(4)
    plt.figure(figsize=(15, 5))
    #for i in range(1, 8):
    #    for j in range(1, 21):
    for i in range(1, 2):
        for j in range(1, 21):
            x = sio.loadmat('bicego_data/Class%i_Sample%i.mat'%(i, j))['x']
            x = np.array(x, dtype = np.float64)
            x = x[0::3, :]
            N = x.shape[0]
            y = np.roll(x, 50, 0)

            WarpDict = getWarpDictionary(N)
            t2 = getWarpingPath(WarpDict, 4, False)
            y = getInterpolatedEuclideanTimeSeries(y, t2)
            y = applyRandomRigidTransformation(y, True)
            y = y + np.array([[30, -220]])

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

            #Walk along projected path until we've gone N samples along
            pathProj = projectPath(path, PCSWM.shape[0], PCSWM.shape[1], 1)
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
            plt.axis('off')
            #plt.scatter(np.zeros(N), np.arange(N), 20, np.arange(N), cmap = 'Spectral', edgecolor = 'none')
            #plt.scatter(np.zeros(N), N+np.arange(N), 20, np.arange(N), cmap = 'Spectral', edgecolor = 'none')
            #plt.scatter(np.arange(N), np.zeros(N), 20, np.arange(N), cmap = 'Spectral', edgecolor = 'none')
            #plt.scatter(N+np.arange(N), np.zeros(N), 20, np.arange(N), cmap = 'Spectral', edgecolor = 'none')

            x = np.concatenate((x, x), 0)
            x = x[pathProj[:, 0], :]
            y = np.concatenate((y, y), 0)
            y = y[pathProj[:, 1], :]
            plt.subplot(133)
            plt.scatter(x[:, 0], x[:, 1], 20, np.arange(N), cmap = 'Spectral', edgecolor = 'none')
            plt.scatter(y[:, 0], y[:, 1], 20, np.arange(N), cmap = 'Spectral', edgecolor = 'none')
            plt.axis('equal')
            plt.axis('off')
            plt.savefig("LoopPCSWM%i_%i.svg"%(i, j), bbox_inches = 'tight')

if __name__ == '__main__':
    initParallelAlgorithms()
    LoopExample()
