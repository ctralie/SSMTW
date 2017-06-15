import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import sys
sys.path.append('..')
sys.path.append('../GeometricCoverSongs')
sys.path.append('../GeometricCoverSongs/SequenceAlignment')
from CSMSSMTools import *
from SyntheticCurves import *
from Alignments import *
from DTWGPU import *

if __name__ == '__main__':
    initParallelAlgorithms()
    np.random.seed(10)
    M = 70
    N = 70
    t1 = np.linspace(0, 1, M)
    t2 = np.linspace(0, 1, N)
    t2 = t2**2
    X = getPinchedCircle(t1)
    Y = getPinchedCircle(t2)
    Y = applyRandomRigidTransformation(Y)
    Y = Y + np.array([[1, 4]])

    Kappa = 0.1
    NRelMag = 5
    NBumps = 3
    #(Y, Bumps) = addRandomBumps(Y, Kappa, NRelMag, NBumps)

    SSMX = np.array(getCSM(X, X), dtype=np.float32)
    SSMY = np.array(getCSM(Y, Y), dtype=np.float32)
    #SSMX = getRankSSM(SSMX)
    #SSMY = getRankSSM(SSMY)

    SSMX = get2DRankSSM(SSMX)
    SSMY = get2DRankSSM(SSMY)

    #SSMX = getZNormSSM(SSMX)
    #SSMY = getZNormSSM(SSMY)
    tic = time.time()
    D = doIBDTW(SSMX, SSMY)
    print "Elapsed Time CPU: ", time.time() - tic
    gSSMX = gpuarray.to_gpu(np.array(SSMX, dtype = np.float32))
    gSSMY = gpuarray.to_gpu(np.array(SSMY, dtype = np.float32))
    D2 = doIBDTWGPU(gSSMX, gSSMY, True, True)
    resGPU = doIBDTWGPU(gSSMX, gSSMY, False, True)

    sio.savemat("D.mat", {"D":D, "D2":D2})

    plt.subplot(131)
    plt.imshow(D, cmap = 'afmhot')
    plt.subplot(132)
    plt.imshow(D2, cmap = 'afmhot')
    plt.subplot(133)
    plt.imshow(D - D2, cmap = 'afmhot')
    plt.show()

    (DAll, CSM, backpointers, path) = DTWCSM(D)
    resCPU = DAll[-1, -1]
    print "GPU Result: ", resGPU
    print "CPU Result: ", resCPU

    c = plt.get_cmap('Spectral')
    C1 = c(np.array(np.round(255*np.arange(M)/float(M)), dtype=np.int32))
    C1 = C1[:, 0:3]
    idx = projectPath(path)
    C2 = c(np.array(np.round(255*idx/float(M)), dtype=np.int32))
    C2 = C2[:, 0:3]

    sio.savemat("IBDTW.mat", {"X":X, "Y":Y, "SSMX":SSMX, "SSMY":SSMY, "D":D})

    plt.figure(figsize=(10, 10))
    plt.subplot(221)
    plt.imshow(SSMX, cmap = 'afmhot', interpolation = 'nearest')
    plt.axis('off')
    plt.title('SSM 1')
    plt.subplot(222)
    plt.imshow(SSMY, cmap = 'afmhot', interpolation = 'nearest')
    plt.axis('off')
    plt.title('SSM 2')

    plt.subplot(223)
    plt.imshow(D, cmap = 'afmhot', interpolation = 'nearest')
    plt.hold(True)
    plt.scatter(path[:, 1], path[:, 0], 5, 'c', edgecolor = 'none')
    plt.xlim([0, CSM.shape[1]])
    plt.ylim([CSM.shape[0], 0])
    plt.axis('off')
    #plt.title("Cost = %g"%resGPU)
    plt.title("Cross-Similarity Warp Matrix")

    plt.subplot(224)
    plt.scatter(X[:, 0], X[:, 1], 3, c=C1, edgecolor='none')
    plt.hold(True)
    plt.scatter(Y[:, 0], Y[:, 1], 3, c=C2, edgecolor='none')
    plt.axis('equal')
    plotbgcolor = (0.15, 0.15, 0.15)
    ax = plt.gca()
    ax.set_axis_bgcolor(plotbgcolor)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    plt.xlim([-3.5, 6])
    plt.ylim([-2.5, 6.5])
    plt.title("TOPCs")

    plt.savefig("IBDTWExample.svg", bbox_inches='tight')
