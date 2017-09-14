import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from SyntheticCurves import *
from Alignment.Alignments import *
from Alignment.DTWGPU import *

if __name__ == '__main__':
    initParallelAlgorithms()
    np.random.seed(10)
    M = 200
    N = 200
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
    #D = doIBDTW(SSMX, SSMY)
    print("Elapsed Time CPU: %g"%(time.time() - tic))
    D2 = doIBDTWGPU(SSMX, SSMY, True, True)
    D = D2
    resGPU = doIBDTWGPU(SSMX, SSMY, False, True)

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
    print("GPU Result: %g"%resGPU)
    print("CPU Result: %g"%resCPU)


    #Project path
    pidx = projectPath(path, M, N)
    res = getProjectedPathParam(path, pidx, 'Spectral')

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
    plt.scatter(X[:, 0], X[:, 1], 3, c=res['C1'], edgecolor='none')
    plt.hold(True)
    plt.scatter(Y[:, 0], Y[:, 1], 3, c=res['C2'], edgecolor='none')
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
