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

if __name__ == '__main__':
    np.random.seed(100)
    M = 200
    N = 200
    t1 = np.linspace(0, 1, M)
    t2 = np.linspace(0, 1, N)
    t2 = t2**2
    X = getPinchedCircle(t1)
    Y = getPinchedCircle(t2)
    Y = applyRandomRigidTransformation(Y)
    Y = Y + np.array([[4, 0]])

    Kappa = 0.1
    NRelMag = 5
    NBumps = 3
    #(Y, Bumps) = addRandomBumps(Y, Kappa, NRelMag, NBumps)

    SSMX = getCSM(X, X)
    SSMY = getCSM(Y, Y)
    D = doIBDTW(SSMX, SSMY)

    (DAll, CSM, backpointers, involved) = DTWCSM(D)
    cost = DAll[-1, -1]

    c = plt.get_cmap('Spectral')
    C1 = c(np.array(np.round(255*np.arange(M)/float(M)), dtype=np.int32))
    C1 = C1[:, 0:3]
    idx = np.argsort(-involved, 0)[0, :]
    C2 = c(np.array(np.round(255*idx/float(M)), dtype=np.int32))
    C2 = C2[:, 0:3]

    sio.savemat("IBDTW.mat", {"X":X, "Y":Y, "SSMX":SSMX, "SSMY":SSMY, "D":D})

    plt.subplot(221)
    plt.imshow(SSMX, cmap = 'afmhot', interpolation = 'none')
    plt.subplot(222)
    plt.imshow(SSMY, cmap = 'afmhot', interpolation = 'none')

    plt.subplot(223)
    plt.imshow(D, cmap = 'afmhot', interpolation = 'none')
    plt.hold(True)
    [J, I] = np.meshgrid(np.arange(involved.shape[1]), np.arange(involved.shape[0]))
    J = J[involved == 1]
    I = I[involved == 1]
    plt.scatter(J, I, 20, 'c', edgecolor = 'none')
    plt.xlim([0, CSM.shape[1]])
    plt.ylim([CSM.shape[0], 0])
    plt.title("Cost = %g"%cost)

    plt.subplot(224)
    plt.scatter(X[:, 0], X[:, 1], 20, c=C1, edgecolor='none')
    plt.hold(True)
    plt.scatter(Y[:, 0], Y[:, 1], 20, c=C2, edgecolor='none')
    plt.axis('equal')

    plt.show()
