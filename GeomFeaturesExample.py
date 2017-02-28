"""
Generating curve examples showing off all of the geometric features used in
my thesis
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.interpolate as interp
import sys
sys.path.append('GeometricCoverSongs')
sys.path.append('GeometricCoverSongs/SequenceAlignment')
from CSMSSMTools import *
from SpectralMethods import *
from CurvatureTools import *
from SyntheticCurves import *
import scipy.sparse
import scipy.sparse.linalg as slinalg
from TDA import *
from fake_parula import *

"""
Curve                                              Velocity/Curvature/Torsion

SSM                    Diffused SSM                D2

Recurrence Plot        Laplacian Eigenmap          Rips H1 Z2/Z3

"""

def SSMToBinary(D, Kappa):
    N = D.shape[0]
    if Kappa == 0:
        return np.ones((N, N))
    #Take one additional neighbor to account for the fact
    #that the diagonal of the SSM is all zeros
    elif Kappa < 1:
        NNeighbs = int(np.round(Kappa*N))+1
    else:
        NNeighbs = Kappa+1
    NNeighbs = min(N, NNeighbs)
    cols = np.argsort(D, 1)
    temp, rows = np.meshgrid(np.arange(N), np.arange(N))
    cols = cols[:, 0:NNeighbs].flatten()
    rows = rows[:, 0:NNeighbs].flatten()
    ret = np.zeros((N, N))
    ret[rows, cols] = 1
    return ret

def getAdjacencyKappa(D, Kappa):
    B1 = SSMToBinary(D, Kappa)
    B2 = SSMToBinary(D.T, Kappa)
    ret = B1*B2.T
    np.fill_diagonal(ret, 0)
    return ret

def getD2(D, D2Samples, hmax):
    N = D.shape[0]
    [I, J] = np.meshgrid(np.arange(N), np.arange(N))
    res = np.histogram(D[I < J], bins = D2Samples, range = (0, hmax))
    D2 = res[0]
    bins = res[1]
    D2 = np.array(D2, dtype=np.float32)
    D2 = D2/np.sum(D2) #Normalize
    return (D2, bins)

if __name__ == '__main__':
    sigma = 2
    Kappa = 0.06
    plt.figure(figsize=(16, 16))
    N = 400
    t = np.linspace(0, 1, N+1)[0:N]
    X = getTorusKnot(3, 5, t); doRotate = False
    #X = getVivianiFigure8(1, t); doRotate = False
    
    if doRotate:
        np.random.seed(6)
        R = np.random.randn(3, 3)
        R, _, _ = np.linalg.svd(R)
        X = X.dot(R)
    

    plt.clf()
    (SSM, _) = getSSM(X, N)
    ax = plt.subplot2grid((3, 3), (0, 0), colspan=2, projection='3d')
    c = plt.get_cmap('Spectral')
    C = c(np.array(np.round(np.linspace(0, 255, X.shape[0])), dtype=np.int32))
    C = C[:, 0:3]
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=C, edgecolor='none')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.title("Viviani Figure 8, %i Samples"%N)
    #plt.title("3-5 Torus Knot, %i Samples"%N)
    plt.subplot(334)
    plt.imshow(SSM, cmap='afmhot')
    plt.title("SSM")

    #Do curvature
    plt.subplot(333)
    Curvs = getCurvVectors(X, 3, sigma, loop = True)
    V = np.sqrt(np.sum(Curvs[1]**2, 1))
    Cv = np.sqrt(np.sum(Curvs[2]**2, 1))
    T = np.sqrt(np.sum(Curvs[3]**2, 1))
    plt.plot(V, 'r', label='Velocity')
    plt.hold(True)
    plt.plot(Cv, 'b', label='Curvature')
    plt.plot(T, 'k', label='Torsion')
    plt.scatter(np.arange(N), -0.025*np.ones(N), 20, c=C, edgecolor='none')
    plt.ylim([-0.05, 2])
    plt.xlim([0, N])
    plt.legend()

    #Do Diffusion Maps
    XD = getDiffusionMap(SSM, Kappa)
    XD[:, -2::]
    (SSMD, _) = getSSM(XD, N)
    SSMD = SSMD*np.max(SSM)/np.max(SSMD) #Rescale diffusion map
    plt.subplot(335)
    plt.imshow(SSMD, interpolation = 'none', cmap='afmhot')
    plt.title('SSM Autotuned Diffusion Map')

    #Get D2 histograms
    NBins = 40
    hmax = np.max(SSM)
    (D2, bins) = getD2(SSM, NBins, hmax)
    plt.subplot(336)
    plt.plot(bins[0:len(D2)], D2, 'r')
    plt.title('D2 Distance Histogram')
    plt.xlabel('Distance')
    plt.ylabel('Probability')

    #Make recurrence plot
    A = getAdjacencyKappa(SSM, Kappa)
    plt.subplot(337)
    plt.imshow(1-(A + np.eye(N)), interpolation = 'none', cmap='gray')
    plt.title("Recurrence Plot, Kappa = %g"%Kappa)

    #Do Laplacian Eigenmaps from Recurrence plot
    deg = np.sum(A, 1)
    DEG = np.eye(N)
    np.fill_diagonal(DEG, deg)
    L = DEG - A
    L = scipy.sparse.csc_matrix(L)
    v0 = np.random.randn(L.shape[0], 3)
    w, v = slinalg.eigsh(L, k=3, sigma = 0, which = 'LM')
    plt.subplot(338)
    plt.scatter(v[:, 1], v[:, 2], c=C, edgecolor = 'none')
    ax = plt.gca()
    a =  0.15
    ax.set_axis_bgcolor((a, a, a))
    plt.title("Laplacian Eigenmap")

    #Do rips filtrations
    plt.subplot(339)
    PDs2 = doRipsFiltrationDM(SSM, 1, thresh = -1, coeff = 2)
    PDs3 = doRipsFiltrationDM(SSM, 1, thresh = -1, coeff = 3)
    print PDs2[1]
    H12 = plotDGM(PDs2[1], color = np.array([1.0, 0.0, 0.2]), label = 'Z2', sz = 100, axcolor = np.array([0.8]*3))
    plt.hold(True)
    H13 = plotDGM(PDs3[1], color = np.array([0.2, 0.0, 1.0]), label = 'Z3', sz = 50, axcolor = np.array([0.8]*3))
    plt.legend(handles=[H12, H13])
    plt.title("Persistence Diagram 1D Rips")
    
    #plt.savefig("VivianiFigure8.svg", dpi=150, bbox_inches='tight')
    plt.savefig("TorusKnot35.svg", dpi=150, bbox_inches='tight')
