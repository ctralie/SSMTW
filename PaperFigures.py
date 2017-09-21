"""
Make some extra figures for the paper
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from SyntheticCurves import *
from Alignment.Alignments import *
from Alignment.AlignmentTools import *
from Alignment.DTWGPU import *

def makeIntroFigure():
    initParallelAlgorithms()
    plotbgcolor = (0.15, 0.15, 0.15)
    np.random.seed(2)
    M = 100
    N = 160
    WarpDict = getWarpDictionary(N)
    t1 = np.linspace(0, 1, M)
    t2 = getWarpingPath(WarpDict, 2, False)
    #X = getLissajousCurve(1, 1, 3, 2, 0, t1)
    #Y = getLissajousCurve(1, 1, 3, 2, 0, t2)
    #Y = Y + np.array([[2, -4]])

    X = getTschirnhausenCubic(1, t1)
    Y = getTschirnhausenCubic(1, t2)
    Y = applyRandomRigidTransformation(Y)
    Y = Y + np.array([-2.5, -1.5])

    SSMX = getSSM(X)
    SSMY = getSSM(Y)

    D = doIBDTWGPU(SSMX, SSMY, True, True)

    (DAll, CSM, backpointers, path) = DTWCSM(D)
    pathProj = projectPath(path, M, N, 1)
    print(pathProj)
    i11 = 15
    i12 = pathProj[i11, 1]
    i21 = 75
    i22 = pathProj[i21, 1]
    color1 = np.array([1.0, 0.0, 0.3])
    color2 = np.array([0.0, 0.5, 1.0])

    gridSize = (10, 12)

    plt.figure(figsize=(24, 4))
    #plt.subplot2grid(gridSize, (0, 0), colspan = 4, rowspan = 6)
    plt.subplot(151)
    plt.scatter(X[:, 0], X[:, 1], 20, np.arange(M), cmap = 'Spectral', edgecolor = 'none')
    plt.scatter(Y[:, 0], Y[:, 1], 20, np.arange(N), cmap = 'Spectral', edgecolor = 'none')
    plt.scatter(X[i11, 0], X[i11, 1], 150, color = color1, edgecolor = 'none')
    plt.scatter(X[i21, 0], X[i21, 1], 150, color = color2, edgecolor = 'none')
    plt.scatter(Y[i12, 0], Y[i12, 1], 150, color = color1, edgecolor = 'none')
    plt.scatter(Y[i22, 0], Y[i22, 1], 150, color = color2, edgecolor = 'none')
    plt.axis('equal')
    plt.title("Time-Ordered Point Clouds")
    ax = plt.gca()
    ax.set_axis_bgcolor(plotbgcolor)

    #plt.subplot2grid(gridSize, (0, 4), colspan = 4, rowspan = 6)
    plt.subplot(152)
    plt.imshow(SSMX, interpolation = 'nearest', cmap = 'gray')
    plt.plot(np.arange(M), i11*np.ones(M), color = color1, lineWidth=4)
    plt.plot(np.arange(M), i21*np.ones(M), color = color2, lineWidth=4)
    plt.scatter(-2*np.ones(M), np.arange(M), 50, np.arange(M), cmap = 'Spectral', edgecolor = 'none')
    plt.scatter(np.arange(M), -2*np.ones(M), 50, np.arange(M), cmap = 'Spectral', edgecolor = 'none')
    plt.xlim([-4, M])
    plt.ylim([M, -4])
    plt.title("SSM 1")
    plt.xlabel("Time Index")
    plt.ylabel("Time Index")

    #plt.subplot2grid(gridSize, (0, 8), colspan = 4, rowspan = 6)
    plt.subplot(153)
    plt.imshow(SSMY, interpolation = 'nearest', cmap = 'gray')
    plt.plot(np.arange(N), i12*np.ones(N), color = color1, lineWidth=4, lineStyle='--')
    plt.plot(np.arange(N), i22*np.ones(N), color = color2, lineWidth=4, lineStyle='--')
    plt.scatter(-2*np.ones(N), np.arange(N), 50, np.arange(N), cmap = 'Spectral', edgecolor = 'none')
    plt.scatter(np.arange(N), -2*np.ones(N), 50, np.arange(N), cmap = 'Spectral', edgecolor = 'none')
    plt.xlim([-5, N])
    plt.ylim([N, -5])
    plt.title("SSM 2")
    plt.xlabel("Time Index")
    plt.ylabel("Time Index")

    #plt.subplot2grid(gridSize, (6, 0), colspan = 6, rowspan = 4)
    plt.subplot(154)
    plt.plot(np.arange(M), SSMX[i11, :], color = color1)
    plt.plot(np.arange(N), SSMY[i12, :], color = color1, lineStyle='--')
    plt.ylim([0, np.max(SSMX)])
    ax = plt.gca()
    #ax.set_axis_bgcolor(plotbgcolor)
    plt.title("Correspondence 1")
    plt.xlabel("Time Index")
    plt.ylabel("Distance")
    plt.legend(["SSM 1 Row %i"%i11, "SSM 2 Row %i"%i12], fontsize=12, loc = (0.02, 0.8))

    plt.subplot(155)
    plt.plot(np.arange(M), SSMX[i21, :], color = color2)
    plt.plot(np.arange(N), SSMY[i22, :], color = color2, lineStyle='--')
    plt.ylim([0, np.max(SSMX)])
    ax = plt.gca()
    #ax.set_axis_bgcolor(plotbgcolor)
    plt.title("Correspondence 2")
    plt.xlabel("Time Index")
    plt.ylabel("Distance")
    plt.legend(["SSM 1 Row %i"%i21, "SSM 2 Row %i"%i22], fontsize=12, loc = (0.02, 0.8))

    plt.savefig("IntroFig.svg", bbox_inches = 'tight')

if __name__ == '__main__':
    makeIntroFigure()
