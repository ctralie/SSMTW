import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import sys
sys.path.append('GeometricCoverSongs')
sys.path.append('GeometricCoverSongs/SequenceAlignment')
sys.path.append('MorseSSM')
from SSMTopological import *
from DGMTools import *
from CSMSSMTools import *
from SyntheticCurves import *
import os
import TDA

def plotDGM2(dgm, color = 'b', sz = 20, label = 'dgm', axcolor = np.array([0.0, 0.0, 0.0]), marker = None):
    if dgm.size == 0:
        return
    # Create Lists
    # set axis values
    axMin = np.min(dgm)
    axMax = np.max(dgm)
    axRange = axMax-axMin
    a = max(axMin - axRange/5, 0)
    b = axMax+axRange/5
    # plot line
    plt.plot([a, b], [a, b], c = axcolor, label = 'none')
    plt.hold(True)
    # plot points
    if marker:
        H = plt.scatter(dgm[:, 0], dgm[:, 1], sz, color, marker, label=label, edgecolor = 'none')
    else:
        H = plt.scatter(dgm[:, 0], dgm[:, 1], sz, color, label=label, edgecolor = 'none')
    # add labels
    plt.xlabel('Time of Birth')
    plt.ylabel('Time of Death')
    return H

def makeBG():
    ax = plt.gca()
    ax.set_axis_bgcolor((0.15, 0.15, 0.15))
    ax.set_xticks([])
    ax.set_yticks([])

def doWassersteinExample():
    N = 200
    t = np.linspace(0, 2*np.pi, N)
    X = np.zeros((N, 2))
    X[:, 0] = np.cos(t)
    X[:, 1] = np.sin(t)
    Y = X + 0.1*np.random.randn(N, 2)

    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.scatter(X[:, 0], X[:, 1], 20, 'b')
    plt.hold(True)
    plt.scatter(Y[:, 0], Y[:, 1], 20, 'r')
    plt.title("Circle And Noisy Circle")
    
    plt.subplot(122)
    I1 = TDA.doRipsFiltration(X, 1)[1]
    I2 = TDA.doRipsFiltration(Y, 2)[1]
    plotDGM2(I1, 'b')
    plt.hold(True)
    plotDGM2(I2, 'r')
    (matchidx, matchdist, D) = getWassersteinDist(I1, I2)
    plotWassersteinMatching(I1, I2, matchidx)
    plt.title("Wassersten Matching, Dist = %.3g"%matchdist)
    plt.savefig("NoisyCircleWasserstein.svg", bbox_inches = 'tight')

if __name__ == '__main__2':
    doWassersteinExample()

if __name__ == '__main__':
    plt.figure(figsize=(15, 4))
    N = 100
    t = np.linspace(0, 1, N)
    X = np.zeros((N, 3))
    X[:, 0] = np.cos(2*np.pi*t)
    X[:, 1] = np.sin(2*np.pi*t)
    #X[:, 2] = t/(6*np.pi)
    plt.subplot(131)
    plt.scatter(X[:, 0], X[:, 1], 20, np.arange(N), cmap = 'Spectral', edgecolor = 'none')
    makeBG()
    plt.title("Sampled Unit Circle")
    D = getCSM(X, X)
    c = SSMComplex(D)
    c.makeMesh()
    #c.ISplit = c.ISplit[:, [1, 0]]
    plt.subplot(132)
    c.plotMesh(False)
    plt.hold(True)
    c.plotCriticalPoints(sz = 50)
    plt.title("SSM with Critical Points")
    plt.subplot(133)
    s = plotDGM2(c.ISplit, 'r', label = 'Maxes')
    plt.hold(True)
    j = plotDGM2(c.IJoin, 'b', label = 'Mins')
    plt.legend(handles=[s, j], bbox_to_anchor=(0.9, 0.4))
    plt.title("Persistence Diagrams")
    plt.savefig("CircleCriticalPts.svg", bbox_inches = 'tight')
    
    
    plt.clf()
    N = 400
    t = np.linspace(0, 1, N)
    X = np.zeros((N, 2))
    X[:, 0] = np.cos(2*np.pi*t)
    X[:, 1] = np.sin(4*np.pi*t)
    plt.subplot(131)
    plt.scatter(X[:, 0], X[:, 1], 20, np.arange(N), cmap = 'Spectral', edgecolor = 'none')
    makeBG()
    plt.title("Figure 8")
    D = getCSM(X, X)
    c8 = SSMComplex(D)
    c8.makeMesh()
    #c.ISplit = c.ISplit[:, [1, 0]]
    plt.subplot(132)
    c8.plotMesh(False)
    plt.hold(True)
    c8.plotCriticalPoints(sz = 50)
    plt.title("SSM with Critical Points")
    plt.subplot(133)
    s = plotDGM2(c8.ISplit, 'r', label = 'Maxes')
    plt.hold(True)
    j = plotDGM2(c8.IJoin, 'b', label = 'Mins')
    plt.legend(handles=[s, j], bbox_to_anchor=(0.5, 0.95))
    plt.title("Persistence Diagrams")
    plt.savefig("Figure8CriticalPts.svg", bbox_inches = 'tight')
    
    
    #Now perturb the figure 8
    np.random.seed(1)
    Kappa = 0.1
    NRelMag = 2
    NBumps = 4
    X[:, 0] = np.cos(2*np.pi*t)
    X[:, 1] = np.sin(4*np.pi*t)
    (Y, Bumps) = addRandomBumps(X, Kappa, NRelMag, NBumps)
    D = getCSM(Y, Y)
    c82 = SSMComplex(D)
    c82.makeMesh()
    plt.clf()
    plt.subplot(131)
    plt.scatter(X[:, 0], X[:, 1], 20, np.arange(N), cmap = 'Spectral', edgecolor = 'none')
    makeBG()
    plt.title("Figure 8")
    
    plt.subplot(132)
    plt.scatter(Y[:, 0], Y[:, 1], 20, np.arange(N), cmap = 'Spectral', edgecolor = 'none')
    makeBG()
    plt.title("Figure 8 Perturbed")
    
    plt.subplot(133)
    plotDGM2(c8.ISplit, 'r')
    plt.hold(True)
    plotDGM2(c8.IJoin, 'b')
    plotDGM2(c82.ISplit, 'r', marker = 'x')
    plotDGM2(c82.IJoin, 'b', marker = 'x')
    (matchidx, matchdist, D) = getWassersteinDist(c8.IJoin, c82.IJoin)
    plotWassersteinMatching(c8.IJoin, c82.IJoin, matchidx)
    (matchidx, matchdist, D) = getWassersteinDist(c8.ISplit[:, [1, 0]], c82.ISplit[:, [1, 0]])
    plotWassersteinMatching(c8.ISplit, c82.ISplit, matchidx)
    plt.title("Wasserstein Matching")
    plt.savefig("PerturbedFigure8.svg", bbox_inches = 'tight')
    
    
    plt.clf()
    N = 400
    t = np.linspace(0, 1, N)
    X = getLissajousCurve(1, 1, 5, 4, 0, t)
    plt.subplot(131)
    plt.scatter(X[:, 0], X[:, 1], 20, np.arange(N), cmap = 'Spectral', edgecolor = 'none')
    makeBG()
    plt.title("Lissajous Curve")
    D = getCSM(X, X)
    c = SSMComplex(D)
    c.makeMesh()
    #c.ISplit = c.ISplit[:, [1, 0]]
    plt.subplot(132)
    c.plotMesh(False)
    plt.hold(True)
    c.plotCriticalPoints(sz = 50)
    plt.title("SSM with Critical Points")
    plt.subplot(133)
    s = plotDGM2(c.ISplit, 'r', label = 'Maxes')
    plt.hold(True)
    j = plotDGM2(c.IJoin, 'b', label = 'Mins')
    plt.legend(handles=[s, j], bbox_to_anchor=(0.5, 0.95))
    plt.title("Persistence Diagrams")
    plt.savefig("LissajousCriticalPts.svg", bbox_inches = 'tight')
