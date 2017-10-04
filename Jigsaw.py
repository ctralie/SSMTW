"""
Programmer: Chris Tralie
Purpose: To provide functions for loading paths from SVG files and
using them to make simple jigsaw puzzle pieces
"""
import numpy as np
import matplotlib.pyplot as plt
from svg.path import parse_path
import xml.etree.ElementTree as ET

def loadSVGPaths(filename = "paths.svg"):
    """
    Given an SVG file, find all of the paths and load them into
    a dictionary of svg.path objects
    :param filename: Path to svg file
    :returns Paths: A dictionary of svg path objects indexed by ID
        specified in the svg file
    """
    tree = ET.parse(filename)
    root = tree.getroot()
    Paths = {}
    for c in root.getchildren():
        if c.tag == '{http://www.w3.org/2000/svg}g':
            for elem in c.getchildren():
                if elem.tag == '{http://www.w3.org/2000/svg}path':
                    Paths[elem.attrib['id']] = parse_path(elem.attrib['d'])
    return Paths

def paramSVGPath(P, t):
    """
    Parameterize an SVG path and return it as a numpy array
    :param P: The SVG path object
    :param t: N values between [0, 1] for parameterizing path
    :returns X: An Nx2 array of path points
    """
    N = t.size
    X = np.zeros((N, 2))
    for i in range(N):
        res = P.point(t[i])
        X[i, 0] = res.real
        X[i, 1] = res.imag
    return X

def make2JigsawPieces(P, tL, tR, t, doPlot = False):
    """
    Given a path, create two complementary jigsaw puzzle pieces
        with the path as the boundary connecting it
    :param P: An svg path object
    :param tL: M-sample parameterization of the boundary along the left piece
    :param tR: N-sample parameterization of the boundary along the right piece
    :param t: Fraction along 1x2 rectangle where centroid of path is placed
    :param doPlot: Whether to plot the pieces
    :return (X1, X2): An Mx2 and an Nx2 array parameterizing the boundaries
        of the left piece and the right piece.  The left piece is parameterized
        in counter-clockwise order, while the right piece is parameterized in
        clockwise order
    """
    XL = paramSVGPath(P, tL)
    XR = paramSVGPath(P, tR)
    #Make sure paths go from bottom to top
    if XL[0, 1] > XL[-1, 1]:
        XL = np.flipud(XL)
    if XR[0, 1] > XR[-1, 1]:
        XR = np.flipud(XR)

    #Figure out height
    H = np.max(XL[:, 1]) - np.min(XL[:, 1])

    #Center curves
    xCenter = t*2*H - np.mean(XL[:, 0])
    XL[:, 0] += xCenter
    XR[:, 0] += xCenter
    XR[:, 1] -= np.min(XL[:, 1])
    XL[:, 1] -= np.min(XL[:, 1])

    #Setup left puzzle piece
    #Bottom Edge
    b = np.linspace(0, XL[0, 0], XL.shape[0]/4)
    B = np.zeros((len(b), 2))
    B[:, 0] = b
    B = B[1::, :]
    #Left Edge
    L = np.zeros((len(b), 2))
    L[:, 1] = np.linspace(H, 0, len(b))
    #Top Edge
    T = np.zeros((len(b), 2))
    T[:, 1] = H
    T[:, 0] = np.linspace(XL[-1, 0], 0, len(b))
    XL = np.concatenate((T, L, B, XL), 0)

    #Setup right puzzle piece
    #Top Edge
    T = np.zeros((len(b), 2))
    T[:, 1] = H
    T[:, 0] = np.linspace(XR[-1, 0], 2*H, len(b))
    #Right edge
    R = np.zeros((len(b), 2))
    R[:, 0] = 2*H
    R[:, 1] = np.linspace(H, 0, len(b))
    #Bottom edge
    B = np.zeros((len(b), 2))
    B[:, 1] = 0
    B[:, 0] = np.linspace(2*H, XR[0, 0], len(b))
    XR = np.concatenate((XR, T, R, B), 0)
    XR[:, 0] += H

    if doPlot:
        plt.scatter(XL[:, 0], XL[:, 1], 20, np.arange(XL.shape[0]), cmap = 'spectral')
        plt.scatter(XR[:, 0], XR[:, 1], 20, np.arange(XR.shape[0]), cmap = 'spectral')
        plt.axis('equal')
        plt.show()

    return (XL, XR)


if __name__ == '__main__':
    from SyntheticCurves import applyRandomRigidTransformation
    from Alignment.AlignmentTools import getSSM, projectPath, getProjectedPathParam
    from Alignment.DTWGPU import initParallelAlgorithms, doIBSMWatGPU
    from Alignment.Alignments import SMWat
    initParallelAlgorithms()

    np.random.seed(22)
    Paths = loadSVGPaths()
    t1 = np.linspace(0, 1, 100)
    t2 = np.linspace(0, 1, 220)
    t2 = t2**2
    (XL, XR) = make2JigsawPieces(Paths[Paths.keys()[-1]], t1, t2, 0.4)
    XR = applyRandomRigidTransformation(XR, special = True)
    XR += np.array([-900, 800])
    DL = getSSM(XL)
    DR = getSSM(XR)
    M = np.max(DL)
    DL = DL/M
    DR = DR/M
    CSWM = doIBSMWatGPU(DL, DR, -0.3, True)
    #CSWM = np.exp(-CSWM/(np.mean(CSWM)))
    CSWM = CSWM - np.median(CSWM)
    CSWM = CSWM/np.max(np.abs(CSWM))
    #plt.imshow(CSWM, interpolation = 'none')
    #plt.show()

    matchfn = lambda x: x
    hvPenalty = -0.4
    M = CSWM.shape[0]
    N = CSWM.shape[1]
    print("M = %i, N = %i"%(M, N))

    #Insert piece
    res = SMWat(CSWM, matchfn, hvPenalty, backtrace = True)
    P = res['path']
    P = np.array(P)
    score1 = res['pathScore']
    path = projectPath(P, M, N)
    res = getProjectedPathParam(path)

    res2 = SMWat(CSWM, matchfn, hvPenalty, backtrace = True, backidx = [75, 285])
    P2 = res2['path']
    P2 = np.array(P2)
    score2 = res2['pathScore']
    path2 = projectPath(P2, M, N)
    res2 = getProjectedPathParam(path2)

    plt.figure(figsize=(11, 10))
    plt.subplot(321)
    plt.scatter(XL[:, 0], XL[:, 1], 20, np.arange(M), cmap = 'Reds', edgecolor = 'none')
    plt.scatter(XR[:, 0], XR[:, 1], 20, np.arange(N), cmap = 'Blues', edgecolor = 'none')
    plt.axis('equal')
    plt.title("Jigsaw Pieces")
    ax = plt.gca()
    ax.set_axis_bgcolor((0.7, 0.7, 0.7))
    plt.axis('off')

    plt.subplot(322)
    plt.imshow(CSWM, cmap = 'afmhot', interpolation = 'nearest')
    plt.scatter(np.arange(N), -4*np.ones(N), 80, 'k')
    plt.scatter(np.arange(N), -4*np.ones(N), 40, np.arange(N), edgecolor = 'none', cmap = 'Blues')
    plt.scatter(-4*np.ones(M), np.arange(M), 100, 'k')
    plt.scatter(-4*np.ones(M), np.arange(M), 80, np.arange(M), edgecolor = 'none', cmap = 'Reds')
    plt.xlim([-7, N])
    plt.ylim([M, -7])
    plt.title("PCSWM")



    plt.subplot(323)
    plt.scatter(XL[:, 0], XL[:, 1], 20, 'k')
    plt.scatter(XL[path[0, 0]:path[-1, 0]+1, 0], XL[path[0, 0]:path[-1, 0]+1, 1], 20, c = res['C1'], edgecolor = 'none')
    plt.scatter(XR[:, 0], XR[:, 1], 20, 'k')
    plt.scatter(XR[path[0, 1]:path[-1, 1]+1, 0], XR[path[0, 1]:path[-1, 1]+1, 1], 20, c = res['C2'], edgecolor = 'none')
    plt.axis('equal')
    plt.title("Jigsaw Pieces Match 1")
    plt.axis('off')
    ax = plt.gca()
    ax.set_axis_bgcolor((0.7, 0.7, 0.7))

    plt.subplot(324)
    plt.imshow(CSWM, cmap = 'afmhot', interpolation = 'nearest')
    plt.xlim([-2, N])
    plt.ylim([CSWM.shape[0], -2])
    plt.title("PCSWM Backtrace 1, Score = %.3g"%score1)
    plt.scatter(P[:, 1], P[:, 0], 30, 'k')
    plt.scatter(path[:, 1], path[:, 0], 20, c = res['C2'], edgecolor = 'none')




    plt.subplot(325)
    plt.scatter(XL[:, 0], XL[:, 1], 20, 'k')
    plt.scatter(XL[path2[0, 0]:path2[-1, 0]+1, 0], XL[path2[0, 0]:path2[-1, 0]+1, 1], 20, c = res2['C1'], edgecolor = 'none')
    plt.scatter(XR[:, 0], XR[:, 1], 20, 'k')
    plt.scatter(XR[path2[0, 1]:path2[-1, 1]+1, 0], XR[path2[0, 1]:path2[-1, 1]+1, 1], 20, c = res2['C2'], edgecolor = 'none')
    plt.axis('equal')
    plt.title("Jigsaw Pieces Match 2")
    plt.axis('off')
    ax = plt.gca()
    ax.set_axis_bgcolor((0.7, 0.7, 0.7))

    plt.subplot(326)
    plt.imshow(CSWM, cmap = 'afmhot', interpolation = 'nearest')
    plt.xlim([-2, N])
    plt.ylim([CSWM.shape[0], -2])
    plt.title("PCSWM Backtrace 2, Score = %.3g"%score2)
    plt.scatter(P2[:, 1], P2[:, 0], 30, 'k')
    plt.scatter(path2[:, 1], path2[:, 0], 20, c = res2['C2'], edgecolor = 'none')

    plt.savefig("Jigsaw.svg", bbox_inches = 'tight')
