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
    from Alignment.AlignmentTools import getSSM
    from Alignment.DTWGPU import initParallelAlgorithms, doIBSMWatGPU
    initParallelAlgorithms()

    Paths = loadSVGPaths()
    t1 = np.linspace(0, 1, 100)
    t2 = t1**2
    (XL, XR) = make2JigsawPieces(Paths[Paths.keys()[-1]], t1, t2, 0.3)
    DL = getSSM(XL)
    DR = getSSM(XR)
    M = np.max(DL)
    DL = DL/M
    DR = DR/M
    CSWM = doIBSMWatGPU(DL, DR, -0.3, True)

    plt.subplot(221)
    plt.imshow(DL, cmap = 'afmhot', interpolation = 'nearest')
    plt.subplot(222)
    plt.imshow(DR, cmap = 'afmhot', interpolation = 'nearest')
    plt.subplot(223)
    plt.imshow(CSWM, cmap = 'afmhot', interpolation = 'nearest')
    plt.show()
