import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from Alignment.SyntheticCurves import *
from Alignment.AlignmentTools import *
from Alignment.DTWGPU import *
from Alignment.ctw.CTWLib import *
from Alignment.AllTechniques import *
from mpl_toolkits.mplot3d import Axes3D

def doExperiment(N, NPerClass, K, Kappa, NRelMag, NBumps, doPlots = False):
    """
    :param N: Number of points per synthetic curve
    :param NPerClass: Number of warped sampled curves per class
    :param K: Number of basis elements to combine (complexity of warping path)
    :param Kappa: Fraction of nearest neighbors to use when making bump
    :param NRelMag: Controls how large the bumps are
    :param NBumps: Number of bumps to add
    :param doPlots: Whether to plot all of the alignments paths from different
        algorithms for each trial
    """
    initParallelAlgorithms()
    eng = initMatlabEngine()
    np.random.seed(NPerClass)

    WarpDict = getWarpDictionary(N)
    Curves = {}
    Curves['VivianiFigure8'] = lambda t: getVivianiFigure8(0.5, t)
    Curves['TSCubic'] = lambda t: getTschirnhausenCubic(1, t)
    Curves['TorusKnot23'] = lambda t: getTorusKnot(2, 3, t)
    Curves['TorusKnot35'] = lambda t: getTorusKnot(3, 5, t)
    Curves['PinchedCircle'] = lambda t: getPinchedCircle(t)
    Curves['Lissajous32'] = lambda t: getLissajousCurve(1, 1, 3, 2, 0, t)
    Curves['Lissajous54'] = lambda t: getLissajousCurve(1, 1, 5, 4, 0, t)
    Curves['ConeHelix'] = lambda t: getConeHelix(1, 16, t)
    Curves['Epicycloid1_3'] = lambda t: getEpicycloid(1.5, 0.5, t)
    #Curves['Epicycloid1_4'] = lambda t: getEpicycloid(2, 0.5, t)

    plotbgcolor = (0.15, 0.15, 0.15)
    t1 = np.linspace(0, 1, N)
    distortionRatios = []
    AllErrors = {}
    if doPlots:
        plt.figure(figsize = (12, 6))
    for name in Curves:
        AllErrors = {}
        curve = Curves[name]
        X1 = curve(t1)
        print("Making %s..."%name)
        tic = time.time()
        for k in range(NPerClass):
            t2 = getWarpingPath(WarpDict, K, False)
            X2 = curve(t2)
            #Introduce some metric distortion
            (x, Bumps) = addRandomBumps(X2, Kappa, NRelMag, NBumps)
            diff = np.sqrt(np.sum((x-X2)**2, 1))
            GHDist = np.max(diff)
            D = getCSM(X2, X2)
            distortionRatios.append(GHDist/np.max(D))
            #Apply a random rigid transformation
            x = applyRandomRigidTransformation(x)
            X2 = x
            if doPlots:
                plt.clf()
                plt.subplot(121)
                X2 = X2 - (np.mean(X2, 0) - np.mean(X1, 0)) + 3*np.std(X1)
                plt.scatter(X1[:, 0], X1[:, 1], 20, np.arange(X1.shape[0]), cmap = 'Spectral', edgecolor = 'none')
                plt.scatter(X2[:, 0], X2[:, 1], 20, np.arange(X2.shape[0]), cmap = 'Spectral', edgecolor = 'none')
                ax = plt.gca()
                ax.set_axis_bgcolor(plotbgcolor)
                ax.set_xticks([])
                ax.set_yticks([])
                plt.axis('equal')
                plt.title("TOPCs")
                plt.subplot(122)
            (errors, Ps) = doAllAlignments(eng, X1, X2, t2, doPCA = 0, drawPaths = doPlots)
            if doPlots:
                plt.xlabel("TOPC 1")
                plt.ylabel("TOPC 2")
                plt.title("Warping Paths")
                plt.savefig("%s_%i.svg"%(name, k))
                sio.savemat("%s_%i.mat"%(name, k), Ps)
            types = errors.keys()
            for t in types:
                if not t in AllErrors:
                    AllErrors[t] = np.zeros(NPerClass)
                AllErrors[t][k] = errors[t]
            sio.savemat("%sErrors.mat"%name, AllErrors)
            sio.savemat("DistortionRatios.mat", {"distortionRatios":np.array(distortionRatios)})
        print("Elapsed Time %s: "%name, time.time() - tic)
    return AllErrors

if __name__ == '__main__2':
    initParallelAlgorithms()
    eng = initMatlabEngine()

    t1 = np.linspace(0, 1, 200)
    t2 = t1**2
    X1 = getVivianiFigure8(0.5, t1)
    X2 = getVivianiFigure8(0.5, t2)
    #getIBDTWAlignment(X1, X2, doPlot = True)
    doAllAlignments(eng, X1, X2, t2, drawPaths = True, drawAlignmentScores = True)

if __name__ == '__main__':
    N = 200
    NPerClass = 200
    K = 3
    Kappa = 0.1
    NRelMag = 2
    NBumps = 2
    AllErrors = doExperiment(N, NPerClass, K, Kappa, NRelMag, NBumps, doPlots = False)
