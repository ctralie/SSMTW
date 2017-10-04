import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import sys
from Geom3D.PolyMesh import *
from SlidingWindowVideoTDA.VideoTools import *
from Alignment.AlignmentTools import *
from Alignment.DTWGPU import *
from AllTechniques import *
import subprocess
import glob
import time

def wrl2Off():
    for i in range(1, 10):
        for name in ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise"]:
            foldername = "BU_4DFE/F%.3i/%s"%(i, name)
            subprocess.call(["avconv", "-r", "30", "-i", "%s/%s3d.jpg"%(foldername, '%'), "-r", "30", "-b", "30000k", "%s/out.avi"%foldername])
            for filename in glob.glob("%s/*.wrl"%foldername):
                print(filename)
                subprocess.call(["meshlabserver", "-i", filename, "-o", filename[0:-4] + ".off", "-s", "BU_4DFE/text_to_vert.mlx"])
                #meshlabserver -i /media/OLD_LAPTOP/data/BU_4DFE/F001/Fear/042.wrl  -o /media/OLD_LAPTOP/data/BU_4DFE/bfu_offFiles/1F_042.off -s /media/OLD_LAPTOP/data/BU_4DFE/text_to_vert.mlx


def loadVideoFolder(foldername):
    #Assume numbering starts at zero
    N = len(glob.glob("%s/*.jpg"%foldername))
    f0 = scipy.misc.imread("%s/%.3i.jpg"%(foldername, 0))
    IDims = f0.shape
    dim = len(f0.flatten())
    I = np.zeros((N, dim))
    I[0, :] = np.array(f0.flatten(), dtype=np.float32)/255.0
    for i in range(1, N):
        f = scipy.misc.imread("%s/%.3i.jpg"%(foldername, i))
        I[i, :] = np.array(f.flatten(), dtype=np.float32)/255.0
    return (I, IDims)

def getSphereSamples(res = 2):
    """
    Sample the unit sphere as evenly as possible.  The higher res is,
    the more samples are taken on the sphere (in an exponential
    relationship with res).  By default, samples 66 points
    """
    m = getSphereMesh(1, res)
    return m.VPos.T

def samplePointCloud(mesh, N):
    """
    Sample a point cloud, center it on its centroid, and then
    scale all of the points so that the RMS distance to the origin is 1
    """
    (Ps, Ns) = mesh.randomlySamplePoints(N)
    ##TODO: Center the point cloud on its centroid and normalize
    #by its root mean square distance to the origin.  Note that this
    #does not change the normals at all, only the points, since it's a
    #uniform scale
    Ps = Ps - np.mean(Ps, 1, keepdims=True)
    Ps = Ps*np.sqrt(Ps.shape[1]/np.sum(Ps**2))
    return (Ps, Ns)

def getShapeShellHistogram(Ps, Ns, NShells, RMax, SPoints):
    """
    Create shape histogram with concentric spherical shells and
    sectors within each shell, sorted in decreasing order of number of points
    :param Ps: 3 x N point cloud
    :param Ns: 3 x N array of normals (not needed here but passed along for consistency)
    :param NShells: number of shells)
    :param RMax: Maximum radius
    :param SPoints: A 3 x S array of points sampled evenly on the unit sphere
                    (get these with the function "getSphereSamples")
    """
    NSectors = SPoints.shape[1] #A number of sectors equal to the number of
    #points sampled on the sphere
    #Create a 2D histogram that is NShells x NSectors
    hist = np.zeros((NShells, NSectors))
    PDists = np.sqrt(np.sum(Ps**2, 0))
    Rs = np.linspace(0, RMax, NShells+1)
    for i in range(0, NShells):
        PSub = Ps[:, (PDists>=Rs[i])*(PDists<Rs[i+1])]
        scores = (PSub.T).dot(SPoints)
        idx = np.argmax(scores, 1)
        for k in range(NSectors):
            hist[i, k] = np.sum(idx == k)
    hist = np.sort(hist, 1)
    return hist.flatten() #Flatten the 2D histogram to a 1D array

def loadMeshVideoFolder(foldername, NShells = 50, RMax = 2.0, NSamples = 1000000):
    SPoints = getSphereSamples(2)
    #Assume numbering starts at zero
    N = len(glob.glob("%s/*.off"%foldername))
    m = PolyMesh()
    (VPos, VColors, ITris) = loadOffFileExternal("%s/%.3i.off"%(foldername, 0))
    m.VPos = VPos
    m.VColors = VColors
    m.ITris = ITris
    m.needsDisplayUpdate = False
    (Ps, Ns) = samplePointCloud(m, NSamples)
    hist = getShapeShellHistogram(Ps, Ns, NShells, RMax, SPoints)
    NBins = hist.size
    I = np.zeros((N, NBins))
    I[0, :] = hist

    for i in range(1, N):
        try:
            print("Loading %i of %i"%(i, N))
            tic = time.time()
            (VPos, VColors, ITris) = loadOffFileExternal("%s/%.3i.off"%(foldername, i))
            m.VPos = VPos
            m.VColors = VColors
            m.ITris = ITris
            m.needsDisplayUpdate = False
            print("Elapsed Time Loading: %g"%(time.time() - tic))
            tic = time.time()
            (Ps, Ns) = samplePointCloud(m, NSamples)
            print("Elapsed Time Sampling: %g"%(time.time() - tic))
            tic = time.time()
            I[i, :] = getShapeShellHistogram(Ps, Ns, NShells, RMax, SPoints)
            print("Elapsed Time histogram: %g"%(time.time() - tic))
        except:
            continue
    return I

def precomputeEuclideanEmbeddings():
    for emotion in ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise"]:
        for i in range(1, 10):
            foldername = "BU_4DFE/F%.3i/%s"%(i, emotion)
            filename = "%s/SSM.mat"%foldername
            if os.path.exists(filename):
                print("Already done %s.  Skipping..."%filename)
            else:
                np.random.seed(100)
                #Load in mesh and compute shell histogram
                NShells = 20
                I = loadMeshVideoFolder(foldername, NShells)
                SSM = np.array(getSSM(I), dtype=np.float32)
                #Save the results
                sio.savemat(filename, {"SSM":SSM, "I":I})
            filename = "%s/VideoPCA.mat"%foldername
            if os.path.exists(filename):
                print("Already done %s.  Skipping..."%filename)
            else:
                (I, IDims) = loadVideoFolder(foldername)
                I = getPCAVideo(I)
                sio.savemat(filename, {"I":I, "IDims":IDims})

def runAlignmentExperiments(eng, seed, K = 10, NPerFace = 10, doPlot = False):
    """
    Run experiments randomly warping the video and trying to align that
    to the 3D histograms
    """
    np.random.seed(seed)
    emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise"]
    NFaces = 9
    AllErrors = {}
    for e in range(len(emotions)):
        emotion = emotions[e]
        for i in range(1, NFaces+1):
            print("Doing %s face %i..."%(emotion, i))
            foldername = "BU_4DFE/F%.3i/%s"%(i, emotion)
            I1 = sio.loadmat("%s/SSM.mat"%foldername)['I']
            I2 = sio.loadmat("%s/VideoPCA.mat"%foldername)['I']
            WarpDict = getWarpDictionary(I2.shape[0])
            for expNum in range(NPerFace):
                t2 = getWarpingPath(WarpDict, K, False)
                I2Warped = getInterpolatedEuclideanTimeSeries(I2, t2)
                sio.savemat("BU.mat", {"I1":I1, "I2Warped":I2Warped, "t2":t2})
                plt.clf()
                (errors, Ps) = doAllAlignments(eng, I1, I2Warped, t2, drawPaths = doPlot)
                if doPlot:
                    plt.savefig("%i_%i_%i.svg"%(e, i, expNum))
                types = errors.keys()
                for t in types:
                    if not t in AllErrors:
                        AllErrors[t] = np.zeros((len(emotions), NFaces, NPerFace))
                    AllErrors[t][e][i-1][expNum] = errors[t]
            sio.savemat("BUErrors.mat", AllErrors)

if __name__ == '__main__':
    initParallelAlgorithms()
    eng = initMatlabEngine()
    runAlignmentExperiments(eng, 1, K = 10, NPerFace = 10, doPlot = True)
