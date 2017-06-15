import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import sys
sys.path.append('..')
sys.path.append('S3DGLPy')
sys.path.append('SlidingWindowVideoTDA')
from CSMSSMTools import *
from SyntheticCurves import *
from Alignments import *
from DTWGPU import *
from PolyMesh import *
from VideoTools import *
import subprocess
import glob
import time

def wrl2Off():
    for i in range(1, 10):
        for name in ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise"]:
            foldername = "BU_4DFE/F%.3i/%s"%(i, name)
            subprocess.call(["avconv", "-r", "30", "-i", "%s/%s3d.jpg"%(foldername, '%'), "-r", "30", "-b", "30000k", "%s/out.avi"%foldername])
            for filename in glob.glob("%s/*.wrl"%foldername):
                print filename
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

#Purpose: To sample the unit sphere as evenly as possible.  The higher
#res is, the more samples are taken on the sphere (in an exponential
#relationship with res).  By default, samples 66 points
def getSphereSamples(res = 2):
    m = getSphereMesh(1, res)
    return m.VPos.T

def compareHistsEMD1D(AllHists):
    N = AllHists.shape[1]
    K = AllHists.shape[0]
    CS = np.cumsum(AllHists, 0)
    D = np.zeros((N, N))
    for k in range(K):
        c = CS[k, :]
        D += np.abs(c[:, None] - c[None, :])
    return D

#Purpose: To sample a point cloud, center it on its centroid, and
#then scale all of the points so that the RMS distance to the origin is 1
def samplePointCloud(mesh, N):
    (Ps, Ns) = mesh.randomlySamplePoints(N)
    ##TODO: Center the point cloud on its centroid and normalize
    #by its root mean square distance to the origin.  Note that this
    #does not change the normals at all, only the points, since it's a
    #uniform scale
    Ps = Ps - np.mean(Ps, 1, keepdims=True)
    Ps = Ps*np.sqrt(Ps.shape[1]/np.sum(Ps**2))
    return (Ps, Ns)

#Purpose: To create shape histogram with concentric spherical shells and
#sectors within each shell, sorted in decreasing order of number of points
#Inputs: Ps (3 x N point cloud), Ns (3 x N array of normals) (not needed here
#but passed along for consistency), NShells (number of shells),
#RMax (maximum radius), SPoints: A 3 x S array of points sampled evenly on
#the unit sphere (get these with the function "getSphereSamples")
def getShapeShellHistogram(Ps, Ns, NShells, RMax, SPoints):
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
            print "Loading %i of %i"%(i, N)
            tic = time.time()
            (VPos, VColors, ITris) = loadOffFileExternal("%s/%.3i.off"%(foldername, i))
            m.VPos = VPos
            m.VColors = VColors
            m.ITris = ITris
            m.needsDisplayUpdate = False
            print "Elapsed Time Loading: ", time.time() - tic
            tic = time.time()
            (Ps, Ns) = samplePointCloud(m, NSamples)
            print "Elapsed Time Sampling: ", time.time() - tic
            tic = time.time()
            I[i, :] = getShapeShellHistogram(Ps, Ns, NShells, RMax, SPoints)
            print "Elapsed Time histogram: ", time.time() - tic
        except:
            continue
    return I

def precomputeEuclideanEmbeddings():
    for emotion in ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise"]:
        for i in range(1, 10):
            foldername = "BU_4DFE/F%.3i/%s"%(i, emotion)
            filename = "%s/SSM.mat"%foldername
            if os.path.exists(filename):
                print "Already done %s.  Skipping..."%filename
            else:
                np.random.seed(100)
                #Load in mesh and compute D2
                NShells = 20
                I = loadMeshVideoFolder(foldername, NShells)
                SSM = np.array(getCSM(I, I), dtype=np.float32)
                #Save the results
                sio.savemat(filename, {"SSM":SSM, "I":I})
            filename = "%s/VideoPCA.mat"%foldername
            if os.path.exists(filename):
                print "Already done %s.  Skipping..."%filename
            else:
                (I, IDims) = loadVideoFolder(foldername)
                I = getPCAVideo(I)
                sio.savemat(filename, {"I":I, "IDims":IDims})

if __name__ == '__main__2':
    foldername = "BU_4DFE/F%.3i/%s"%(1, "Angry")
    (I, IDims) = loadVideoFolder(foldername)
    Dict = getWarpDictionary(200)
    np.random.seed(100)
    t = getWarpingPath(Dict, 3, doPlot = True)
    plt.savefig("WarpingPath.svg")
    I2 = getInterpolatedEuclideanTimeSeries(I, t)
    saveVideo(I2, IDims, "WarpedVideo.avi")

if __name__ == '__main__':
    filename = "SSMsVideoWarped.mat"
    if not os.path.exists(filename):
        (I1, IDims) = loadVideo("OriginalVideo.avi")
        (I2, IDims) = loadVideo("WarpedVideo.avi")
        SSMX = np.array(getCSM(I1, I1), dtype=np.float32)
        SSMY = np.array(getCSM(I2, I2), dtype=np.float32)
        #SSMXRank = getRankSSM(SSMX)
        #SSMYRank = getRankSSM(SSMY)

        SSMXRank = get2DRankSSM(SSMX)
        SSMYRank = get2DRankSSM(SSMY)

        #SSMXRank = getZNormSSM(SSMX)
        #SSMYRank = getZNormSSM(SSMY)
        sio.savemat(filename, {"SSMX":SSMX, "SSMY":SSMY, "SSMXRank":SSMXRank, "SSMYRank":SSMYRank})
    else:
        X = sio.loadmat(filename)
        [SSMX, SSMY, SSMXRank, SSMYRank] = [X['SSMX'], X['SSMY'], X['SSMXRank'], X['SSMYRank']]

    Dict = getWarpDictionary(200)
    np.random.seed(100)
    t2 = getWarpingPath(Dict, 3, doPlot = True)
    t2 = t2*(SSMY.shape[0]-1)
    t1 = SSMX.shape[0]*np.linspace(0, 1, len(t2))



    tic = time.time()
    D = doIBDTW(SSMX, SSMY)
    DRank = doIBDTW(SSMXRank, SSMYRank)
    (DAll, CSM, backpointers, involved) = DTWCSM(D)
    (DAllR, CSMR, backpointersR, involvedR) = DTWCSM(DRank)
    print "Elapsed Time CPU: ", time.time() - tic

    plt.figure(figsize=(10, 10))
    plt.subplot(321)
    plt.imshow(SSMX, cmap = 'afmhot', interpolation = 'nearest')
    plt.axis('off')
    plt.title('SSM 1')
    plt.subplot(322)
    plt.imshow(SSMY, cmap = 'afmhot', interpolation = 'nearest')
    plt.axis('off')
    plt.title('SSM 2')

    plt.subplot(323)
    plt.imshow(SSMXRank, cmap = 'afmhot', interpolation = 'nearest')
    plt.axis('off')
    plt.title('SSM 1 Rank')
    plt.subplot(324)
    plt.imshow(SSMYRank, cmap = 'afmhot', interpolation = 'nearest')
    plt.axis('off')
    plt.title('SSM 2 Rank')

    plt.subplot(325)
    plt.imshow(D, cmap = 'afmhot', interpolation = 'nearest')
    plt.hold(True)
    [J, I] = np.meshgrid(np.arange(involved.shape[1]), np.arange(involved.shape[0]))
    J = J[involved == 1]
    I = I[involved == 1]
    plt.scatter(J, I, 5, 'c', edgecolor = 'none')
    plt.scatter(t1, t2, 5, 'r', edgecolor = 'none')
    plt.xlim([0, CSM.shape[1]])
    plt.ylim([CSM.shape[0], 0])
    plt.axis('off')
    #plt.title("Cost = %g"%resGPU)
    plt.title("Cross-Similarity Warp Matrix Orig")

    plt.subplot(326)
    plt.imshow(DRank, cmap = 'afmhot', interpolation = 'nearest')
    plt.hold(True)
    [J, I] = np.meshgrid(np.arange(involvedR.shape[1]), np.arange(involvedR.shape[0]))
    J = J[involvedR == 1]
    I = I[involvedR == 1]
    plt.scatter(J, I, 5, 'c', edgecolor = 'none')
    plt.scatter(t1, t2, 5, 'r', edgecolor = 'none')
    plt.xlim([0, CSMR.shape[1]])
    plt.ylim([CSMR.shape[0], 0])
    plt.axis('off')
    #plt.title("Cost = %g"%resGPU)
    plt.title("Cross-Similarity Warp Matrix Ranks")

    plt.show()


if __name__ == '__main__2':

    SSMX = np.array(getCSM(X, X), dtype=np.float32)
    SSMY = np.array(getCSM(Y, Y), dtype=np.float32)
    SSMX = get2DRankSSM(SSMX)
    SSMY = get2DRankSSM(SSMY)
    #SSMX = getZNormSSM(SSMX)
    #SSMY = getZNormSSM(SSMY)
    tic = time.time()
    D = doIBDTW(SSMX, SSMY)
    print "Elapsed Time CPU: ", time.time() - tic
    gSSMX = gpuarray.to_gpu(np.array(SSMX, dtype = np.float32))
    gSSMY = gpuarray.to_gpu(np.array(SSMY, dtype = np.float32))

    sio.savemat("D.mat", {"D":D, "D2":D2})

    plt.imshow(D, cmap = 'afmhot')
    plt.show()

    (DAll, CSM, backpointers, involved) = DTWCSM(D)
    resCPU = DAll[-1, -1]
    print "CPU Result: ", resCPU

    c = plt.get_cmap('Spectral')
    C1 = c(np.array(np.round(255*np.arange(M)/float(M)), dtype=np.int32))
    C1 = C1[:, 0:3]
    idx = np.argsort(-involved, 0)[0, :]
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
    [J, I] = np.meshgrid(np.arange(involved.shape[1]), np.arange(involved.shape[0]))
    J = J[involved == 1]
    I = I[involved == 1]
    plt.scatter(J, I, 5, 'c', edgecolor = 'none')
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
