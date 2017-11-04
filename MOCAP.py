import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from SlidingWindowVideoTDA.VideoTools import *
from Alignment.AllTechniques import *
from Alignment.AlignmentTools import *
from Alignment.Alignments import *
from Alignment.DTWGPU import *
from Skeleton import *
from Weizmann import *
from PaperFigures import makeColorbar

def getQuaterionsSSM(XQ, projPlane = True):
    N = XQ.shape[2]
    D = np.zeros((N, N))
    for i in range(N):
        q1 = XQ[:, :, i]
        for j in range(i+1, N):
            q2 = XQ[:, :, j]
            cosTheta = np.sum(q1*q2, 0)
            if projPlane:
                cosTheta = np.abs(cosTheta)
            cosTheta[cosTheta > 1] = 1
            cosTheta[cosTheta < -1] = -1
            D[i, j] = np.sum(np.arccos(cosTheta))
    D = D + D.T
    return D


def MOCAPJumpingJacksExample(doPartial = True, doInterpolated = False):
    #Load in MOCAP walking data
    skeleton = Skeleton()
    skeleton.initFromFile("MOCAP/22.asf")
    activity = SkeletonAnimator(skeleton)
    res = activity.initFromFileUsingOctave("MOCAP/22.asf", "MOCAP/22_16.amc")
    #Get quaternions
    XQ = res['XQ']
    DQ = getQuaterionsSSM(XQ)
    DQS = getQuaterionsSSM(XQ, False)
    XQE = np.reshape(XQ, (XQ.shape[0]*XQ.shape[1], XQ.shape[2]))
    DQE = getSSM(XQE.T)

    #Load in Weizmann walking mask
    #(I, IDims) = loadImageIOVideo("MOCAP/jumpingjackscropped.avi")
    #I = I[25::, :]
    (I, IDims) = loadImageIOVideo("MOCAP/jumpingjacks2men.ogg")
    I = I[0:70]
    #I = I[0:int(I.shape[0]/2), :]
    
    if doInterpolated:
        I = getInterpolatedEuclideanTimeSeries(I, np.linspace(0, 1, 300))
    DV = getSSM(I)
    
    print("DQ.shape = {}".format(DQ.shape))
    print("DV.shape = {}".format(DV.shape))

    plt.figure(figsize=(10, 10))
    plt.subplot(221)
    plt.imshow(DQ, cmap = 'afmhot', interpolation = 'nearest')
    plt.title("Quaternions $\mathbb{R}P^3$ Embedding")
    plt.subplot(222)
    plt.imshow(DQS, cmap = 'afmhot', interpolation = 'nearest')
    plt.title("Quaternions $S^3$ Embedding")
    plt.subplot(223)
    plt.imshow(DQE, cmap = 'afmhot', interpolation = 'nearest')
    plt.title("Quaternions Euclidean Embedding")
    plt.subplot(224)
    plt.imshow(DV, cmap = 'afmhot', interpolation = 'nearest')
    plt.title("Jumping Jacks Video")
    plt.savefig("JumpingJacksEmbeddings.svg", bbox_inches = 'tight')

    L = 200
    D1 = DQ
    D2 = DV
    (D1N1, D2N1) = matchSSMDist(D1, D2, L)
    (D2N2, D1N2) = matchSSMDist(D2, D1, L)

    if doPartial:
        matchfn = lambda x: x
        hvPenalty = -0.4
        #Try 1 To 2 Normalization
        CSWM1 = doIBSMWatGPU(D1N1, D2N1, 0.3, True)
        CSWM1 = CSWM1 - np.median(CSWM1)
        CSWM1 = CSWM1/np.max(np.abs(CSWM1))
        res1 = SMWat(CSWM1, matchfn, hvPenalty, backtrace = True)
        #Try 2 To 1 Normalization
        CSWM2 = doIBSMWatGPU(D1N2, D2N2, 0.3, True)
        CSWM2 = CSWM2 - np.median(CSWM2)
        CSWM2 = CSWM2/np.max(np.abs(CSWM2))
        res2 = SMWat(CSWM2, matchfn, hvPenalty, backtrace = True)
        res = res1
        CSWM = CSWM1
        if res2['pathScore'] > res1['pathScore']:
            res = res2
            CSWM = CSWM2
        path = res['path']
    else:
        CSWM1 = doIBDTWGPU(D1N1, D2N1, returnCSM = True)
        CSWM2 = doIBDTWGPU(D1N2, D2N2, returnCSM = True)
        (DAll, CSSM1, backpointers, path1) = DTWCSM(CSWM1)
        (DAll, CSSM2, backpointers, path2) = DTWCSM(CSWM2)
        CSWM = CSWM1
        path = path1
        if CSSM2[-1, -1] < CSSM1[-1, -1]:
            CSWM = CSWM2
            path = path2
    
    if not doPartial:
        #For better visualization for CSWM for IBDTW
        CSWM = np.log(0.001+CSWM)
    
    plt.clf()
    plt.subplot(2, 2, 1)
    plt.imshow(DQ, cmap = 'afmhot', interpolation = 'nearest')
    plt.subplot(2, 2, 2)
    plt.imshow(DV, cmap = 'afmhot', interpolation = 'nearest')
    plt.subplot(2, 2, 3)
    plt.imshow(CSWM, cmap = 'afmhot', interpolation = 'nearest', aspect = 'auto')
    plt.subplot(2, 2, 4)
    plt.imshow(CSWM, cmap = 'afmhot', interpolation = 'nearest', aspect = 'auto')
    plt.scatter(path[:, 1], path[:, 0], 5, 'c', edgecolor = 'none')

    path = projectPath(path, CSWM.shape[0], CSWM.shape[1], 1)

    plt.savefig("CSWM.svg")
    #Plot frames aligned to each other
    (IM, IMDims) = loadImageIOVideo("MOCAP/22_16.avi")
    plt.figure(figsize=(15, 5))
    for i in range(path.shape[0]):
        plt.clf()
        plt.subplot(1, 3, 1)
        F = I[path[i, 1], :]
        F = np.reshape(F, IDims)
        plt.imshow(F)
        plt.title("Frame %i"%path[i, 1])
        plt.axis("off")

        plt.subplot(1, 3, 2)
        F = IM[path[i, 0], :]
        F = np.reshape(F, IMDims)
        plt.imshow(F)
        plt.axis('off')
        plt.title("Frame %i"%path[i, 0])

        plt.subplot(133)
        plt.imshow(CSWM, aspect = 'auto', cmap = 'afmhot', interpolation = 'nearest')
        plt.scatter(path[:, 1], path[:, 0], 5, 'c', edgecolor = 'none')
        plt.scatter(path[i, 1], path[i, 0], 30, 'm')

        plt.savefig("MOCAPAligned%i.png"%i, bbox_inches = 'tight')

if __name__ == '__main__':
    initParallelAlgorithms()
    MOCAPJumpingJacksExample(doPartial = False, doInterpolated = False)
