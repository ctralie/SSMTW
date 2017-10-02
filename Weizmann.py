import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from SlidingWindowVideoTDA.VideoTools import *
from AllTechniques import *
from Alignment.AlignmentTools import *
from Alignment.Alignments import *
from Alignment.DTWGPU import *
from CTWLib import *

WEIPATH = "ctw/data/wei/"
WEIVIDEOS = ["daria_walk", "denis_walk", "eli_walk", "ira_walk"]
WEICROP = {'daria_walk':[10, 61], 'denis_walk':[8, 61], 'eli_walk':[3, 61], 'ido_walk':[0, 44], 'ira_walk':[0, 61]}

def getWeiNamesFromStruct(dtype):
    """
    Purpose: For some reason the Weizmann struct got unrolled so the dtype
    holds the action names.  Given the dtype, split it up into a list of
    strings
    """
    s = "%s"%dtype
    s = s.replace(", ", "")
    names = [(n.replace("('", "")).replace("'", "") for n in s.split("'O')")[0:-1]]
    names[0] = names[0][1::]
    return names

def getWeiAlignedMask(action, doCrop = True):
    X = sio.loadmat(WEIPATH + "/mask.mat")
    X = X['aligned_masks'][0][0]

    names = getWeiNamesFromStruct(X.dtype)
    idx = 0
    for idx in range(len(names)):
        if names[idx] == action:
            break
    print("name = %s, idx = %i"%(action, idx))
    I = X[idx]
    I = np.rollaxis(I, 2, 0)
    IDims = I.shape[1::]
    I = np.reshape(I, (I.shape[0], IDims[0]*IDims[1]))
    if doCrop:
        idx = WEICROP[action]
        I = I[idx[0]:idx[1], :]
    return (I, IDims)

def getEDT(I, IDims, doBorderDistance = False, doPlot = False):
    """
    Return the Euclidean distance transform of each frame
    :param I: An NFrames x NPixels binary mask video
    :param IDims: Dimensions of each frame
    :returns X: An NFrames x NPixels real valued Euclidean distance
        transform sequence
    """
    import scipy.ndimage.morphology
    X = 0*I
    for i in range(I.shape[0]):
        F = np.reshape(I[i, :], IDims)
        F2 = scipy.ndimage.morphology.distance_transform_edt(F)
        if doBorderDistance:
            F2 += scipy.ndimage.morphology.distance_transform_edt(1-F)
        if doPlot:
            plt.imshow(F2, cmap = 'afmhot')
            plt.show()
        X[i, :] = F2.flatten()
    return X

def alignWalkingVideos(eng):
    IsMask = []
    Is = []
    SSMs = []
    paths = [[]]
    counter = 1
    plt.figure(figsize=(5*len(WEIVIDEOS), 10))
    for video in WEIVIDEOS:
        (I, IDims) = getWeiAlignedMask(video)
        IsMask.append(1.0*I)
        if counter > 1:
            I = getEDT(I, IDims)
        D = getSSM(I)
        D = get2DRankSSM(D)
        SSMs.append(D)
        plt.subplot(2, len(WEIVIDEOS), counter)
        plt.imshow(SSMs[-1], cmap = 'afmhot', interpolation = 'nearest')
        plt.title(video)

        if counter > 1:
            D = doIBDTW(SSMs[0], SSMs[-1])
            (DAll, CSM, backpointers, pathIBDTW) = DTWCSM(D)
            plt.subplot(2, len(WEIVIDEOS), len(WEIVIDEOS) + counter)
            plt.imshow(D, cmap = 'afmhot', interpolation = 'nearest')
            plt.plot(pathIBDTW[:, 1], pathIBDTW[:, 0], '.')
            #paths.append(projectPath(path, D.shape[0], D.shape[1], 1))
            res = getCTWAlignments(eng, IsMask[0], IsMask[-1])
            for ptype in res:
                path = res[ptype]
                plt.plot(path[:, 1], path[:, 0], '.')
            plt.legend(['IBDTW'] + res.keys())
            paths.append(projectPath(pathIBDTW, D.shape[0], D.shape[1], 1))
        counter += 1
        (I, IDims) = loadImageIOVideo("%s/walk/%s.avi"%(WEIPATH, video))
        idx = WEICROP[video]
        I = I[idx[0]:idx[1], :]
        Is.append((I, IDims))
    plt.savefig("WeiAlignmentPaths.svg", bbox_inches = 'tight')

    #Plot frames aligned to each other
    plt.figure(figsize=(5*len(WEIVIDEOS), 5))
    for i in range(Is[0][0].shape[0]):
        plt.clf()
        plt.subplot(1, len(WEIVIDEOS), 1)
        I = Is[0][0]
        F = I[i, :]
        F = np.reshape(F, Is[0][1])
        plt.imshow(F)
        plt.title("Frame %i"%i)
        plt.axis("off")
        for vidx in range(1, len(Is)):
            plt.subplot(1, len(WEIVIDEOS), vidx+1)
            I = Is[vidx][0]
            idx = paths[vidx][i, 1]
            if idx >= I.shape[0]:
                idx = I.shape[0]-1
            F = np.reshape(I[idx, :], Is[vidx][1])
            plt.imshow(F)
            plt.axis('off')
            plt.title("Frame %i"%idx)
        plt.savefig("%i.png"%i, bbox_inches = 'tight')

def partialAlignWalkingVideos(crossModal = True):
    IsMask = []
    Is = []
    SSMs = []
    paths = [[]]
    CSWMs = [[]]
    counter = 0
    plt.figure(figsize=(15, 5))
    for video in WEIVIDEOS:
        (I, IDims) = getWeiAlignedMask(video, doCrop = False)
        IsMask.append(1.0*I)
        if counter == 0 or not crossModal:
            D = getSSM(I)
            D = get2DRankSSM(D)
            SSMs.append(D)
        else:
            I = getEDT(I, IDims)
            D = getSSM(I)
            D = get2DRankSSM(D)
            SSMs.append(D)
        (I, IDims) = loadImageIOVideo("%s/walk/%s.avi"%(WEIPATH, video))
        Is.append((I, IDims))
        if counter == 0:
            counter += 1
            continue
        plt.clf()
        plt.subplot(1, 3, 1)
        plt.imshow(SSMs[0], cmap = 'afmhot', interpolation = 'nearest')
        plt.title(WEIVIDEOS[0])

        plt.subplot(1, 3, 2)
        plt.imshow(SSMs[-1], cmap = 'afmhot', interpolation = 'nearest')
        plt.title(video)

        CSWM = doIBSMWatGPU(SSMs[0], SSMs[-1], 0.3, True)
        CSWM = CSWM - np.median(CSWM)
        CSWM = CSWM/np.max(np.abs(CSWM))
        CSWMs.append(CSWM)
        matchfn = lambda x: x
        hvPenalty = -0.4
        res = SMWat(CSWM, matchfn, hvPenalty, backtrace = True)
        path = res['path']

        plt.subplot(1, 3, 3)
        plt.imshow(CSWM, cmap = 'afmhot', interpolation = 'nearest')
        plt.scatter(path[:, 1], path[:, 0], 5, 'c', edgecolor = 'none')
        paths.append(projectPath(path, CSWM.shape[0], CSWM.shape[1], 1))

        plt.savefig("WeiPartialAlignmentPaths%i.svg"%counter, bbox_inches = 'tight')
        counter += 1

    #Plot frames aligned to each other
    plt.figure(figsize=(15, 5))
    for vidx in range(1, len(Is)):
        path = paths[vidx]
        CSWM = CSWMs[vidx]
        for i in range(path.shape[0]):
            plt.clf()
            plt.subplot(1, 3, 1)
            I = Is[0][0]
            F = I[path[i, 0], :]
            F = np.reshape(F, Is[0][1])
            plt.imshow(F)
            plt.title("Frame %i"%path[i, 0])
            plt.axis("off")

            plt.subplot(1, 3, 2)
            I = Is[vidx][0]
            F = I[path[i, 1], :]
            F = np.reshape(F, Is[vidx][1])
            plt.imshow(F)
            plt.axis('off')
            plt.title("Frame %i"%path[i, 1])

            plt.subplot(133)
            plt.imshow(CSWM, cmap = 'afmhot', interpolation = 'nearest')
            plt.scatter(path[:, 1], path[:, 0], 5, 'c', edgecolor = 'none')
            plt.scatter(path[i, 1], path[i, 0], 30, 'k')

            plt.savefig("%i_%i.png"%(vidx, i), bbox_inches = 'tight')

def runAlignmentExperiments(eng, K = 10, NPerVideo = 50):
    """
    Run experiments randomly warping the video and trying to align that
    to the 3D histograms
    """
    from Alignment.DTWGPU import initParallelAlgorithms
    initParallelAlgorithms()
    np.random.seed(NPerVideo)
    AllErrors = {}
    count = 1
    NVideos = len(WEIVIDEOS)
    for vidx in range(NVideos):
        print("Doing Video %i of %i..."%(vidx+1, NVideos))
        video = WEIVIDEOS[vidx]
        (I1, IDims) = getWeiAlignedMask(video)
        I2 = getEDT(I1, IDims)
        I1 = 1.0*I1
        WarpDict = getWarpDictionary(I2.shape[0])
        for expNum in range(NPerVideo):
            print("Doing video %i trial %i of %i"%(vidx, expNum+1, NPerVideo))
            t2 = getWarpingPath(WarpDict, K, False)
            I2Warped = getInterpolatedEuclideanTimeSeries(I2, t2)
            sio.savemat("Weizmann.mat", {"I1":I1, "I2Warped":I2Warped, "t2":t2})
            plt.clf()
            (errors, Ps) = doAllAlignments(eng, I1, I2Warped, t2, drawPaths = True)
            plt.savefig("%i_%i.svg"%(vidx, expNum))
            types = errors.keys()
            for t in types:
                if not t in AllErrors:
                    AllErrors[t] = np.zeros((NVideos, NPerVideo))
                AllErrors[t][vidx][expNum] = errors[t]
        sio.savemat("WeizmannErrors%i.mat"%NPerVideo, AllErrors)

if __name__ == '__main__':
    eng = initMatlabEngine()
    initParallelAlgorithms()
    #alignWalkingVideos(eng)
    runAlignmentExperiments(eng, NPerVideo = 50)
    #partialAlignWalkingVideos(crossModal = True)

if __name__ == '__main__2':
    X = sio.loadmat(WEIPATH + "/mask.mat")
    X = X['aligned_masks'][0][0]
    names = getWeiNamesFromStruct(X.dtype)
    #print(names)

    for video in WEIVIDEOS:
        (I1, IDims) = getWeiAlignedMask(video)
        I2 = getEDT(I1, IDims, doPlot = True)
        D1 = getSSM(I1)
        D2 = getSSM(I2)
        plt.subplot(121)
        plt.imshow(D1, cmap = 'afmhot', interpolation = 'none')
        plt.title("SSM Mask")
        plt.subplot(122)
        plt.imshow(D2, cmap = 'afmhot', interpolation = 'none')
        plt.title("SSM EDT")
        plt.show()
        #saveVideo(I, IDims, "%s.avi"%video, 20)
