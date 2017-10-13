import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from SlidingWindowVideoTDA.VideoTools import *
from Alignment.AllTechniques import *
from Alignment.AlignmentTools import *
from Alignment.Alignments import *
from Alignment.DTWGPU import *
from Alignment.ctw.CTWLib import *
from PaperFigures import makeColorbar

WEIPATH = "Alignment/ctw/data/wei/"
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

def getEDT(I, IDims, doBorderDistance = False, doPlot = False, doFlip = False):
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
        if doFlip:
            F = np.fliplr(F)
        F2 = scipy.ndimage.morphology.distance_transform_edt(F)
        if doBorderDistance:
            F2 += scipy.ndimage.morphology.distance_transform_edt(1-F)
        if doPlot:
            plt.imshow(F2, cmap = 'afmhot')
            plt.show()
        X[i, :] = F2.flatten()
    return X

def alignWalkingVideos(eng, L = 200):
    IsMask = []
    Is = []
    Xs = []
    SSMs = []
    paths = [[]]
    counter = 1
    f1 = plt.figure(figsize=(5*len(WEIVIDEOS), 10))
    f2 = plt.figure(figsize=(14, 12))
    for video in WEIVIDEOS:
        (I, IDims) = getWeiAlignedMask(video)
        if counter > 1:
            I = getEDT(I, IDims)
        I = 1.0*I
        IsMask.append((I, IDims))
        D = getSSM(I)
        SSMs.append(D)
        plt.figure(f1.number)
        plt.subplot(2, len(WEIVIDEOS), counter)
        plt.imshow(SSMs[-1], cmap = 'afmhot', interpolation = 'nearest')
        plt.title(video)

        if counter > 1:
            v1str = WEIVIDEOS[0][0:-5]
            v1str = v1str[0].capitalize() + v1str[1::]
            v2str = WEIVIDEOS[counter-1][0:-5]
            v2str = v2str[0].capitalize() + v2str[1::]
            [D1, D2] = [SSMs[0], SSMs[-1]]
            (D1N1, D2N1) = matchSSMDist(D1, D2, L)
            (D2N2, D1N2) = matchSSMDist(D2, D1, L)
            CSWM = doIBDTW(D1, D2)
            CSWM1 = doIBDTW(D1N1, D2N1)
            CSWM2 = doIBDTW(D1N2, D2N2)
            (DAll, CSSM, backpointers, path) = DTWCSM(CSWM)
            (DAll, CSSM1, backpointers, path1) = DTWCSM(CSWM1)
            (DAll, CSSM2, backpointers, path2) = DTWCSM(CSWM2)
            pathIBDTW = path1 #Best path
            if D2[-1, -1] < D1[-1, -1]:
                pathIBDTW = path2
            plt.subplot(2, len(WEIVIDEOS), len(WEIVIDEOS) + counter)
            plt.imshow(CSWM, cmap = 'afmhot', interpolation = 'nearest')
            plt.plot(pathIBDTW[:, 1], pathIBDTW[:, 0], '.')
            #paths.append(projectPath(path, D.shape[0], D.shape[1], 1))
            res = getCTWAlignments(eng, IsMask[0][0], IsMask[-1][0])
            for ptype in res:
                path = res[ptype]
                plt.plot(path[:, 1], path[:, 0], '.')
            plt.legend(['IBDTW'] + res.keys())
            paths.append(projectPath(pathIBDTW, CSWM.shape[0], CSWM.shape[1], 1))

            #Plot the different normalized SSMs
            plt.figure(f2.number)
            plt.clf()
            plt.subplot(331)
            plt.imshow(D1, cmap = 'afmhot', interpolation = 'nearest')
            plt.title("SSM %s Mask"%v1str)
            plt.ylabel("Frame Number")
            makeColorbar(3, 3, 1)
            plt.subplot(334)
            plt.imshow(D2, cmap = 'afmhot', interpolation = 'nearest')
            plt.title("SSM %s EDT"%v2str)
            plt.ylabel("Frame Number")
            makeColorbar(3, 3, 4)
            plt.subplot(337)
            plt.imshow(CSWM, cmap = 'afmhot', interpolation = 'nearest')
            plt.scatter(path[:, 1], path[:, 0], 10, 'r', edgecolor = 'none')
            plt.xlim([0, CSWM.shape[1]])
            plt.ylim([CSWM.shape[0], 0])
            plt.xlabel("%s Frame Number"%v2str)
            plt.ylabel("%s Frame Number"%v1str)
            plt.title("CSWM, Score = %.3g"%CSSM[-1, -1])

            plt.subplot(332)
            plt.imshow(D1N1, cmap = 'afmhot', interpolation = 'nearest')
            plt.title("SSM %s Mask Remapped"%v1str)
            makeColorbar(3, 3, 2)
            plt.subplot(335)
            plt.imshow(D2N1, cmap = 'afmhot', interpolation = 'nearest')
            plt.title("SSM %s EDT Discretized"%v2str)
            makeColorbar(3, 3, 5)
            plt.subplot(338)
            plt.imshow(np.log(CSWM1), cmap = 'afmhot', interpolation = 'nearest')
            plt.scatter(path1[:, 1], path1[:, 0], 10, 'c', edgecolor = 'none')
            plt.xlim([0, CSWM1.shape[1]])
            plt.ylim([CSWM1.shape[0], 0])
            plt.title("CSWM, Score = %.3g"%CSSM1[-1, -1])

            plt.subplot(333)
            plt.imshow(D1N2, cmap = 'afmhot', interpolation = 'nearest')
            plt.title("SSM %s Mask Discretized"%v1str)
            makeColorbar(3, 3, 3)
            plt.subplot(336)
            plt.imshow(D2N2, cmap = 'afmhot', interpolation = 'nearest')
            plt.title("SSM %s EDT Remapped"%v2str)
            makeColorbar(3, 3, 6)
            plt.subplot(339)
            plt.imshow(np.log(CSWM2), cmap = 'afmhot', interpolation = 'nearest')
            plt.scatter(path2[:, 1], path2[:, 0], 10, 'm', edgecolor = 'none')
            plt.xlim([0, CSWM2.shape[1]])
            plt.ylim([CSWM2.shape[0], 0])
            plt.title("CSWM, Score = %.3g"%CSSM2[-1, -1])

            plt.savefig("WeizmannRankNorm%i.svg"%counter, bbox_inches = 'tight')
        counter += 1
        (I, IDims) = loadImageIOVideo("%s/walk/%s.avi"%(WEIPATH, video))
        idx = WEICROP[video]
        I = I[idx[0]:idx[1], :]
        Is.append((I, IDims))
    plt.figure(f1.number)
    plt.savefig("WeiAlignmentPaths.svg", bbox_inches = 'tight')

    """
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
    """

def plotAlignedWalkingVideos(L = 200):
    IsMask = []
    MaskDims = []
    Is = []
    SSMs = []
    paths = [[]]
    counter = 1
    frames = range(20, 40, 3)
    videos = ["daria_walk", "ira_walk"]
    for video in videos:
        (I, IDims) = getWeiAlignedMask(video)
        if counter > 1:
            I = getEDT(I, IDims, doFlip = True)
        I = 1.0*I
        IsMask.append(I)
        MaskDims.append(IDims)
        D = getSSM(I)
        SSMs.append(D)

        if counter > 1:
            [D1, D2] = [SSMs[0], SSMs[-1]]
            (D1N1, D2N1) = matchSSMDist(D1, D2, L)
            (D2N2, D1N2) = matchSSMDist(D2, D1, L)
            D1 = doIBDTW(D1N1, D2N1)
            D2 = doIBDTW(D1N2, D2N2)
            (DAll, CSSM1, backpointers, path1) = DTWCSM(D1)
            (DAll, CSSM2, backpointers, path2) = DTWCSM(D2)
            D = D1
            pathIBDTW = path1
            if CSSM2[-1, -1] < CSSM1[-1, -1]:
                D = D2
                pathIBDTW = path2
            paths.append(projectPath(pathIBDTW, D.shape[0], D.shape[1], 1))
        counter += 1
        (I, IDims) = loadImageIOVideo("%s/walk/%s.avi"%(WEIPATH, video))
        idx = WEICROP[video]
        I = I[idx[0]:idx[1], :]
        Is.append((I, IDims))

    #Plot frames aligned to each other
    plt.figure(figsize=(1.5*len(frames), 4))
    for i in range(len(frames)):
        plt.subplot(2, len(frames), i+1)
        I = IsMask[0]
        F = I[frames[i], :]
        F = np.reshape(F, MaskDims[0])
        plt.imshow(F, cmap = 'gray', interpolation = 'nearest')
        plt.title("Frame %i"%frames[i])
        plt.axis("off")

        vidx = 1
        plt.subplot(2, len(frames), len(frames)+i+1)
        I = IsMask[vidx]
        idx = paths[vidx][frames[i], 1]
        if idx >= I.shape[0]:
            idx = I.shape[0]-1
        F = np.reshape(I[idx, :], MaskDims[1])
        plt.imshow(F, cmap = 'afmhot', interpolation = 'nearest')
        plt.axis('off')
        plt.title("Frame %i"%idx)
    plt.savefig("WeiAlignedFeatures.svg", bbox_inches = 'tight')

def partialAlignWalkingVideos(crossModal = True, L = 200):
    IsMask = []
    Is = []
    SSMs = []
    paths = [[]]
    CSWMs = [[]]
    counter = 0
    plt.figure(figsize=(15, 5))
    for video in WEIVIDEOS:
        (I, IDims) = getWeiAlignedMask(video, doCrop = False)
        I = 1.0*I
        IsMask.append(I)
        if counter == 0 or not crossModal:
            D = getSSM(I)
            SSMs.append(D)
        else:
            I = getEDT(I, IDims)
            D = getSSM(I)
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

        [D1, D2] = [SSMs[0], SSMs[-1]]
        (D1N1, D2N1) = matchSSMDist(D1, D2, L)
        (D2N2, D1N2) = matchSSMDist(D2, D1, L)
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

        CSWMs.append(CSWM)
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

def runAlignmentExperiments(eng, K = 10, NPerVideo = 50, doPlot = False):
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
        I2 = 1.0*getEDT(I1, IDims)
        I1 = 1.0*I1
        WarpDict = getWarpDictionary(I2.shape[0])
        for expNum in range(NPerVideo):
            print("Doing video %i trial %i of %i"%(vidx, expNum+1, NPerVideo))
            t2 = getWarpingPath(WarpDict, K, False)
            I2Warped = getInterpolatedEuclideanTimeSeries(I2, t2)
            sio.savemat("Weizmann.mat", {"I1":I1, "I2Warped":I2Warped, "t2":t2})
            if doPlot:
                plt.clf()
                sio.savemat("%i_%i.mat"%(vidx, expNum), {"X1":I1, "X2":I2Warped})
            (errors, Ps) = doAllAlignments(eng, I1, I2Warped, t2, drawPaths = doPlot)
            if doPlot:
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
    #plotAlignedWalkingVideos()
    #alignWalkingVideos(eng)
    runAlignmentExperiments(eng, NPerVideo = 100)
    #partialAlignWalkingVideos(crossModal = False)

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
