import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.misc
import sys
from SlidingWindowVideoTDA.VideoTools import *
from Alignment.AlignmentTools import *
from SimilarityFusion import *
from RQA import *
import scipy.ndimage.morphology
import time
import os
from multiprocessing import Pool as PPool

def imresize(D, dims, kind='cubic', use_scipy=False):
    """
    Resize a floating point image
    Parameters
    ----------
    D : ndarray(M1, N1)
        Original image
    dims : tuple(M2, N2)
        The dimensions to which to resize
    kind : string
        The kind of interpolation to use
    use_scipy : boolean
        Fall back to scipy.misc.imresize.  This is a bad idea
        because it casts everything to uint8, but it's what I
        was doing accidentally for a while
    Returns
    -------
    D2 : ndarray(M2, N2)
        A resized array
    """
    if use_scipy:
        return scipy.misc.imresize(D, dims)
    else:
        M, N = dims
        x1 = np.array(0.5 + np.arange(D.shape[1]), dtype=np.float32)/D.shape[1]
        y1 = np.array(0.5 + np.arange(D.shape[0]), dtype=np.float32)/D.shape[0]
        x2 = np.array(0.5 + np.arange(N), dtype=np.float32)/N
        y2 = np.array(0.5 + np.arange(M), dtype=np.float32)/M
        f = scipy.interpolate.interp2d(x1, y1, D, kind=kind)
        return f(x2, y2)

def getZNorm(X):
    Y = X - np.mean(X, 0)[None, :]
    Norm = np.sqrt(np.sum(Y**2, 1))
    Norm[Norm == 0] = 1
    Y = Y/Norm[:, None]
    return Y

def getAudioLibrosa(filename):
    """
    Use librosa to load audio
    :param filename: Path to audio file
    :return (XAudio, Fs): Audio in samples, sample rate
    """
    import librosa
    XAudio, Fs = librosa.load(filename)
    XAudio = librosa.core.to_mono(XAudio)
    return (XAudio, Fs)

def getMFCCsLibrosa(XAudio, Fs, winSize, hopSize = 512, NBands = 40, fmax = 8000, NMFCC = 20, doZNorm = False, lifterexp = 0):
    """
    Get MFCC features using librosa functions
    :param XAudio: A flat array of audio samples
    :param Fs: Sample rate
    :param winSize: Window size to use for STFT
    :param hopSize: Hop size to use for STFT (default 512)
    :param NBands: Number of mel bands to use
    :param fmax: Maximum frequency
    :param NMFCC: Number of MFCC coefficients to return
    :param lifterexp: Lifter exponential
    :return X: An (NMFCC x NWindows) array of MFCC samples
    """
    import librosa
    S = librosa.core.stft(XAudio, winSize, hopSize)
    M = librosa.filters.mel(Fs, winSize, n_mels = NBands, fmax = fmax)

    X = M.dot(np.abs(S))
    X = librosa.core.amplitude_to_db(X)
    X = np.dot(librosa.filters.dct(NMFCC, X.shape[0]), X) #Make MFCC
    #Do liftering
    coeffs = np.arange(NMFCC)**lifterexp
    coeffs[0] = 1
    X = coeffs[:, None]*X
    X = np.array(X, dtype = np.float32).T
    
    if doZNorm:
        X = getZNorm(X)
    return X

def resizeVideo(I, IDims, NewDims, do_plot = False):
    INew = []
    for i in range(I.shape[0]):
        F = np.reshape(I[i, :], IDims)
        F = rgb2gray(F)
        F = 1.0*scipy.misc.imresize(F, NewDims)
        if do_plot:
            plt.imshow(F)
            plt.show()
        INew.append(F.flatten())
    INew = np.array(INew)
    return INew

def getAVSSMsFused(s, i, ssmdim, K, do_plot = False):
    """
    Parameters
    ----------
    s : int
        Subject number
    i : int
        Sequence number (between 1-30)
    ssmdim : int
        Resized SSM dimension to rescale audio and video to same time domain
    K : int
        Number of nearest neighbors for similarity network fusion
    """
    winSize = 4096
    hopSize = 256
    filename = "OuluVS2/cropped_mouth_mp4_digit/%i/1/s%i_v1_u%i.mp4"%(s, s, i)
    (I1, IDims) = loadImageIOVideo(filename)
    I1 = resizeVideo(I1, IDims, (25, 25))
    filename = "OuluVS2/cropped_audio_dat/s%i_u%i.wav"%(s, i)
    (XAudio, Fs) = getAudioLibrosa(filename)
    XAudio = np.concatenate((XAudio, np.zeros(winSize-hopSize)))
    I2 = getMFCCsLibrosa(XAudio, Fs, winSize, hopSize)

    #[I1, _] = getTimeDerivative(I1, 5)
    #[I2, _] = getTimeDerivative(I2, 5)
    I1 = getZNorm(I1)
    I2 = getZNorm(I2)
    
    D1 = getSSM(I1)
    D2 = getSSM(I2)
    D1 = imresize(D1, (ssmdim, ssmdim))
    D2 = imresize(D2, (ssmdim, ssmdim))

    D3 = doSimilarityFusion([D1, D2], K=K)

    if do_plot:
        pD3 = np.array(D3)
        np.fill_diagonal(pD3, 0)
        plt.subplot(331)
        plt.imshow(D1, cmap = 'afmhot', interpolation = 'none')
        plt.title("Video")
        plt.subplot(332)
        plt.imshow(D2, cmap = 'afmhot', interpolation = 'none')
        plt.title("Audio")
        plt.subplot(333)
        plt.imshow(pD3, cmap = 'afmhot', interpolation = 'none')
        plt.title("Fused")

        Kappa = 0.1
        pD1 = CSMToBinaryMutual(D1, Kappa)
        pD2 = CSMToBinaryMutual(D2, Kappa)
        pD3 = CSMToBinaryMutual(-pD3, Kappa)
        plt.subplot(334)
        plt.imshow(pD1, cmap = 'afmhot', interpolation = 'none')
        plt.subplot(335)
        plt.imshow(pD2, cmap = 'afmhot', interpolation = 'none')
        plt.subplot(336)
        plt.imshow(pD3, cmap = 'afmhot', interpolation = 'none')

        pD1 = scipy.ndimage.morphology.distance_transform_edt(1-pD1)
        pD2 = scipy.ndimage.morphology.distance_transform_edt(1-pD2)
        pD3 = scipy.ndimage.morphology.distance_transform_edt(1-pD3)
        plt.subplot(337)
        plt.imshow(pD1, cmap = 'afmhot', interpolation = 'none')
        plt.subplot(338)
        plt.imshow(pD2, cmap = 'afmhot', interpolation = 'none')
        plt.subplot(339)
        plt.imshow(pD3, cmap = 'afmhot', interpolation = 'none')
    
    return np.concatenate((D1[:, :, None], D2[:, :, None], D3[:, :, None]), 2)

def makedret(I1, I2, ssmdim):
    D1 = getSSM(getZNorm(I1)) / 2.0
    D2 = getSSM(getZNorm(I2)) / 2.0

    D1Resized = imresize(D1, (ssmdim, ssmdim))
    D2Resized = imresize(D2, (ssmdim, ssmdim))

    DRet = np.zeros((ssmdim, ssmdim*2))
    DRet[:, 0:ssmdim] = D1Resized
    DRet[:, ssmdim::] = D2Resized
    DRet[DRet < 0] = 0
    DRet[DRet > 1] = 1

    DRet = DRet[:, :, None]
    DRet = np.concatenate((DRet, DRet, DRet), 2)
    return np.array(np.round(255*DRet), dtype = np.uint8)

def getAVSSMsWarped(s, i, ssmdim, K, NWarps):
    """
    Parameters
    ----------
    s : int
        Subject number
    i : int
        Sequence number (between 1-30)
    ssmdim : int
        Resized SSM dimension to rescale audio and video to same time domain
    K : int
        Number of basis elements in the warping path dictionary
    NWarps : int
        Number of warps to sample (including the unwarped image)
    
    Returns
    -------
    List of warped images, including the original
    """
    winSize = 4096
    hopSize = 256
    filename = "OuluVS2/cropped_mouth_mp4_digit/%i/1/s%i_v1_u%i.mp4"%(s, s, i)
    (I1, IDims) = loadImageIOVideo(filename)
    I1 = resizeVideo(I1, IDims, (25, 50))
    filename = "OuluVS2/cropped_audio_dat/s%i_u%i.wav"%(s, i)
    (XAudio, Fs) = getAudioLibrosa(filename)
    XAudio = np.concatenate((XAudio, np.zeros(winSize-hopSize)))
    I2 = getMFCCsLibrosa(XAudio, Fs, winSize, hopSize)
    # Interpolate video so it has the same time resolution as audio
    I1 = getInterpolatedEuclideanTimeSeries(I1, np.linspace(0, 1, I2.shape[0]))
    Images = [makedret(I1, I2, ssmdim)]

    WarpDict = getWarpDictionary(I1.shape[0])
    for w in range(NWarps-1):
        t = getWarpingPath(WarpDict, K, False)
        I1Warped = getInterpolatedEuclideanTimeSeries(I1, t)
        I2Warped = getInterpolatedEuclideanTimeSeries(I2, t)
        Images.append(makedret(I1Warped, I2Warped, ssmdim))
    return Images
    

def getSSMsHelper(args):
    (subj, seq, ssmdim, K) = args
    print(args)
    return getAVSSMsFused(subj, seq, ssmdim, K)

def writeWekaHeader(fout, NSeq = 30):
    rqa = getRQAArr(getRQAStats(np.random.randn(10, 10) > 0, 5, 5))[1]
    fout.write("@RELATION RQAs\n")
    for r in rqa:
        fout.write("@ATTRIBUTE %s real\n"%r)
    labels = ["Seq%i"%i for i in range(1, NSeq+1)]
    fout.write("@ATTRIBUTE sequence {")
    labels = [l for l in labels]
    for i in range(len(labels)):
        fout.write(labels[i])
        if i < len(labels)-1:
            fout.write(",")
    fout.write("}\n")
    fout.write("@DATA\n")

def doComparisonExperiments(NThreads = 8):
    ssmdim = 400
    K = int(ssmdim*0.1)
    [I, J] = np.meshgrid(np.arange(ssmdim), np.arange(ssmdim))
    NSeq = 30
    NSubj = 52

    parpool = None
    if NThreads > -1:
        parpool = PPool(NThreads)

    if not os.path.exists("AllSSMs.mat"):
        AllSSMs = [ [], [], [] ]
        for seq in range(1, NSeq+1):
            print("Getting SSMs for sequence %i of %i..."%(seq, NSeq))
            res = []
            #Skip subject 29 because data is missing for some reason
            if NThreads > -1:
                subjs = np.arange(1, NSubj+1)
                subjs = np.concatenate((subjs[0:28], subjs[29::]))
                args = zip(subjs, [seq]*NSubj, [ssmdim]*NSubj, [K]*NSubj)
                res = parpool.map(getSSMsHelper, args)
            else:
                for subj in range(1, NSubj+1):
                    if subj == 29:
                        continue
                    print(".")
                    res.append(getSSMsHelper((subj, seq, ssmdim, K)))
            for i in range(len(res)):
                for k in range(res[0].shape[2]):
                    D = res[i][:, :, k]
                    AllSSMs[k].append(D[I > J])
        VideoSSMs = np.array(AllSSMs[0])
        AudioSSMs = np.array(AllSSMs[1])
        FusedSSMs = np.array(AllSSMs[2])
        sio.savemat("AllSSMs.mat", {"VideoSSMs":VideoSSMs, "AudioSSMs":AudioSSMs, "FusedSSMs":FusedSSMs})
    else:
        print("Loading SSMs...")
        res = sio.loadmat("AllSSMs.mat")
        print("Finished loading SSMs")
        VideoSSMs, AudioSSMs, FusedSSMs = res['VideoSSMs'], res['AudioSSMs'], res['FusedSSMs']
    if not os.path.exists("SSMsDist.mat"):
        print("Getting Video Dists...")
        DVideo = getCSM(VideoSSMs, VideoSSMs)
        print("Getting Audio Dists...")
        DAudio = getCSM(AudioSSMs, AudioSSMs)
        print("Getting Fused DIsts...")
        DFused = getCSM(FusedSSMs, FusedSSMs)
        sio.savemat("SSMsDist.mat", {"DVideo":DVideo, "DAudio":DAudio, "DFused":DFused})
    else:
        res = sio.loadmat("SSMsDist.mat")
        DVideo, DAudio, DFused = res['DVideo'], res['DAudio'], res['DFused']
    AllRQAs = [[], [], []]
    DRQAs = []
    NSubj -= 1 #Skipping subject 29
    for i, SSMs in enumerate([VideoSSMs, AudioSSMs, FusedSSMs]):
        fout = open("RQA%i.arff"%i, "w")
        writeWekaHeader(fout, NSeq)
        for k in range(SSMs.shape[0]):
            print("%i %i"%(i, k))
            D = np.zeros((ssmdim, ssmdim))
            D[I < J] = SSMs[k, :]
            D = D + D.T
            D = CSMToBinaryMutual(D, 0.2)
            AllRQAs[i].append(getRQAArr(getRQAStats(D, 5, 5, do_norm = True))[0])
            print(AllRQAs[i][-1])
            for val in AllRQAs[i][-1]:
                fout.write("%g, "%val)
            fout.write("Seq%i\n"%(1+int(k)/NSubj))
        AllRQAs[i] = np.array(AllRQAs[i])
        DRQAs.append(getSSM(AllRQAs[i]))
        fout.close()
    [DVideoRQA, DAudioRQA, DFusedRQA] = DRQAs
    sio.savemat("SSMsRQA.mat", {"DVideoRQA":DVideoRQA, "DAudioRQA":DAudioRQA, "DFusedRQA":DFusedRQA})

def getPrecisionRecall(pD, NPerClass):
    PR = np.zeros(NPerClass-1)
    D = np.array(pD)
    np.fill_diagonal(D, 0)
    DI = np.array(np.argsort(D, 1), dtype=np.int64)
    for i in range(DI.shape[0]):
        pr = np.zeros(NPerClass-1)
        recall = 0
        for j in range(1, DI.shape[1]): #Skip the first point (don't compare to itself)
            if DI[i, j]/NPerClass == i/NPerClass:
                pr[recall] = float(recall+1)/j
                recall += 1
            if recall == NPerClass-1:
                break
        PR += pr
    return PR/float(DI.shape[0])

def getWarpedTrainingCollection():
    foldername = "oulussms"
    if not os.path.exists(foldername):
        os.mkdir(foldername)
        for t in ["train", "val", "test"]:
            os.mkdir("%s/%s"%(foldername, t))
    # Do 3 warps per pair
    NSeq = 30
    WarpsPerSeq = 4
    trainsubjs = np.arange(1, 29).tolist()
    valsubjs = np.arange(30, 40).tolist()
    testsubjs = np.arange(40, 52).tolist()
    #trainsubjs = [1, 2]
    #valsubjs = [5, 6]
    #testsubjs = [7, 8]
    idx = 0
    for (t, subjs) in zip(["train", "val", "test"], [trainsubjs, valsubjs, testsubjs]):
        idx = 1
        for subj in subjs:
            for seq in range(1, NSeq+1):
                filename = "%s/%s/%i.jpg"%(foldername, t, idx)
                if os.path.exists(filename):
                    print("Skipping subject %i sequence %i"%(subj, seq))
                    idx += WarpsPerSeq
                    continue
                print("Doing subject %i sequence %i"%(subj, seq))
                Images = getAVSSMsWarped(subj, seq, 256, 6, WarpsPerSeq)
                for Im in Images:
                    filename = "%s/%s/%i.jpg"%(foldername, t, idx)
                    scipy.misc.imsave(filename, Im)
                    idx += 1


if __name__ == '__main__2':
    doComparisonExperiments()
    res = sio.loadmat("SSMsDist.mat")
    DVideo, DAudio, DFused = res['DVideo'], res['DAudio'], res['DFused']
    res = sio.loadmat("SSMsRQA.mat")
    DVideoRQA, DAudioRQA, DFusedRQA = res['DVideoRQA'], res['DAudioRQA'], res['DFusedRQA']
    NPerClass = DVideo.shape[0]/30
    AUROCs = []
    for D in [DVideo, DAudio, DFused, DVideoRQA, DAudioRQA, DFusedRQA, np.random.rand(DVideo.shape[0], DVideo.shape[1])]:
        PR = getPrecisionRecall(D, NPerClass)
        plt.plot(PR)
        AUROCs.append(np.mean(PR))
    legend = ["VideoL2", "AudioL2", "FusedL2", "VideoRQA", "AudioRQA", "FusedRQA", "Random"]
    legend = ["%s (%.3g)"%(s, a) for (s, a) in zip(legend, AUROCs)]
    plt.legend(legend)
    plt.show()


if __name__ == '__main__':
    getWarpedTrainingCollection()
    """
    for i in range(1, 30):
        res = getAVSSMsWarped(1, i, 256, 6)
        scipy.misc.imsave("%i.jpg"%i, res)
    """