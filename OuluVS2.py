import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.misc
import sys
from SlidingWindowVideoTDA.VideoTools import *
from Alignment.AlignmentTools import *
from SimilarityFusion import *
from RQA import *
from HKS import *
import scipy.ndimage.morphology
from sklearn.cluster import KMeans
import time
import os
from multiprocessing import Pool as PPool
from Scattering import *
import essentia
import essentia.standard as ess

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
    S = librosa.core.stft(XAudio, winSize, hopSize, center=False)
    print("winSize = %i"%winSize)
    print("S.shape = ", S.shape)
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

def getMFCCsEssentia(filename):
    fs = 44100
    audio = ess.MonoLoader(filename = filename, 
                                          sampleRate = fs)()
    # dynamic range expansion as done in HTK implementation
    audio = audio*2**15

    frameSize = 1102 # corresponds to htk default WINDOWSIZE = 250000.0 
    hopSize = 441 # corresponds to htk default TARGETRATE = 100000.0
    fftSize = 2048
    spectrumSize= fftSize//2+1
    zeroPadding = fftSize - frameSize

    w = ess.Windowing(type = 'hamming', #  corresponds to htk default  USEHAMMING = T
                        size = frameSize, 
                        zeroPadding = zeroPadding,
                        normalized = False,
                        zeroPhase = False)

    spectrum = ess.Spectrum(size = fftSize)

    mfcc_htk = ess.MFCC(inputSize = spectrumSize,
                        type = 'magnitude', # htk uses mel filterbank magniude
                        warpingFormula = 'htkMel', # htk's mel warping formula
                        weighting = 'linear', # computation of filter weights done in Hz domain
                        highFrequencyBound = 8000, # corresponds to htk default
                        lowFrequencyBound = 0, # corresponds to htk default
                        numberBands = 26, # corresponds to htk default  NUMCHANS = 26
                        numberCoefficients = 13,
                        normalize = 'unit_max', # htk filter normaliation to have constant height = 1  
                        dctType = 3, # htk uses DCT type III
                        logType = 'log',
                        liftering = 22) # corresponds to htk default CEPLIFTER = 22


    mfccs = []
    # startFromZero = True, validFrameThresholdRatio = 1 : the way htk computes windows
    for frame in ess.FrameGenerator(audio, frameSize = frameSize, hopSize = hopSize , startFromZero = True, validFrameThresholdRatio = 1):
        spect = spectrum(w(frame))
        mel_bands, mfcc_coeffs = mfcc_htk(spect)
        mfccs.append(mfcc_coeffs)

    return np.array(mfccs)

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

def getAudioVideoFeatures(s, i):
    winSize = 512
    hopSize = 256
    videofilename = "OuluVS2/cropped_mouth_mp4_digit/%i/1/s%i_v1_u%i.mp4"%(s, s, i)
    (I1, IDims) = loadImageIOVideo(videofilename)
    I1 = resizeVideo(I1, IDims, (25, 25))
    audiofilename = "OuluVS2/cropped_audio_dat/s%i_u%i.wav"%(s, i)
    (XAudio, Fs) = getAudioLibrosa(audiofilename)
    #XAudio = np.concatenate((XAudio, np.zeros(winSize-hopSize)))
    I2 = getMFCCsLibrosa(XAudio, Fs, winSize, hopSize)
    #I2 = getMFCCsEssentia(audiofilename)

    #[I1, _] = getTimeDerivative(I1, 5)
    #[I2, _] = getTimeDerivative(I2, 5)
    #I1 = getZNorm(I1)
    #I2 = getZNorm(I2)
    return (I1, I2)

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
    I1, I2 = getAudioVideoFeatures(s, i)
    
    D1 = imresize(getCSM(I1, I1), (ssmdim, ssmdim))
    D2 = imresize(getCSM(I2, I2), (ssmdim, ssmdim))

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
    

def getSSMsHelper(args):
    (subj, seq, ssmdim, K) = args
    print(args)
    return getAVSSMsFused(subj, seq, ssmdim, K)


def promoteDiagonal(W, bias):
    """
    Make things off diagonal less similar
    """
    N = W.shape[0]
    I, J = np.meshgrid(np.arange(N), np.arange(N))
    weight = bias + (1.0-bias)*(1.0 - np.abs(I-J)/float(N))
    weight = weight**4
    return weight*W

def get_simulated_missing_chunks(perc_dropped, drop_chunk, ssmdim):
    chunklen = int(drop_chunk*ssmdim)
    nchunks = int(perc_dropped/drop_chunk)
    idxs = np.arange(ssmdim)
    missingidx = np.array([], dtype=int)
    for c in range(nchunks):
        i1 = np.random.randint(0, len(idxs)-chunklen+1)
        missingidx = np.concatenate((missingidx, idxs[i1:i1+chunklen]))
        idxs = np.concatenate((idxs[0:i1], idxs[i1+1::]))
    return missingidx

def doComparisonExperiments(NThreads = 8):
    ssmdim = 400
    ssmdimres = 256
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
    NSubj -= 1 #Skipping subject 29

    plt.figure(figsize=(20, 5))
    # Now compute all of the persistence diagrams
    DsScatter = []
    batchsize = 100
    for i, SSMs in enumerate([VideoSSMs, AudioSSMs, FusedSSMs]):
        Ds = []
        for k in range(SSMs.shape[0]):
            print("%i %i"%(i, k))
            D = np.zeros((ssmdim, ssmdim))
            D[I > J] = SSMs[k, :]
            D = D + D.T
            W = np.array(D)
            if i < 2:
                W = getW(D, K) #Convert to similarity matrix
            W[np.abs(I-J) <= 1] = 0 #Zero diagonal and off-diagonal
            W = imresize(W, (ssmdimres, ssmdimres))
            Ds.append(W)
        AllScattering = []
        NBatches = int(np.ceil(1.0*len(Ds)/batchsize))
        for k in range(NBatches):
            print("Doing SSMType %i batch %i of %i"%(i, k, NBatches))
            AllScattering += getScatteringTransform(Ds[k*batchsize:(k+1)*batchsize], renorm=False)
        ScatteringFeats = np.array([])
        for j, images in enumerate(AllScattering):
            scattering = np.array([])
            for k in range(len(images)):
                norm = np.sqrt(np.sum(images[k]**2))
                if norm == 0:
                    norm = 1
                images[k] /= norm
                scattering = np.concatenate((scattering, images[k].flatten()))
            """
            plt.clf()
            plt.subplot(1, len(images)+1, 1)
            plt.imshow(Ds[j], cmap = 'afmhot')
            for k in range(len(images)):
                plt.subplot(1, len(images)+1, k+2)
                plt.imshow(images[k], cmap='afmhot')
                plt.title("Scattering %i"%k)
            """
            if ScatteringFeats.size == 0:
                ScatteringFeats = np.zeros((len(AllScattering), scattering.size), dtype=np.float32)
            ScatteringFeats[j, :] = scattering.flatten()
            #plt.savefig("%i_%i.png"%(i, j), bbox_inches='tight')
        ScatteringFeats[np.isnan(ScatteringFeats)] = 0
        DsScatter.append(getSSM(ScatteringFeats))
    [DVideoScatter, DAudioScatter, DFusedScatter] = DsScatter
    sio.savemat("SSMsScatter.mat", {"DVideoScatter":DVideoScatter, "DAudioScatter":DAudioScatter, "DFusedScatter":DFusedScatter})


def doComparisonExperimentsRaw():
    NSeq = 30
    NSubj = 52
    VideoRes = 256
    AudioRes = 512
    tvideo = np.linspace(0, 1, VideoRes)
    taudio = np.linspace(0, 1, AudioRes)

    AudioFeats = []
    VideoFeats = []
    for seq in range(1, NSeq+1):
        print("Getting features for sequence %i of %i..."%(seq, NSeq))
        #Skip subject 29 because data is missing for some reason
        for subj in range(1, NSubj+1):
            if subj == 29:
                continue
            print(".")
            IVideo, IAudio = getAudioVideoFeatures(subj, seq)
            IVideo = getInterpolatedEuclideanTimeSeries(IVideo, tvideo)
            IAudio = getInterpolatedEuclideanTimeSeries(IAudio, taudio)
            AudioFeats.append(IAudio.flatten())
            VideoFeats.append(IVideo.flatten())
    NSubj -= 1 #Skipping subject 29
    AudioFeats = np.array(AudioFeats, dtype=np.float32)
    VideoFeats = np.array(VideoFeats, dtype=np.float32)
    DAudioRaw = getSSM(AudioFeats)
    AudioFeats = None
    DVideoRaw = getSSM(VideoFeats)
    sio.savemat("SSMsRaw.mat", {"DAudioRaw":DAudioRaw, "DVideoRaw":DVideoRaw})



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


def makePrecisionRecall():
    res = sio.loadmat("OuluVS2Results/SSMsDist.mat")
    DVideo, DAudio, DFused = res['DVideo'], res['DAudio'], res['DFused']
    res = sio.loadmat("OuluVS2Results/SSMsScatter.mat")
    DVideoScatter, DAudioScatter, DFusedScatter = res['DVideoScatter'], res['DAudioScatter'], res['DFusedScatter']
    res = sio.loadmat("OuluVS2Results/SSMsRQA.mat")
    DVideoRQA, DAudioRQA, DFusedRQA = res['DVideoRQA'], res['DAudioRQA'], res['DFusedRQA']
    res = sio.loadmat("OuluVS2Results/SSMsRaw.mat")
    DAudioRaw, DVideoRaw = res['DAudioRaw'], res['DVideoRaw']
    NPerClass = DVideo.shape[0]/30
    AUROCs = []
    for D in [DVideo, DAudio, DFused, DVideoScatter, DAudioScatter, DFusedScatter, \
             DVideoRQA, DAudioRQA, DFusedRQA, DVideoRaw, DAudioRaw,\
             np.random.rand(DVideo.shape[0], DVideo.shape[1])]:
        PR = getPrecisionRecall(D, NPerClass)
        plt.plot(PR)
        AUROCs.append(np.mean(PR))
    legend = ["VideoL2", "AudioL2", "FusedL2", "VideoScatter", "AudioScatter", "FusedScatter", "VideoRQA", "AudioRQA", "FusedRQA", "VideoRaw", "AudioRaw", "Random"]
    legend = ["%s (%.3g)"%(s, a) for (s, a) in zip(legend, AUROCs)]
    plt.legend(legend)
    plt.show()

def getD(SSMs, k, ssmdim):
    [I, J] = np.meshgrid(np.arange(ssmdim), np.arange(ssmdim))
    D = np.zeros((ssmdim, ssmdim))
    D[I > J] = SSMs[k, :]
    D = D + D.T
    D[D < 0] = 0
    return D



def missingDataExample(subj, seq, stretch_fac, perc_dropped = 0.1, drop_chunk = 0.02, ssmdim = 400, use_precomputed = False, plot_SSMs = True):
    from SSMGUI import saveResultsJSON
    import pyrubberband as pyrb

    #For similarity fusion
    K = int(0.1*ssmdim)
    NIters = 10
    
    NSeq = 30
    NSubj = 51

    # Step 1: Load in or compute SSMs
    if use_precomputed:
        k = seq*NSubj+subj
        print("k = %i"%k)
        print("Loading SSMs...")
        res = sio.loadmat("AllSSMs.mat")
        print("Finished loading SSMs")
        VideoSSMs, AudioSSMs, FusedSSMs = res['VideoSSMs'], res['AudioSSMs'], res['FusedSSMs']
        DVideo = getD(VideoSSMs, k, ssmdim)
        DAudio = getD(AudioSSMs, k, ssmdim)
        DFused = getD(FusedSSMs, k, ssmdim)
    else:
        IVideo, IAudio = getAudioVideoFeatures(subj+1, seq+1)
        DAudio = imresize(getCSM(IAudio, IAudio), (ssmdim, ssmdim))
        DVideo = imresize(getCSM(IVideo, IVideo), (ssmdim, ssmdim))

    # Convert to similarity scale from metric scale on video/audio
    DVideo = getW(DVideo, K)
    DAudio = getW(DAudio, K)

    # Step 2: Randomly drop audio data and do the fusion
    missingidx = get_simulated_missing_chunks(perc_dropped, drop_chunk, ssmdim)
    DAudioDisp = np.array(DAudio)
    DAudioDisp[missingidx, :] = -0.3
    DAudioDisp[:, missingidx] = -0.3
    DAudio[missingidx, :] = DVideo[missingidx, :]
    DAudio[:, missingidx] = DVideo[:, missingidx]
    
    DFused = doSimilarityFusionWs([DAudio, DVideo], K=K, NIters=NIters)

    # Step 3: Plot Results
    np.fill_diagonal(DAudio, 0)
    np.fill_diagonal(DVideo, 0)
    np.fill_diagonal(DFused, np.min(DFused))

    plt.figure(figsize=(12, 12))
    if plot_SSMs:
        Composited = np.zeros((ssmdim, ssmdim, 3))
        Composited[:, :, 0] = DVideo
        Composited[:, :, 1] = DAudio
        Composited[:, :, 0] *= 255.0/np.max(Composited[:, :, 0])
        Composited[:, :, 1] *= 255.0/np.max(Composited[:, :, 1])
        Composited = np.array(np.round(Composited), dtype=np.uint8)
        plt.subplot(221)
        plt.imshow(DVideo)
        plt.colorbar()
        plt.title("Video")
        plt.subplot(222)
        plt.imshow(DAudioDisp)
        plt.colorbar()
        plt.title("Audio")
        plt.subplot(223)
        plt.imshow(Composited)
        plt.colorbar()
        plt.title("Composited")
        plt.subplot(224)
        plt.imshow(DFused)
        plt.colorbar()
        plt.title("Fused")
        plt.savefig("%i_%i.png"%(subj+1, seq+1), bbox_inches='tight')

    # Step 4: Output (possibly stretched) audio
    audiofilename = "OuluVS2/cropped_audio_dat/s%i_u%i.wav"%(subj+1, seq+1)
    XAudio, Fs = getAudioLibrosa(audiofilename)
    XAudio = pyrb.time_stretch(XAudio, Fs, stretch_fac)
    sio.wavfile.write("temp.wav", Fs, XAudio)
    filename = "%i_%i.mp3"%(subj+1, seq+1)
    if os.path.exists(filename):
        os.remove(filename)
    subprocess.call(["avconv", "-i", "temp.wav", filename])

    # Step 5: Output (possibly stretched) video
    videofilename = "OuluVS2/cropped_mouth_mp4_digit/%i/1/s%i_v1_u%i.mp4"%(subj+1, subj+1, seq+1)
    #TODO: FINISH THIS

    # Step 6: Output JSON file for GUI
    LenSec = float(XAudio.size)/Fs
    times = np.linspace(0, LenSec, ssmdim)
    saveResultsJSON(filename, times, DFused, "%i_%i.json"%(subj+1, seq+1))

if __name__ == '__main__2':
    doComparisonExperiments()
    #doComparisonExperimentsRaw()

if __name__ == '__main__':
    np.random.seed(4)
    missingDataExample(subj = 20, seq = 10, stretch_fac = 0.3, perc_dropped=0.0,\
                         drop_chunk=0.05, use_precomputed=True)

if __name__ == '__main__2':
    makePrecisionRecall()

if __name__ == '__main__2':
    getWarpedTrainingCollection()
    """
    for i in range(1, 30):
        res = getAVSSMsWarped(1, i, 256, 6)
        scipy.misc.imsave("%i.jpg"%i, res)
    """
