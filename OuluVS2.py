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
from sklearn.cluster import KMeans
import time
import os
from multiprocessing import Pool as PPool
from Scattering import *


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

def getMFCCsLibrosa(XAudio, Fs, winSize, hopSize = 512, NBands = 40, fmax = 8000, NMFCC = 20, lifterexp = 0):
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

def getAudioVideoFeatures(s, i):
    winSize = 512
    hopSize = 256
    videofilename = "OuluVS2/cropped_mouth_mp4_digit/%i/1/s%i_v1_u%i.mp4"%(s, s, i)
    (I1, IDims) = loadImageIOVideo(videofilename)
    I1 = resizeVideo(I1, IDims, (25, 25))
    audiofilename = "OuluVS2/cropped_audio_dat/s%i_u%i.wav"%(s, i)
    (XAudio, Fs) = getAudioLibrosa(audiofilename)
    XAudio = np.concatenate((XAudio, np.zeros(winSize-hopSize)))
    I2 = getMFCCsLibrosa(XAudio, Fs, winSize, hopSize)
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
    
    return np.concatenate((D1[:, :, None], D2[:, :, None], D3[:, :, None]), 2)
    

def getSSMsHelper(args):
    (subj, seq, ssmdim, K) = args
    print(args)
    return getAVSSMsFused(subj, seq, ssmdim, K)

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

def getD(SSMs, k, ssmdim):
    """
    Reconstruct an SSM from the upper triangular part
    """
    [I, J] = np.meshgrid(np.arange(ssmdim), np.arange(ssmdim))
    D = np.zeros((ssmdim, ssmdim))
    D[I > J] = SSMs[k, :]
    D = D + D.T
    D[D < 0] = 0
    return D

def doComparisonExperiments(NThreads = 8, perc_dropped = 0.0, drop_chunk = 0.02, seed = 0, plot_scattering = False):
    ssmdim = 400
    ssmdimres = 256
    K = int(ssmdim*0.1)
    [I, J] = np.meshgrid(np.arange(ssmdim), np.arange(ssmdim))
    NSeq = 30
    NSubj = 52

    ## Step 1: Compute SSMs
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

    ## Step 2: Convert self similarity to neighborhood normalized dissimilarity
    ##          and Simulate missing data if applicable
    np.random.seed(seed)
    for k in range(AudioSSMs.shape[0]):
        print("Fusing missing data %i of %i"%(k, AudioSSMs.shape[0]))
        WAudio = getW(getD(AudioSSMs, k, ssmdim), K)
        WVideo = getW(getD(VideoSSMs, k, ssmdim), K)
        if perc_dropped > 0:
            missingidx = get_simulated_missing_chunks(perc_dropped, drop_chunk, ssmdim)
            WAudio[missingidx, :] = WVideo[missingidx, :]
            WAudio[:, missingidx] = WVideo[:, missingidx]
            WFused = doSimilarityFusionWs([WAudio, WVideo], K=K)
            FusedSSMs[k, :] = WFused[I > J]
        AudioSSMs[k, :] = WAudio[I > J]
        VideoSSMs[k, :] = WVideo[I > J]
        

    ## Step 3: Compute L2 distances
    DVideoL2 = getCSM(VideoSSMs, VideoSSMs)
    DAudioL2 = getCSM(AudioSSMs, AudioSSMs)
    DFusedL2 = getCSM(FusedSSMs, FusedSSMs)
    sio.savemat("SSMsDist.mat", {"DAudio":DAudioL2, "DVideo":DVideoL2, "DFused":DFusedL2})


    plt.figure(figsize=(20, 5))
    ## Step 4: Compute scattering transform
    NSubj -= 1 #Skipping subject 29
    DsScatter = []
    batchsize = 100
    for i, SSMs in enumerate([VideoSSMs, AudioSSMs, FusedSSMs]):
        Ds = []
        for k in range(SSMs.shape[0]):
            print("%i %i"%(i, k))
            W = np.array(getD(SSMs, k, ssmdim))
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
            if plot_scattering:
                plt.clf()
                plt.subplot(1, len(images)+1, 1)
                plt.imshow(Ds[j], cmap = 'afmhot')
                for k in range(len(images)):
                    plt.subplot(1, len(images)+1, k+2)
                    plt.imshow(images[k], cmap='afmhot')
                    plt.title("Scattering %i"%k)
                plt.savefig("%i_%i.png"%(i, j), bbox_inches='tight')
            if ScatteringFeats.size == 0:
                ScatteringFeats = np.zeros((len(AllScattering), scattering.size), dtype=np.float32)
            ScatteringFeats[j, :] = scattering.flatten()
        ScatteringFeats[np.isnan(ScatteringFeats)] = 0
        DsScatter.append(getSSM(ScatteringFeats))
    [DVideoScatter, DAudioScatter, DFusedScatter] = DsScatter
    sio.savemat("SSMsScatter.mat", {"DVideoScatter":DVideoScatter, "DAudioScatter":DAudioScatter, "DFusedScatter":DFusedScatter})


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
    np.fill_diagonal(DAudioDisp, 0)
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
    saveResultsJSON(filename, times, DAudio, "%i_%i.json"%(subj+1, seq+1))

if __name__ == '__main__':
    doComparisonExperiments()

if __name__ == '__main__2':
    np.random.seed(4)
    missingDataExample(subj = 20, seq = 10, stretch_fac = 0.3, perc_dropped=0.0,\
                         drop_chunk=0.05, use_precomputed=True)