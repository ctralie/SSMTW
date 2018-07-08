import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.misc
import sys
from SlidingWindowVideoTDA.VideoTools import *
from Alignment.AlignmentTools import *
from Alignment.DTWGPU import *
from Alignment.AllTechniques import *
import time

def getZNorm(X):
    Y = X - np.mean(X, 0)[None, :]
    Norm = np.sqrt(np.sum(Y**2, 1))
    Norm[Norm == 0] = 1
    Y = Y/Norm[:, None]
    return Y

def getAudioLibrosa(filename):
    r"""
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

    #if usecmp:
    #    #Hynek's magical equal-loudness-curve formula
    #    fsq = M**2
    #    ftmp = fsq + 1.6e5
    #    eql = ((fsq/ftmp)**2)*((fsq + 1.44e6)/(fsq + 9.61e6))

    X = M.dot(np.abs(S))
    X = librosa.core.logamplitude(X)
    X = np.dot(librosa.filters.dct(NMFCC, X.shape[0]), X) #Make MFCC
    #Do liftering
    coeffs = np.arange(NMFCC)**lifterexp
    coeffs[0] = 1
    X = coeffs[:, None]*X
    X = np.array(X, dtype = np.float32).T
    
    if doZNorm:
        X = getZNorm(X)
    return X

def resizeVideo(I, IDims, NewDims, doPlot = False):
    INew = []
    for i in range(I.shape[0]):
        F = np.reshape(I[i, :], IDims)
        F = rgb2gray(F)
        F = 1.0*scipy.misc.imresize(F, NewDims)
        if doPlot:
            plt.imshow(F)
            plt.show()
        INew.append(F.flatten())
    INew = np.array(INew)
    print("I.dtype = ", I.dtype)
    print("INew.dtype = ", INew.dtype)
    return INew

def runAlignmentExperiments(eng, seed, K, NSubjects, NPerSubject, doPlot = False):
    """
    Run experiments randomly warping the audio and trying to align it to
    the video
    """
    np.random.seed(seed)
    AllErrors = {}
    for s in range(1, NSubjects+1):
        for i in range(1, NPerSubject+1):
            print("Doing subject %i trial %i..."%(s, i))
            filename = "OuluVS2/cropped_mouth_mp4_digit/%i/1/s%i_v1_u%i.mp4"%(s, s, i)
            if not os.path.exists(filename):
                continue
            (I1, IDims) = loadImageIOVideo(filename)
            I1 = getPCAVideo(I1)
            filename = "OuluVS2/cropped_audio_dat/s%i_u%i.wav"%(s, i)
            if not os.path.exists(filename):
                continue
            (XAudio, Fs) = getAudioLibrosa(filename)
            I2 = getMFCCsLibrosa(XAudio, Fs, 4096, int(Fs/30))

            #[I1, _] = getTimeDerivative(I1, derivWin)
            #[I2, _] = getTimeDerivative(I2, derivWin)
            #I1 = getSlidingWindowVideo(I1, SWWin, 1, 1)
            #I2 = getSlidingWindowVideo(I2, SWWin, 1, 1)
            I1 = getZNorm(I1)
            I2 = getZNorm(I2)
            
            WarpDict = getWarpDictionary(I2.shape[0])
            t2 = getWarpingPath(WarpDict, K, False)
            I2Warped = getInterpolatedEuclideanTimeSeries(I2, t2)
            plt.clf()
            (errors, Ps) = doAllAlignments(eng, I1, I2Warped, t2, drawPaths = doPlot)
            if doPlot:
                plt.savefig("Oulu%i_%i.svg"%(s, i))
                plt.clf()
                D1 = getSSM(I1)
                D2 = getSSM(I2)
                plt.subplot(121)
                plt.imshow(D1, cmap = 'afmhot', interpolation = 'none')
                plt.title("Video")
                plt.subplot(122)
                plt.imshow(D2, cmap = 'afmhot', interpolation = 'none')
                plt.title("Audio")
                plt.savefig("Oulu%i_%iSSM.png"%(s, i), bbox_inches = 'tight')
                plt.clf()
            types = errors.keys()
            for t in types:
                if not t in AllErrors:
                    AllErrors[t] = np.inf*np.ones((NSubjects, NPerSubject))
                AllErrors[t][s-1][i-1] = errors[t]
            sio.savemat("OuluVs2Errors.mat", AllErrors)

if __name__ == '__main__':
    initParallelAlgorithms()
    eng = initMatlabEngine()
    runAlignmentExperiments(eng, 100, 4, 10, 30, True)

if __name__ == '__main__2':
    initParallelAlgorithms()
    s = 2
    i = 3
    filename = "OuluVS2/cropped_mouth_mp4_digit/%i/1/s%i_v1_u%i.mp4"%(s, s, i)
    print(filename)
    (I1, IDims) = loadImageIOVideo(filename)
    #I1 = resizeVideo(I1, IDims, (25, 50))
    filename = "OuluVS2/cropped_audio_dat/s%i_u%i.wav"%(s, i)
    print(filename)
    (XAudio, Fs) = getAudioLibrosa(filename)
    I2 = getMFCCsLibrosa(XAudio, Fs, 4096, int(Fs/30))

    #[I1, _] = getTimeDerivative(I1, 5)
    #[I2, _] = getTimeDerivative(I2, 5)
    I1 = getZNorm(I1)
    I2 = getZNorm(I2)

    t = np.linspace(0, 1, I2.shape[0])
    t2 = t**2
    I2Orig = np.array(I2)
    I2 = getInterpolatedEuclideanTimeSeries(I2, t2)
    
    D1 = getSSM(I1)
    D2 = getSSM(I2Orig)
    plt.subplot(121)
    plt.imshow(D1, cmap = 'afmhot', interpolation = 'none')
    plt.title("Video")
    plt.subplot(122)
    plt.imshow(D2, cmap = 'afmhot', interpolation = 'none')
    plt.title("Audio")
    
    plt.savefig("SSMs.svg", bbox_inches = 'tight')
    plt.clf()
    getIBDTWAlignment(I1, I2, doPlot = True)
    for i in range(3):
        plt.subplot(3, 3, 7+i)
        plt.scatter(t*I2.shape[0], t2*I2.shape[0], edgecolor = 'none')
    plt.savefig("OuluAligned.svg", bbox_inches = 'tight')

if __name__ == '__main__2':
    initParallelAlgorithms()
    
    for i in range(1, 31):
        #Get Video SSM
        (I, IDims) = loadImageIOVideo("OuluVS2/cropped_mouth_mp4_digit/1/1/s1_v1_u%i.mp4"%i)
        t = np.linspace(0, 1, I.shape[0])
        t = t**1.5
        IWarped = getInterpolatedEuclideanTimeSeries(I, t)
        
        #Get Audio SSM
        (XAudio, Fs) = getAudioLibrosa("OuluVS2/cropped_audio_dat/s1_u%i.wav"%i)
        X = getMFCCsLibrosa(XAudio, Fs, 4096, int(Fs/30))
        
        print("I.shape = {}".format(I.shape))
        print("X.shape = {}".format(X.shape))
        getIBDTWAlignment(IWarped, X, doPlot = True)
        plt.savefig("Oulu1_%i.svg"%i, bbox_inches = 'tight')
        plt.clf()
