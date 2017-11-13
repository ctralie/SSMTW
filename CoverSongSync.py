import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.ndimage
import sys
import time
import pyrubberband as pyrb
from SlidingWindowVideoTDA.VideoTools import *
from Alignment.AlignmentTools import *
from Alignment.DTWGPU import *
from Alignment.AllTechniques import *
from GeometricCoverSongs.CSMSSMTools import *
from GeometricCoverSongs.BlockWindowFeatures import *
from GeometricCoverSongs.pyMIRBasic.AudioIO import *
from GeometricCoverSongs.pyMIRBasic.Onsets import *
from GeometricCoverSongs.SimilarityFusion import *

def getSimilarityFusedSSMs(Features):
    X = Features['MFCCs']
    DMFCCs = getCSM(X, X)
    X = Features['Chromas']
    DChromas = getCSMCosine(X, X)
    Ds = [DMFCCs, DChromas]
    D = doSimilarityFusion(Ds, K = 5, NIters = 10, reg = 1, PlotNames = ['MFCCs', 'Chromas']) 
    return {"D":D, "DMFCCs":DMFCCs, "DChromas":DChromas}

def doSync(FeatureParams, filename1, filename2, hopSize = 512, winFac = 2):
    print "Loading %s..."%filename1
    (XAudio1, Fs) = getAudioLibrosa(filename1)
    print("Fs = ", Fs)
    print "Loading %s..."%filename2
    (XAudio2, Fs) = getAudioLibrosa(filename2)
    print("Fs = ", Fs)
    tempo = 120
    #Compute features in intervals evenly spaced by the hop size
    nHops = int((XAudio1.size-hopSize*winFac*FeatureParams['MFCCBeatsPerBlock'])/hopSize)
    beats1 = np.arange(0, nHops, winFac)
    nHops = int((XAudio2.size-hopSize*winFac*FeatureParams['MFCCBeatsPerBlock'])/hopSize)
    beats2 = np.arange(0, nHops, winFac)
    print("Getting Features1...")
    (Features1, O1) = getBlockWindowFeatures((XAudio1, Fs, tempo, beats1, hopSize, FeatureParams))
    print("Getting Features2...")
    (Features2, O2) = getBlockWindowFeatures((XAudio2, Fs, tempo, beats2, hopSize, FeatureParams))
    D1 = getSimilarityFusedSSMs(Features1)
    D2 = getSimilarityFusedSSMs(Features2)
       
    sio.savemat("D1.mat", D1)
    sio.savemat("D2.mat", D2)

if __name__ == '__main__':
    hopSize = 512
    winFac = 2
    zoom = 1
    filename1 = "MJ.mp3"
    filename2 = "AAF.mp3"
    print "Loading %s..."%filename1
    (XAudio1, Fs) = getAudioLibrosa(filename1)
    print "Loading %s..."%filename2
    (XAudio2, Fs) = getAudioLibrosa(filename2)


    initParallelAlgorithms()
    D1 = sio.loadmat("D1.mat")["D"]
    D2 = sio.loadmat("D2.mat")["D"]
    
    #D1 = D1[153:990, 153:990]
    #D2 = D2[1:786, 1:786]
    #offset1 = 153
    #offset2 = 1
    offset1 = 0
    offset11 = int(D1.shape[0]/2)#1725
    offset2 = 0
    offset22 = int(D2.shape[0]/2)#1427
    D1 = D1[offset1:offset11, offset1:offset11]
    D2 = D2[offset2:offset22, offset2:offset22]
    [I, J] = np.meshgrid(np.arange(D1.shape[0]), np.arange(D1.shape[1]))
    D1[np.abs(I - J) < 8] = 0
    [I, J] = np.meshgrid(np.arange(D2.shape[0]), np.arange(D2.shape[1]))
    D2[np.abs(I - J) < 8] = 0
    D1 = scipy.ndimage.interpolation.zoom(D1, zoom)
    D2 = scipy.ndimage.interpolation.zoom(D2, zoom)

    (D2, D1) = matchSSMDist(D2, D1)
    
    plt.subplot(221)
    plt.imshow(D1, cmap = 'afmhot', interpolation = 'none')
    plt.subplot(222)
    plt.imshow(D2, cmap = 'afmhot', interpolation = 'none')
    plt.subplot(223)
    plt.plot(D1[0, :])
    plt.plot(D2[0, :])
    plt.show()
    
    """
    CSWM = doIBDTWGPU(D1, D2, returnCSM = True)
    (DAll, CSSM, backpointers, path) = DTWCSM(CSWM)
    """
    
    """
    print("Doing IBSMWat...")
    #matchfn = lambda x: np.exp(-x/(0.3**2))-0.6
    hvPenalty = -0.4
    #CSWM = doIBSMWatGPU(D1, D2, hvPenalty, True)
    CSWM = sio.loadmat("PCSWM_winFac2.mat")["CSWM"]
    #CSWM = doIBSMWat(D1, D2, matchfn, hvPenalty, Verbose = True)
    CSWM = CSWM - np.median(CSWM)
    CSWM = CSWM/np.max(np.abs(CSWM))
    matchfn = lambda x: x
    hvPenalty = -0.4
    res = SMWat(CSWM, matchfn, hvPenalty, backtrace = True)
    path = res['path']
    path = np.flipud(path)
    
    sio.savemat("CSWM.mat", {"CSWM":CSWM, "path":path})
    """
    Saved = sio.loadmat("CSWM.mat")
    CSWM = Saved['CSWM']
    path = Saved['path']
    
    path = makePathStrictlyIncrease(path)
    plt.imshow(CSWM, cmap = 'afmhot', interpolation = 'none')
    plt.scatter(path[:, 1], path[:, 0])
    plt.show()
    
    XFinal = np.array([[0, 0]])
    fileprefix = ""
    for i in range(path.shape[0]-1):
        [j, k] = [path[i, 0], path[i, 1]]
        [j2, k2] = [path[i+1, 0], path[i+1, 1]]
        print("i = %i, j = %i, j2 = %i, k = %i, k2 = %i"%(i, j, j2, k, k2))
        t1 = int(j*winFac*hopSize/zoom) + offset1*hopSize
        t2 = int(j2*winFac*hopSize/zoom) + offset1*hopSize
        s1 = int(k*winFac*hopSize/zoom) + offset2*hopSize
        s2 = int(k2*winFac*hopSize/zoom) + offset2*hopSize
        x1 = XAudio1[t1:t2]
        x2 = XAudio2[s1:s2]
        #Figure out the time factor by which to stretch x2 so it aligns
        #with x1
        fac = float(len(x1))/len(x2)
        print "fac = ", fac
        x2 = pyrb.time_stretch(x2, Fs, 1.0/fac)
        print "len(x1) = %i, len(x2) = %i"%(len(x1), len(x2))
        N = min(len(x1), len(x2))
        x1 = x1[0:N]
        x2 = x2[0:N]
        X = np.zeros((N, 2))
        X[:, 0] = x1
        X[:, 1] = x2
        if len(fileprefix) > 0:
            filename = "%s_%i.mp3"%(fileprefix, i)
            sio.wavfile.write("temp.wav", Fs, X)
            subprocess.call(["avconv", "-i", "temp.wav", filename])
        XFinal = np.concatenate((XFinal, X))
    
    sio.wavfile.write("Synced.wav", Fs, XFinal)

    
    
    

if __name__ == '__main__2':
    FeatureParams = {'MFCCBeatsPerBlock':20, 'MFCCSamplesPerBlock':200, 'ChromaBeatsPerBlock':20, 'ChromasPerBlock':40}
    #filename1 = "ELPOrig.webm"
    #filename2 = "ELPCover.m4a"
    filename1 = "MJ.mp3"
    filename2 = "AAF.mp3"
    doSync(FeatureParams, filename1, filename2)
