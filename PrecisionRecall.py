import numpy as np 
import scipy.io as sio 
import matplotlib.pyplot as plt

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

def makePrecisionRecall(resultsdir = "OuluVS2Results"):
    plt.figure(figsize=(9, 5))
    res = sio.loadmat("%s/SSMsDist.mat"%resultsdir)
    DVideo, DAudio, DFused = res['DVideo'], res['DAudio'], res['DFused']
    res = sio.loadmat("%s/SSMsScatter.mat"%resultsdir)
    DVideoScatter, DAudioScatter, DFusedScatter = res['DVideoScatter'], res['DAudioScatter'], res['DFusedScatter']
    #res = sio.loadmat("%s/SSMsRQA.mat"%resultsdir)
    #DVideoRQA, DAudioRQA, DFusedRQA = res['DVideoRQA'], res['DAudioRQA'], res['DFusedRQA']
    res = sio.loadmat("%s/SSMsRaw.mat"%resultsdir)
    DAudioRaw, DVideoRaw = res['DAudioRaw'], res['DVideoRaw']
    NPerClass = DVideo.shape[0]/10
    AUROCs = []
    for D in [DVideo, DAudio, DFused, DVideoScatter, DAudioScatter, DFusedScatter, \
             DAudioRaw, DVideoRaw,\
             #DVideoRQA, DAudioRQA, DFusedRQA, \
             np.random.rand(DVideo.shape[0], DVideo.shape[1])]:
        PR = getPrecisionRecall(D, NPerClass)
        plt.plot(PR)
        AUROCs.append(np.mean(PR))
    legend = ["VideoL2", "AudioL2", "FusedL2", "VideoScatter", "AudioScatter", "FusedScatter", "AudioRaw", "VideoRaw", "Random"]
    legend = ["%s (%.3g)"%(s, a) for (s, a) in zip(legend, AUROCs)]
    plt.legend(legend)
    plt.savefig("%s_PrecisionRecall.svg"%resultsdir, bbox_inches='tight')

if __name__ == '__main__':
    makePrecisionRecall("OuluVS2Results_Win4096")