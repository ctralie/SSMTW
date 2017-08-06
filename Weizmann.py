import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from SlidingWindowVideoTDA.VideoTools import *
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
    
def getWeiAlignedMask(action):
    X = sio.loadmat(WEIPATH + "/mask.mat")
    X = X['aligned_masks'][0][0]
    
    names = getWeiNamesFromStruct(X.dtype)
    idx = 0
    for idx in range(len(names)):
        if names[idx] == action:
            break
    print "name = %s, idx = %i"%(action, idx)
    I = X[idx]
    I = np.rollaxis(I, 2, 0)
    IDims = I.shape[1::]
    I = np.reshape(I, (I.shape[0], IDims[0]*IDims[1]))
    idx = WEICROP[action]
    I = I[idx[0]:idx[1], :]
    return (I, IDims)

def alignWalkingVideos(eng):
    IsMask = []
    Is = []
    SSMs = []
    reindexes = [[]]
    counter = 1
    plt.figure(figsize=(5*len(WEIVIDEOS), 10))
    for video in WEIVIDEOS:
        (I, IDims) = getWeiAlignedMask(video)
        IsMask.append(1.0*I)
        D = getSSM(I)
        D = get2DRankSSM(D)
        SSMs.append(D)
        plt.subplot(2, len(WEIVIDEOS), counter)
        plt.imshow(SSMs[-1], cmap = 'afmhot', interpolation = 'nearest')
        plt.title(video)
        
        if counter > 1:
            D = doIBDTW(SSMs[0], SSMs[-1])
            plt.subplot(2, len(WEIVIDEOS), len(WEIVIDEOS) + counter)
            plt.imshow(D, cmap = 'afmhot', interpolation = 'nearest')
            (DAll, CSM, backpointers, path) = DTWCSM(D)
            plt.scatter(path[:, 1], path[:, 0],5, 'c', edgecolor = 'none')
            reindexes.append(projectPath(path))
            #res = getCTWAlignments(eng, IsMask[0], IsMask[-1])
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
        F = np.reshape(Is[0][0][i, :], Is[0][1])
        plt.imshow(F)
        plt.axis("off")
        for k in range(1, len(Is)):
            plt.subplot(1, len(WEIVIDEOS), k+1)
            idxs = reindexes[k]
            F = np.reshape(Is[k][0][idxs[i], :], Is[k][1])
            plt.imshow(F)
            plt.axis('off')
            plt.title("%i"%idxs[i])
        plt.savefig("%i.png"%i, bbox_inches = 'tight')

if __name__ == '__main__':
    eng = initMatlabEngine()
    alignWalkingVideos(eng)

if __name__ == '__main__2':
    X = sio.loadmat(WEIPATH + "/mask.mat")
    X = X['aligned_masks'][0][0]
    names = getWeiNamesFromStruct(X.dtype)
    #print names

    for video in WEIVIDEOS:
        (I, IDims) = getWeiAlignedMask(video)
        saveVideo(I, IDims, "%s.avi"%video, 20)
