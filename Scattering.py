"""
Code that wraps around Matlab ScatNet to compute scattering transforms
"""
import numpy as np
import scipy.io as sio
import subprocess
import matplotlib.pyplot as plt
import os
import sys
from Alignment.AlignmentTools import *

def getPrefix():
    return 'scatnet-0.2'

def getScatteringTransform(imgs, renorm=True):
    intrenorm = 0
    if renorm:
        intrenorm = 1
    prefix = getPrefix()
    sio.savemat("%s/x.mat"%prefix, {"x":imgs, "renorm":intrenorm})
    subprocess.call(["matlab", "-nodisplay", "-nosplash", "-nodesktop", "-r", \
                    "cd %s; getScatteringImages; exit;"%(prefix)])
    res = sio.loadmat("%s/res.mat"%prefix)['res']
    images = []
    for i in range(len(res[0])):
        image = []
        for j in range(len(res[0][i][0])):
            image.append(res[0][i][0][j])
        images.append(image)
    return images

def flattenCoefficients(images):
    ret = []
    for im in images:
        ret.append(im.flatten())
    return np.array(ret)

def poolFeatures(image, res):
    M = int(image.shape[0]/res)
    N = int(image.shape[1]/res)
    ret = np.zeros((M, N))
    for i in range(M):
        for j in range(N):
            ret[i, j] = np.mean(image[i*res:(i+1)*res, j*res:(j+1)*res])
    return ret

def testScatteringBasicWarp():
    N = 256
    t1 = np.linspace(0, 1, N)
    t2 = t1**2
    X1 = np.zeros((N, 2))
    X1[:, 0] = np.cos(4*np.pi*t1)
    X1[:, 1] = np.sin(8*np.pi*t1)
    D1 = getSSM(X1)

    X2 = np.zeros((N, 2))
    X2[:, 0] = np.cos(4*np.pi*t1)
    X2[:, 1] = np.sin(12*np.pi*t1)
    D2 = getSSM(X2)

    D = np.concatenate((D1[:, :, None], D2[:, :, None]), 2)
    images = getScatteringTransform(D)
    idx = 1
    for im in range(len(images)):
        for i, I in enumerate(images[im]):
            plt.subplot(len(images), len(images[im]), idx)
            if i > 0:
                I = poolFeatures(I, 32)
            plt.imshow(I)
            plt.colorbar()
            plt.title("Image %i: %i"%(im, i+1))
            idx += 1
    plt.show()

def plot2SameDiffHists(pDs, NWarps):
    pix = np.arange(pDs.shape[0])
    I, J = np.meshgrid(pix, pix)
    diff = np.abs(I - J)
    diff = diff[I > J]
    Ds = pDs[I > J]
    dsSame = Ds[diff < NWarps]
    dsDiff = Ds[diff >= NWarps]
    countsSame, binsSame = np.histogram(dsSame, bins='auto')
    binsSame = 0.5*(binsSame[1::] + binsSame[0:-1])
    countsSame = np.array(countsSame, dtype=np.float)
    countsSame /= np.sum(countsSame)
    countsDiff, binsDiff = np.histogram(dsDiff, bins='auto')
    binsDiff = 0.5*(binsDiff[1::] + binsDiff[0:-1])
    countsDiff = np.array(countsDiff, dtype=np.float)
    countsDiff /= np.sum(countsDiff)
    plt.plot(binsSame, countsSame)
    plt.plot(binsDiff, countsDiff)
    plt.xlabel("Distance")
    plt.ylabel("Probability Mass")
    plt.legend(["Same", "Different"])

def testScattering2WarpedFamilies(NWarps = 100, K=5):
    N = 256
    WarpDict = getWarpDictionary(N)

    t = np.linspace(0, 1, N)
    X1 = np.zeros((N, 2))
    X1[:, 0] = np.cos(4*np.pi*t)
    X1[:, 1] = np.sin(8*np.pi*t)
    X2 = np.zeros((N, 2))
    X2[:, 0] = np.cos(4*np.pi*t)
    X2[:, 1] = np.sin(12*np.pi*t)

    Ds = getSSM(X1)[:, :, None]
    for w in range(NWarps-1):
        print("Generating curve 1 %i"%w)
        t = getWarpingPath(WarpDict, K, False)
        X1Warped = getInterpolatedEuclideanTimeSeries(X1, t)
        Ds = np.concatenate((Ds, getSSM(X1Warped)[:, :, None]), 2)
    Ds = np.concatenate((Ds, getSSM(X2)[:, :, None]), 2)
    for w in range(NWarps-1):
        print("Generating curve 2 %i"%w)
        t = getWarpingPath(WarpDict, K, False)
        X2Warped = getInterpolatedEuclideanTimeSeries(X2, t)
        Ds = np.concatenate((Ds, getSSM(X2Warped)[:, :, None]), 2)

    # First compare straight euclidean distances
    plt.subplot(321)
    DsEuclidean = np.reshape(Ds, [Ds.shape[0]*Ds.shape[1], Ds.shape[2]]).T
    DsEuclidean = getSSM(DsEuclidean)
    plt.imshow(DsEuclidean, cmap='afmhot')
    plt.plot([0, NWarps*2], [NWarps, NWarps], 'C0')
    plt.plot([NWarps, NWarps], [0, NWarps*2], 'C0')
    plt.title("Euclidean Distances")
    plt.subplot(322)
    plot2SameDiffHists(DsEuclidean, NWarps)

    # Now compare scattering transform
    allimages = getScatteringTransform(Ds)
    ScatterCoeffs = []
    PoolScatterCoeffs = []
    for images in allimages:
        coeffs1 = np.array(images[0]).flatten()
        coeffs2 = np.array(images[1]).flatten()
        for j in range(1, len(images)):
            coeffs1 = np.concatenate((coeffs1, images[j].flatten()))
            coeffs2 = np.concatenate((coeffs2, poolFeatures(images[j], 32).flatten()))
        ScatterCoeffs.append(coeffs1)
        PoolScatterCoeffs.append(coeffs2)
    ScatterCoeffs = np.array(ScatterCoeffs)
    PoolScatterCoeffs = np.array(PoolScatterCoeffs)

    plt.subplot(323)
    DsScatter = getSSM(ScatterCoeffs)
    plt.imshow(DsScatter, cmap='afmhot')
    plt.plot([0, NWarps*2], [NWarps, NWarps], 'C0')
    plt.plot([NWarps, NWarps], [0, NWarps*2], 'C0')
    plt.title("Scattering Coefficients")
    plt.subplot(324)
    plot2SameDiffHists(DsScatter, NWarps)

    plt.subplot(325)
    DsScatterPooled = getSSM(PoolScatterCoeffs)
    plt.imshow(DsScatterPooled, cmap='afmhot')
    plt.plot([0, NWarps*2], [NWarps, NWarps], 'C0')
    plt.plot([NWarps, NWarps], [0, NWarps*2], 'C0')
    plt.title("Pooled Scattering Coefficients")
    plt.subplot(326)
    plot2SameDiffHists(DsScatterPooled, NWarps)

    plt.show()


if __name__ == '__main__':
    testScattering2WarpedFamilies()
    #testScatteringBasicWarp()