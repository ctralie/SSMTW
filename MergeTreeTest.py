"""
A test showing basic SSM-based time warping between two pinched
ellipses which have been re-paramterized/rotated/translated
"""
import time
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from Alignment import Alignments, AlignmentTools, DTWGPU, SyntheticCurves, AllTechniques

DTWGPU.initParallelAlgorithms()

#Setup curves
N = 210
np.random.seed(N)
t1 = np.linspace(0, 1, N)
t2 = t1**2
X1 = SyntheticCurves.getPinchedCircle(t1)
X2 = SyntheticCurves.getPinchedCircle(t2)
X2 = SyntheticCurves.applyRandomRigidTransformation(X2)


#Align curves with Merge Tree Based IBDTW
tic = time.time()
D1 = AlignmentTools.getSSM(X1)
D2 = AlignmentTools.getSSM(X2)
(D1, D2) = AlignmentTools.matchSSMDist(D1, D2)
res = Alignments.doIBDTWMergeTrees(D1, D2, 0.01)
D = res['D']
(DAll, CSSM, backpointers, path) = Alignments.DTWCSM(D)
toc = time.time()
print("Elapsed Time: %.3g"%(time.time() - tic))
sio.savemat("path.mat", {"path":path})


#Get CSWM for ordinary DTW
CSWM = DTWGPU.doIBDTWGPU(D1, D2, True)


##Plot alignment results
#First plot unaligned
plt.figure(figsize=(10, 10))
plt.subplot(221)
plt.scatter(X1[:, 0], X1[:, 1], 20, np.arange(N), cmap = 'Spectral', edgecolor='none')
plt.scatter(X2[:, 0], X2[:, 1], 20, np.arange(N), cmap = 'Spectral', edgecolor='none')
plt.axis('equal')
plt.title("Before Alignment")


#Now come up with correspondences from warping path
plt.subplot(222)
projPath = AlignmentTools.projectPath(path, N, N)
res = AlignmentTools.getProjectedPathParam(projPath)
plt.scatter(X1[:, 0], X1[:, 1], 20, c=res['C1'], edgecolor='none')
plt.scatter(X2[:, 0], X2[:, 1], 20, c=res['C2'], edgecolor='none')
plt.axis('equal')
plt.title("After Alignment")

plt.subplot(223)
plt.imshow(CSWM, cmap = 'afmhot', interpolation = 'nearest')
plt.title("DTW CSWM")

plt.subplot(224)
plt.imshow(D, cmap = 'afmhot', interpolation = 'nearest')
plt.scatter(path[:, 1], path[:, 0], 5, 'm', edgecolor = 'none')
plt.title("Merge Tree CSWM")

plt.savefig("IBDTWMergeTree.svg", bbox_inches = 'tight')