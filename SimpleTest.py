"""
A test showing basic SSM-based time warping between two pinched
ellipses which have been re-paramterized/rotated/translated
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from Alignment import Alignments, AlignmentTools, DTWGPU, SyntheticCurves, AllTechniques

useGPU = False

if useGPU:
    #Must compile CUDA kernels before doing anything
    DTWGPU.initParallelAlgorithms()

#Setup curves
N = 210
np.random.seed(N)
t1 = np.linspace(0, 1, N)
t2 = t1**2
X1 = SyntheticCurves.getPinchedCircle(t1)
X2 = SyntheticCurves.getPinchedCircle(t2)
X2 = SyntheticCurves.applyRandomRigidTransformation(X2)


#Align curves with IBDTW
tic = time.time()
(path, pathN) = AllTechniques.getIBDTWAlignment(X1, X2, useGPU = useGPU, doPlot = True)
print("Elapsed Time: %g"%(time.time() - tic))
plt.show()


##Plot alignment results
#First plot unaligned
plt.subplot(121)
plt.scatter(X1[:, 0], X1[:, 1], 20, np.arange(N), cmap = 'Spectral', edgecolor='none')
plt.scatter(X2[:, 0], X2[:, 1], 20, np.arange(N), cmap = 'Spectral', edgecolor='none')
plt.axis('equal')
plt.title("Before Alignment")
#Now come up with correspondences from warping path
plt.subplot(122)
projPath = AlignmentTools.projectPath(path, N, N)
res = AlignmentTools.getProjectedPathParam(projPath)
plt.scatter(X1[:, 0], X1[:, 1], 20, c=res['C1'], edgecolor='none')
plt.scatter(X2[:, 0], X2[:, 1], 20, c=res['C2'], edgecolor='none')
plt.axis('equal')
plt.title("After Alignment")
plt.show()
