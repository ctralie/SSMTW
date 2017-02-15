import numpy as np
import matplotlib.pyplot as plt

def getWarpingPathsRec(M, N, path, AllPaths):
    PLast = path[-1]
    if PLast[0] >= M or PLast[1] >= N:
        return #Out of bounds
    elif PLast[0] == M-1 and PLast[1] == N-1:
        AllPaths.append(np.array(path))
        return
    getWarpingPathsRec(M, N, path + [[PLast[0]+1, PLast[1]]], AllPaths)
    getWarpingPathsRec(M, N, path + [[PLast[0], PLast[1]+1]], AllPaths)
    getWarpingPathsRec(M, N, path + [[PLast[0]+1, PLast[1]+1]], AllPaths)

def getWarpingPaths(M, N):
    AllPaths = []
    getWarpingPathsRec(M, N, [[0, 0]], AllPaths)
    return AllPaths
    

def plotWarpingPath(P, M, N):
    plt.subplot(121)
    plt.scatter(np.arange(M), np.array([0]*M))
    plt.hold(True)
    plt.scatter(np.arange(N), np.array([M]*N))
    for i in range(P.shape[0]):
        plt.plot(P[i, :], [0, M], 'b')
    plt.title("%i Correspondences"%P.shape[0])
    plt.axis('off')
    
    plt.subplot(122)
    plt.scatter(P[:, 1], P[:, 0])
    plt.ylim([-1, M])
    plt.xlim([-1, N])
    plt.title('Warping Path')
    plt.axis('off')

#Score all isometry blind warping paths between X and Y
def getIBDTWs(X, Y, distfn):
    M = X.shape[0]
    N = Y.shape[0]
    AllPaths = getWarpingPaths(M, N)
    
    P = AllPaths[1]
    

def countSums(M, N):
    AllPaths = getWarpingPaths(M, N)
    #First compute K1
    K1 = {}
    for gamma in AllPaths:
        #print "yee"

if __name__ == '__main__':
    #countSums(3, 3)
    paths = getWarpingPaths(3, 3)
    print paths

if __name__ == '__main__2':
    #Do an example of a triangle and a square each under L1
    X = np.array([[-5, 1], [-5, 0], [-4, 0]])
    Y = np.array([[0, 1], [0, 0], [1, 0], [1, 1]])
    getIBDTWs(X, Y, lambda x, y: np.sum(np.abs(x - y)))

