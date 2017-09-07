import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import _SequenceAlignment as SAC
from AlignmentTools import *
import time

def Backtrace(backpointers, node, path):
    optimal = False
    for P in backpointers[node]:
        if Backtrace(backpointers, (P[0], P[1]), path):
            P[2] = True
            optimal = True
            path.append([node[0]-1, node[1]-1])
    if node[0] == 0 and node[1] == 0:
        return True #Reached the beginning
    return optimal

def LevDist(a, b):
    """
    Compute the Levenshtein distance between two strings
    :param a: First string of length M
    :param b: Second string of length N
    :returns ((M+1)x(N+1) dynamic programming matrix, optimal path)
    """
    #Third element in backpointers stores whether this is part
    #of an optimal path
    M = len(a)
    N = len(b)
    D = np.zeros((M+1, N+1))
    D[:, 0] = np.arange(M+1)
    D[0, :] = np.arange(N+1)
    backpointers = {}
    for i in range(0, M+1):
        for j in range(0, N+1):
            backpointers[(i, j)] = []
    for i in range(0, M):
        backpointers[(i+1, 0)].append([i, 0, False])
    for j in range(0, N):
        backpointers[(0, j+1)].append([0, j, False])
    for i in range(1, M+1):
        for j in range(1, N+1):
            delt = 1
            if a[i-1] == b[j-1]:
                delt = 0
            dul = delt + D[i-1, j-1]
            dl = 1 + D[i, j-1]
            du = 1 + D[i-1, j]
            D[i, j] = min(min(dul, dl), du)
            if dul == D[i, j]:
                backpointers[(i, j)].append([i-1, j-1, False])
            if dl == D[i, j]:
                backpointers[(i, j)].append([i, j-1, False])
            if du == D[i, j]:
                backpointers[(i, j)].append([i-1, j, False])
    path = []
    Backtrace(backpointers, (M, N), path) #Recursive backtrace from the end
    path = np.array(path)
    return (D, path)


def DTWCSM(CSM):
    #TODO: Update this code to be more like Smith Waterman code
    """
    Perform dynamic time warping on a cros-similarity matrix
    :param CSM: An MxN cross-similarity matrix
    :return ((M+1)x(N+1) dynamic programming matrix, CSM, backpointers,
            optimal path)
    """
    M = CSM.shape[0]
    N = CSM.shape[1]
    backpointers = {}
    for i in range(1, M+1):
        for j in range(1, N+1):
            backpointers[(i, j)] = []
    backpointers[(0, 0)] = []

    D = np.zeros((M+1, N+1))
    D[0, 0] = 0
    D[1::, 0] = np.inf
    D[0, 1::] = np.inf
    for i in range(1, M+1):
        for j in range(1, N+1):
            d = CSM[i-1, j-1]
            dul = d + D[i-1, j-1]
            dl = d + D[i, j-1]
            du = d + D[i-1, j]
            D[i, j] = min(min(dul, dl), du)
            if dul == D[i, j]:
                backpointers[(i, j)].append([i-1, j-1, False])
            if dl == D[i, j]:
                backpointers[(i, j)].append([i, j-1, False])
            if du == D[i, j]:
                backpointers[(i, j)].append([i-1, j, False])
    path = []
    Backtrace(backpointers, (M, N), path) #Recursive backtrace from the end
    path = np.array(path)
    return (D, CSM, backpointers, path)

def DTW(X, Y, distfn):
    """
    Implements dynamic time warping
    :param X: An M-length time-ordered point cloud
    :param Y: An N-length time ordered point cloud
    :param distfn: A function to compute distances between points
    """
    M = X.shape[0]
    N = Y.shape[0]
    CSM = np.zeros((M, N))
    for i in range(M):
        for j in range(N):
            CSM[i, j] = distfn(X[i, :], Y[j, :])
    return DTWCSM(CSM)


def DTWConstrained(X, Y, distfn, ci, cj):
    """
    Perform constrained dynamic time warping between two TOPCs
    :param X: Mxd matrix
    :param Y: Nxd matrix
    :param distfn: Function for computing distances betwen points
    :param ci: Index in X that must be matched to cj
    :param cj: Index in Y that must be matched to ci
    :returns (Dynamic Programming Matrix, Cross-Similarity matrix,
                None, optimal path)
    """
    print "Constraint: (%i, %i)"%(ci, cj)
    M = X.shape[0]
    N = Y.shape[0]
    CSM = np.zeros((M, N))
    for i in range(M):
        for j in range(N):
            CSM[i, j] = distfn(X[i, :], Y[j, :])
    (D1, _, _, path1) = DTW(X[0:ci+1, :], Y[0:cj+1, :], distfn)
    (D2, _, _, path2) = DTW(X[ci::, :], Y[cj::, :], distfn)
    D2 = D2 - CSM[ci, cj] + D1[-1, -1]
    D = np.inf*np.ones((M+1, N+1))
    D[0:D1.shape[0], 0:D1.shape[1]] = D1
    D[D1.shape[0]-1::, D1.shape[1]-1::] = D2[1::, 1::]
    path2 += [ci, cj]
    path = np.concatenate((path1, path2), 0)
    return (D, CSM, None, path)

def writeChar(fout, i, j, c):
    fout.write("\\node at (%g, %g) {%s};\n"%(j+0.5, i+0.5, c))

def drawPointers(fout, backpointers, M, N):
    for idx in backpointers:
        for P in backpointers[idx]:
            color = 'black'
            if P[2]:
                color = 'red'
            s = [idx[1]+1.4, M-idx[0]+0.8]
            if idx[0]-P[0] == 0: #Left arrow
                fout.write("\\draw [thick, ->, %s] (%g, %g) -- (%g, %g);\n"%(color, s[0], s[1], P[1]+1.6, s[1]))
            elif idx[1]-P[1] == 0: #Up arrow
                fout.write("\\draw [thick, ->, %s] (%g, %g) -- (%g, %g);\n"%(color, s[0], s[1], s[0], M-P[0]+0.2))
            else: #Diagonal Arrow
                fout.write("\\draw [thick, ->, %s] (%g, %g) -- (%g, %g);\n"%(color, s[0], s[1], P[1]+1.6, M-P[0]+0.2))

def doIBDTW(SSMA, SSMB):
    """
    Do partial isometry blind dynamic time warping between two
    self-similarity matrices
    :param SSMA: MXM self-similarity matrix
    :param SSMB: NxN self-similarity matrix
    :returns D: MxN cross-similarity matrix
    """
    M = SSMA.shape[0]
    N = SSMB.shape[0]
    D = np.zeros((M, N))
    for i in range(M):
        print "Finished row %i of %i"%(i, M)
        for j in range(N):
            row = SSMA[i, :]
            col = SSMB[:, j]
            CSM = row[:, None] - col[None, :]
            CSM = np.abs(CSM)
            D[i, j] = SAC.DTWConstrained(CSM, i, j)
    return D

def SMWat(CSM, matchFunction, hvPenalty = -0.2, backtrace = False, backidx = [], animate = False):
    """
    Implicit Smith Waterman alignment on a real-valued cross-similarity matrix
    :param CSM: A binary N x M cross-similarity matrix
    :param matchFunction: A function that scores matching/mismatching
    :param hvPenalty: The amount by which to penalize horizontal/vertical moves
    :returns (Distance (scalar), (N+1)x(M+1) dynamic programming matrix,
              optimal subsequence alignment path)
    """
    N = CSM.shape[0]+1
    M = CSM.shape[1]+1
    D = np.zeros((N, M))
    if backtrace:
        B = np.zeros((N, M), dtype = np.int64) #Backpointer indices
        pointers = [[-1, 0], [0, -1], [-1, -1], None] #Backpointer directions
    maxD = 0
    maxidx = [0, 0]
    for i in range(1, N):
        for j in range(1, M):
            d1 = D[i-1, j]
            d2 = D[i, j-1]
            d3 = D[i-1, j-1]
            cost = matchFunction(CSM[i-1, j-1])
            arr = [d1+cost+hvPenalty, d2+cost+hvPenalty, d3+cost, 0.0]
            D[i, j] = np.max(arr)
            if backtrace:
                B[i, j] = np.argmax(arr)
            if (D[i, j] > maxD):
                maxD = D[i, j]
                if backtrace:
                    maxidx = [i, j]

    res = {'maxD':maxD, 'D':D}
    #Backtrace starting at the largest index
    if backtrace:
        res['B'] = B
        res['maxidx'] = maxidx
        path = [maxidx]
        if len(backidx) > 0:
            path = [backidx]
        idx = path[-1]
        while B[idx[0], idx[1]] < 3:
            i = B[idx[0], idx[1]]
            idx = [idx[0]+pointers[i][0], idx[1] + pointers[i][1]]
            if idx[0] < 1 or idx[1] < 1:
                break
            path.append(idx)
            if animate:
                plt.clf()
                plt.imshow(CSM, cmap = 'afmhot', interpolation = 'nearest')
                P = np.array(path) - 1
                plt.scatter(P[:, 1], P[:, 0], 5, 'b', edgecolor = 'none')
                plt.scatter(P[-1, 1], P[-1, 0], 20, 'r', edgecolor = 'none')
                plt.axis('off')
                plt.savefig("BackTrace%i.png"%P.shape[0], bbox_inches = 'tight')
        res['path'] = path

    return res

def doIBSMWat(SSMA, SSMB, matchfn, hvPenalty = -0.3):
    #matchfn = lambda(x): {0:-3, 1:2}[x]
    #hvPenalty = -0.3

    M = SSMA.shape[0]
    N = SSMB.shape[0]
    D = np.zeros((M, N))
    for i in range(M):
        print "Finished row %i of %i"%(i, M)
        for j in range(N):
            row = SSMA[i, :]
            col = SSMB[:, j]
            CSM = row[:, None] - col[None, :]
            CSM = np.abs(CSM)
            D[i, j] = SAC.DTWConstrained(CSM, i, j)
    return D

def SMWatConstrained(CSM, ci, cj, matchFunction, hvPenalty = -0.3, backtrace = False):
    """
    Implicit Smith Waterman alignment on a binary cross-similarity matrix
    with constraints
    :param CSM: A binary N x M cross-similarity matrix
    :param ci: The index along the first sequence that must be matched to cj
    :param cj: The index along the second sequence that must be matched to ci
    :param matchFunction: A function that scores matching/mismatching
    :param hvPenalty: The amount by which to penalize horizontal/vertical moves
    :returns (Distance (scalar), (N+1)x(M+1) dynamic programming matrix)
    """
    res1 = SMWat(CSM[0:ci+1, 0:cj+1], matchFunction, hvPenalty, backtrace = backtrace, backidx = [ci+1, cj+1])
    CSM2 = np.fliplr(np.flipud(CSM[ci::, cj::]))
    res2 = SMWat(CSM2, matchFunction, hvPenalty, backtrace = backtrace, backidx = [CSM2.shape[0], CSM2.shape[1]])
    res = {'score':res1['D'][-1, -1] + res2['D'][-1, -1]}
    res['D1'] = res1['D']
    res['D2'] = res2['D']
    if backtrace:
        path2 = [[ci+1+(CSM2.shape[0]+1-x), cj+1+(CSM2.shape[1]+1-y)] for [x, y] in res2['path']]
        res['path'] = res1['path'] + path2
    #res['score'] = res2['D'][-1, -1]
    return res

def SMWatExampleBinary():
    Kappa = 0.05
    np.random.seed(100)
    t = np.linspace(0, 1, 300)
    t1 = t
    X1 = 0.3*np.random.randn(400, 2)
    X1[50:50+len(t1), 0] = np.cos(2*np.pi*t1)
    X1[50:50+len(t1), 1] = np.sin(4*np.pi*t1)
    t2 = t**2
    X2 = 0.3*np.random.randn(350, 2)
    X2[0:len(t2), 0] = np.cos(2*np.pi*t2)
    X2[0:len(t2), 1] = np.sin(4*np.pi*t2)
    CSM = getCSM(X1, X2)
    CSM = CSMToBinaryMutual(CSM, Kappa)

    plt.figure(figsize=(8, 8))
    [ci, cj] = [100, 100]

    matchfn = lambda x: {0:-3, 1:2}[x]
    res = SMWatConstrained(CSM, ci, cj, matchfn, hvPenalty = -1.8, backtrace = True)
    path = np.array(res['path'])
    plt.imshow(CSM, cmap = 'afmhot', interpolation = 'none')
    plt.scatter(path[:, 1], path[:, 0], 5, edgecolor = 'none')
    plt.scatter([cj], [ci], 20, 'r', edgecolor = 'none')
    plt.title("Score = %g"%res['score'])
    plt.savefig("Constrained_%i_%i.svg"%(ci, cj), bbox_inches = 'tight')


def SMWatExampleSSMRows():
    Kappa = 0.05
    np.random.seed(100)
    t = np.linspace(0, 1, 150)
    t1 = t
    X1 = 0.3*np.random.randn(200, 2)
    X1[50:50+len(t1), 0] = np.cos(2*np.pi*t1)
    X1[50:50+len(t1), 1] = np.sin(4*np.pi*t1)
    t2 = t**2
    X2 = 0.3*np.random.randn(180, 2)
    X2[0:len(t2), 0] = np.cos(2*np.pi*t2)
    X2[0:len(t2), 1] = np.sin(4*np.pi*t2)

    SSMA = get2DRankSSM(getSSM(X1))
    SSMB = get2DRankSSM(getSSM(X2))
    M = SSMA.shape[0]
    N = SSMB.shape[0]

    [ci, cj] = [60, 60]
    row1 = SSMA[ci, :]
    row2 = SSMB[cj, :]
    CSM = np.abs(row1[:, None] - row2[None, :])
    matchfn = lambda x: np.exp(-x/(0.3**2))-0.6
    D = matchfn(CSM)
    hvPenalty = -0.3

    tic = time.time()
    res = SMWatConstrained(CSM, ci, cj, matchfn, hvPenalty, backtrace = True)
    print("Elapsed Time Python: ", time.time() - tic)
    tic = time.time()
    d = SAC.SMWatConstrained(D, ci, cj, hvPenalty)
    print("Elapsed Time C: ", time.time() - tic)
    print("Python Answer: %g\nC Answer: %g"%(res['score'], d))

    #"""
    plt.figure(figsize=(12, 12))
    plt.subplot(221)
    plt.imshow(SSMA, cmap = 'afmhot', interpolation = 'nearest')
    plt.subplot(222)
    plt.imshow(SSMB, cmap = 'afmhot', interpolation = 'nearest')
    plt.subplot(223)
    plt.plot(row1, 'b')
    plt.plot(row2, 'r')
    plt.subplot(224)
    plt.imshow(D, cmap = 'afmhot', interpolation = 'nearest')
    P = np.array(res['path'])
    plt.scatter(P[:, 1], P[:, 0], 5, 'b', edgecolor = 'none')
    plt.scatter([cj], [ci], 50, 'r', edgecolor = 'none')
    plt.title("Score = %g"%res['score'])
    plt.savefig("Constrained_%i_%i.svg"%(ci, cj), bbox_inches = 'tight')
    #"""


    """
    CSM = np.zeros((M, N))
    for i in range(M):
        print("Doing row %i of %i"%(i, M))
        for j in range(N):
            row1 = SSMA[i, :]
            row2 = SSMB[j, :]
            C = np.abs(row1[:, None] - row2[None, :])
            C = matchfn(C)
            CSM[i, j] = SAC.SMWatConstrained(C, i, j, hvPenalty)
        sio.savemat("CSM.mat", {"CSM":CSM})
    """


def LevenshteinExample():
    #Make Levenshtein Example
    a = "fools"
    b = "school"
    M = len(a)
    N = len(b)
    (D, backpointers) = LevDist(a, b)
    fout = open("levfig.tex", "w")
    fout.write("\\begin{tikzpicture}")
    fout.write("\\draw [help lines] (0, 0) grid (%i,%i);\n"%(N+2, M+2))

    writeChar(fout, M, 0, '\\_')
    for i in range(M):
        writeChar(fout, M-(i+1), 0, a[i])

    writeChar(fout, M+1, 1, '\\_')
    for j in range(N):
        writeChar(fout, M+1, j+2, b[j])

    for i in range(M+1):
        for j in range(N+1):
            if i == M and j == N:
                continue
            writeChar(fout, M-i, j+1, int(D[i, j]))
    writeChar(fout, 0, N+1, int(D[-1, -1]))

    drawPointers(fout, backpointers, M, N)
    fout.write("\\end{tikzpicture}")
    fout.close()

def DTWExample():
    #Make dynamic time warping example
    np.random.seed(100)
    t1 = np.linspace(0, 1, 50)
    t1 = t1
    t2 = np.linspace(0, 1, 60)
    #t2 = np.sqrt(t2)
    #t1 = t1**2

    X = np.zeros((len(t1), 2))
    X[:, 0] = t1
    X[:, 1] = np.cos(4*np.pi*t1) + t1
    Y = np.zeros((len(t2), 2))
    Y[:, 0] = t2
    Y[:, 1] = np.cos(4*np.pi*t2) + t2 + 0.5

    (D, CSM, backpointers, path) = DTW(X, Y, lambda x,y: np.sqrt(np.sum((x-y)**2)))
    print "Unconstrained DTW: ", D[-1, -1]

    constraint = [20, 30]
    (D, CSM, backpointers, path) = DTWConstrained(X, Y, lambda x,y: np.sqrt(np.sum((x-y)**2)), constraint[0], constraint[1])
    print "Cost Python: ", D[-1, -1]
    tic = time.time()
    print "Cost C: ", SAC.DTWConstrained(CSM, constraint[0], constraint[1])
    toc = time.time()
    print "Elapsed Time: ", toc-tic

    plt.figure(figsize=(12, 12))
    plt.subplot2grid((2, 2), (0, 0), colspan=2)
    plt.scatter(X[:, 0], X[:, 1], 20, 'r')
    plt.hold(True)
    plt.scatter(Y[:, 0], Y[:, 1], 20, 'b')
    plt.plot(X[:, 0], X[:, 1], 'r')
    plt.plot(Y[:, 0], Y[:, 1], 'b')
    for k in range(path.shape[0]):
        [i, j] = path[k, :]
        plt.plot([X[i, 0], Y[j, 0]], [X[i, 1], Y[j, 1]], 'k')
    [i, j] = [constraint[0], constraint[1]]
    plt.scatter([X[i, 0], Y[j, 0]], [X[i, 1], Y[j, 1]], 100, color = '#00ff00', edgecolor = 'k')
    plt.axis('off')
    plt.title("Curves, Cost = %.3g"%D[-1, -1])

    plt.subplot(223)
    plt.imshow(CSM, interpolation = 'nearest', cmap=plt.get_cmap('afmhot'), aspect = 'auto')
    plt.hold(True)
    plt.plot(path[:, 1], path[:, 0], '.')
    plt.scatter(constraint[1], constraint[0], 100, color = '#00ff00', edgecolor = 'k')
    plt.xlim([-1, D.shape[1]])
    plt.ylim([D.shape[0], -1])
    plt.xlabel("Blue Curve")
    plt.ylabel("Red Curve")
    plt.title('Cross-Similarity Matrix')

    plt.subplot(224)
    plt.imshow(D[1::, 1::], interpolation = 'nearest', cmap=plt.get_cmap('afmhot'), aspect = 'auto')
    plt.plot(path[:, 1], path[:, 0], '.')
    plt.xlim([-1, D.shape[1]])
    plt.ylim([D.shape[0], -1])
    plt.xlabel("Blue Curve")
    plt.ylabel("Red Curve")
    plt.title("Dynamic Programming Matrix")

    plt.savefig("DTWExample_%i_%i.svg"%(constraint[0], constraint[1]), bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    #LevenshteinExample()
    #DTWExample()
    #SMWatExampleBinary()
    SMWatExampleSSMRows()
