import numpy as np
import matplotlib.pyplot as plt

def backtrace(backpointers, node, involved):
    optimal = False
    for P in backpointers[node]:
        if backtrace(backpointers, (P[0], P[1]), involved):
            P[2] = True
            optimal = True
            involved[node[0], node[1]] = 1
    if node[0] == 0 and node[1] == 0:
        return True #Reached the beginning
    return optimal

def LevDist(a, b):
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
    involved = np.zeros((M+1, N+1))
    backtrace(backpointers, (M, N), involved) #Recursive backtrace from the end
    return (D, backpointers)

def DTW(X, Y, distfn, initialCost = 0):
    """
    Implements dynamic time warping
    :param X: An M-length time-ordered point cloud
    :param Y: An N-length time ordered point cloud
    :param distfn: A function to compute distances between points
    :param initalCost: A starting cost that's inherited at the
        beginning (can be useful when enforcing constraints)
    """
    M = X.shape[0]
    N = Y.shape[0]
    CSM = np.zeros((M, N))
    for i in range(M):
        for j in range(N):
            CSM[i, j] = distfn(X[i, :], Y[j, :])

    backpointers = {}
    for i in range(1, M+1):
        for j in range(1, N+1):
            backpointers[(i, j)] = []
    backpointers[(0, 0)] = []

    D = np.zeros((M+1, N+1))
    D[0, 0] = initialCost
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
    involved = np.zeros((M+1, N+1))
    backtrace(backpointers, (M, N), involved) #Recursive backtrace from the end
    return (D, CSM, backpointers, involved)

def constrainedDTW(X, Y, distfn, ci, cj):
    print "Constraint: (%i, %i)"%(ci, cj)
    M = X.shape[0]
    N = Y.shape[0]
    CSM = np.zeros((M, N))
    for i in range(M):
        for j in range(N):
            CSM[i, j] = distfn(X[i, :], Y[j, :])
    (D1, _, _, involved1) = DTW(X[0:ci+1, :], Y[0:cj+1, :], distfn)
    (D2, _, _, involved2) = DTW(X[ci::, :], Y[cj::, :], distfn, D1[-1, -1])
    involved = np.zeros((M+1, N+1))
    involved[0:D1.shape[0], 0:D1.shape[1]] = involved1
    involved[D1.shape[0]-1::, D1.shape[1]-1::] = involved2[1::, 1::]
    D = np.inf*np.ones((M+1, N+1))
    D[0:D1.shape[0], 0:D1.shape[1]] = D1
    D[D1.shape[0]-1::, D1.shape[1]-1::] = D2[1::, 1::]
    return (D, CSM, None, involved)

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
    t2 = np.sqrt(t1)
    t1 = t1**2

    X = np.zeros((len(t1), 2))
    X[:, 0] = t1
    X[:, 1] = np.cos(4*np.pi*t1) + t1
    Y = np.zeros((len(t2), 2))
    Y[:, 0] = t2
    Y[:, 1] = np.cos(4*np.pi*t2) + t2 + 0.5

    constraint = [20, 4]
    (D, CSM, backpointers, involved) = constrainedDTW(X, Y, lambda x,y: np.sqrt(np.sum((x-y)**2)), constraint[0], constraint[1])
    involved = involved[1::, 1::]

    plt.figure(figsize=(12, 12))
    plt.subplot2grid((2, 2), (0, 0), colspan=2)
    plt.scatter(X[:, 0], X[:, 1], 20, 'r')
    plt.hold(True)
    plt.scatter(Y[:, 0], Y[:, 1], 20, 'b')
    plt.plot(X[:, 0], X[:, 1], 'r')
    plt.plot(Y[:, 0], Y[:, 1], 'b')
    [J, I] = np.meshgrid(np.arange(involved.shape[1]), np.arange(involved.shape[0]))
    J = J[involved == 1]
    I = I[involved == 1]
    for i in range(len(J)):
        plt.plot([X[I[i], 0], Y[J[i], 0]], [X[I[i], 1], Y[J[i], 1]], 'k')
    [i, j] = [constraint[0], constraint[1]]
    plt.plot([X[i, 0], Y[j, 0]], [X[i, 1], Y[j, 1]], 'r', linewidth=4.0)
    plt.axis('off')
    plt.title("Curves, Cost = %.3g"%D[-1, -1])

    plt.subplot(223)
    plt.imshow(CSM, interpolation = 'nearest', cmap=plt.get_cmap('afmhot'), aspect = 'auto')
    plt.hold(True)
    plt.plot(J, I, '.')
    plt.scatter(constraint[1], constraint[0], 100, 'r')
    plt.xlim([-1, D.shape[1]])
    plt.ylim([D.shape[0], -1])
    plt.xlabel("Blue Curve")
    plt.ylabel("Red Curve")
    plt.title('Cross-Similarity Matrix')

    plt.subplot(224)
    plt.imshow(D[1::, 1::], interpolation = 'nearest', cmap=plt.get_cmap('afmhot'), aspect = 'auto')
    plt.plot(J, I, '.')
    plt.xlim([-1, D.shape[1]])
    plt.ylim([D.shape[0], -1])
    plt.xlabel("Blue Curve")
    plt.ylabel("Red Curve")
    plt.title("Dynamic Programming Matrix")

    #plt.show()
    plt.savefig("DTWExample_%i_%i.svg"%(constraint[0], constraint[1]), bbox_inches='tight')

if __name__ == '__main__':
    #LevenshteinExample()
    DTWExample()
