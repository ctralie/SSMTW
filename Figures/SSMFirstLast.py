import numpy as np
import matplotlib.pyplot as plt

def getSSM(X):
    dotX = np.reshape(np.sum(X*X, 1), (X.shape[0], 1))
    D = (dotX + dotX.T) - 2*(np.dot(X, X.T))
    D[D < 0] = 0
    D = np.sqrt(D)
    return D

if __name__ == '__main__':
    N = 50
    t1 = np.linspace(0, 1, N+1)[0:N]
    t2 = t1**2
    t1 = 2*np.pi*t1
    t1 = t1[t1 < 1.3*np.pi]
    t2 = 2*np.pi*t2
    t2 = t2[t2 < 1.3*np.pi]
    X1 = np.zeros((len(t1), 2))
    X2 = np.zeros((len(t2), 2))
    X1[:, 0] = np.cos(t1)
    X1[:, 1] = np.sin(2*t1)
    X2[:, 0] = np.cos(t2)
    X2[:, 1] = np.sin(2*t2)
    [C, S] = [np.cos(np.pi/6), np.sin(np.pi/6)]
    X2 = X2.dot(np.array([[C, -S], [S, C]]))
    X2 = X2 + np.array([[2, -2]])
    D1 = getSSM(X1)
    D2 = getSSM(X2)
    
    plt.figure(figsize=(15, 10))
    plt.subplot(231)
    plt.imshow(D1, interpolation = 'none')
    plt.title('Blue SSM')
    plt.subplot(232)
    plt.imshow(D2, interpolation = 'none')
    plt.title('Red SSM')
    plt.subplot(233)
    plt.scatter(X1[:, 0], X1[:, 1], 20, 'k', edgecolors = 'none')
    plt.hold(True)
    plt.plot(X1[:, 0], X1[:, 1], 'b')
    plt.scatter(X2[:, 0], X2[:, 1], 20, 'k', edgecolors = 'none')
    plt.plot(X2[:, 0], X2[:, 1], 'r')
    plt.axis('equal')
    plt.title('TOPCs')
    
    plt.subplot(234)
    plt.plot(D1[0, :], 'b')
    plt.hold(True)
    plt.plot(D2[0, :], 'r')
    plt.title('First Point Distance')
    
    plt.subplot(235)
    plt.plot(D1[:, -1], 'b')
    plt.plot(D2[:, -1], 'r')
    plt.title('Last Point Distance')
    
    plt.subplot(236)
    plt.plot(D1[0, :], D1[:, -1], 'b.')
    plt.hold(True)
    plt.plot(D2[0, :], D2[:, -1], 'rx')
    plt.xlabel('First Point Distance')
    plt.ylabel('Last Point Distance')
    plt.title('First Last Point TOPCs')
    plt.xlim([-0.2, np.max(D2)])
    plt.ylim([-0.2, np.max(D2)])
    
    plt.savefig('FLDTWExample.pdf', bbox_inches = 'tight')
