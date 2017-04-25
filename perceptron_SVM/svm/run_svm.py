import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio # loadmat function  
from sklearn import svm


'''
   This function takes data X, Y, and libsvm parameter string arg and 
   runs SVM classifier. 
'''
def run_svm(X, Y, C):
    # preprocessing Y 
    Y = Y.astype(np.float)
    Y[Y==0] = -1

    # fit the model
    clf = svm.SVC(kernel='linear', C=C)
    clf.fit(X, Y.reshape(Y.shape[0]))
    
    # get the separating hyperplane
    w = clf.coef_[0]
    a = -w[0] / w[1]
    left, right = np.min(X[:,0])-1, np.max(X[:,1])+1
    xx = np.linspace(left, right)
    yy = a * xx - (clf.intercept_[0]) / w[1]

    # get margin for positive and negative hyperplane
    m = 1.0 / np.linalg.norm(clf.coef_)
    b = -a * m
    dist = np.sqrt(m**2 + b**2)
    yy_up = yy + dist
    yy_down = yy - dist

    # compute slack variables
    Xi = np.zeros((Y.shape))
    for i in range(X.shape[0]):
        Xi[i,:] = np.float(Y[i,:])*((clf.coef_.dot(X[i,:])) + clf.intercept_)

    # plotting points, decision boundary, and margin 
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    f.suptitle("SVM Linear Classifier", fontsize=15)
    boundary = ax1.plot(xx, yy, 'k-')
    margin_neg = ax1.plot(xx, yy_down, '--', c='blue')
    margin_pos = ax1.plot(xx, yy_up, '--', c='red')
    ax1.axis('tight')
    
    pos = ax1.scatter(X[:, 0][(Y==1).ravel()], X[:, 1][(Y==1).ravel()], c='red', cmap=plt.cm.Paired, label='Positive')
    neg = ax1.scatter(X[:, 0][(Y==-1).ravel()], X[:, 1][(Y==-1).ravel()], c='blue', cmap=plt.cm.Paired, label='Negative')
    ax1.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
                s=200, facecolors='none', edgecolors='green', label = 'Support Vector')
    ax1.legend(loc='lower right')

    # plotting decision boundary, margins, and points based on slack variable values
    threshold = 0.01
    idx =(Xi<1 -threshold).ravel()
    idx2 = (Xi> 1+threshold).ravel()
    idx3 = (abs(Xi-1)<= threshold).ravel()
    
    ax2.scatter(X[idx,0], X[idx,1], color='red')
    ax2.scatter(X[idx2,0], X[idx2,1], color='black' )
    ax2.scatter(X[idx3,0], X[idx3,1], color='blue' )
    ax2.plot(xx, yy, 'k-', label='Decision Boundary')
    margin_neg2 = ax2.plot(xx, yy_down, '--', c='blue', label='Negative Hyperplane')
    margin_pos2 = ax2.plot(xx, yy_up, '--', c='red', label='Positive Hyperplane')
    ax2.axis('tight')
    ax2.legend(loc='lower right')
    plt.show()

    return clf

'''
   This function takes path to .matfile and returns a tuple of a feature matrix X 
   and a label vector Y. 
'''
def get_data(path):
    data = sio.loadmat(path)
    X, Y = data['X'], data['Y']
    return (X,Y)
