import os
import math
import csv
import numpy as np


def LinReg_ReadInputs(filepath):
    
    #function that reads all four of the Linear Regression csv files and outputs
    #them as such

    #Input
    #filepath : The path where all the four csv files are stored.
    #output 
    #XTrain : NxK+1 numpy matrix containing N number of K+1 dimensional training features
    XTrain = np.genfromtxt(os.path.join(filepath, 'LinReg_XTrain.csv'), delimiter=',')
    #yTrain : Nx1 numpy vector containing the actual output for the training features
    yTrain = np.genfromtxt(os.path.join(filepath, 'LinReg_yTrain.csv'), delimiter=',')
    #XTest  : nxK+1 numpy matrix containing n number of K+1 dimensional testing features
    XTest = np.genfromtxt(os.path.join(filepath, 'LinReg_XTest.csv'), delimiter=',')
    #yTest  : nx1 numpy vector containing the actual output for the testing features
    yTest = np.genfromtxt(os.path.join(filepath, 'LinReg_yTest.csv'), delimiter=',')

    Xkmin = np.min(XTrain,axis=0)
    Xkmax = np.max(XTrain,axis=0)
    XTrain = np.subtract(XTrain,Xkmin)
    XTrain = np.true_divide(XTrain,(Xkmax-Xkmin))
    XTest = np.subtract(XTest,Xkmin)
    XTest = np.true_divide(XTest,(Xkmax-Xkmin))
    # standardize features in XTrain and XTest in same formula

    N = np.size(XTrain,0)
    N_one = np.ones([N,1])
    n = np.size(XTest,0)
    n_one = np.ones([n,1])
    XTrain = np.concatenate((N_one,XTrain),axis=1)
    XTest = np.concatenate((n_one,XTest),axis=1)

    # add bias column into XTrain and XTest                     )
    return (XTrain, yTrain, XTest, yTest)
    
def LinReg_CalcObj(X, y, w):
    
    #function that outputs the value of the loss function L(w) we want to minimize.
    tmp = np.dot(X,w) - y
    n = np.size(X,0)
    #Input
    #w      : numpy weight vector of appropriate dimensions
    #AND EITHER
    #XTrain : Nx(K+1) numpy matrix containing N number of K+1 dimensional training features
    #yTrain : Nx1 numpy vector containing the actual output for the training features
    #OR
    #XTest  : nx(K+1) numpy matrix containing n number of K+1 dimensional testing features
    #yTest  : nx1 numpy vector containing the actual output for the testing features
    sqr = np.multiply(tmp,tmp)
    lossVal = np.sum(sqr)
    lossVal = np.true_divide(lossVal,n)
    #Output
    #loss   : The value of the loss function we want to minimize
    #N = np.size(XTrain,0)
    #lossVal = np.trace(w_t*X_t*X*w)-2*np.trace(y_t*X*w)
    #lossVal = np.true_divide(lossVal,N)
    return lossVal



def LinReg_CalcSG(x, y, w):
    
    #Function that calculates and returns the stochastic gradient value using a
    #particular data point (x, y).

    #Input
    #x : 1x(K+1) dimensional feature point
    #y : Actual output for the input x
    #w : (K+1)x1 dimensional weight vector 

    #Output
    #sg : gradient of the weight vector
    # w_t = np.transpose(w)
    tmp = y - np.dot(x,w)
    sg = - 2*np.multiply(tmp,x)

    return sg

def LinReg_UpdateParams(w, sg, eta):
    
    #Function which takes in your weight vector w, the stochastic gradient
    #value sg and a learning constant eta and returns an updated weight vector w.
    #print 'w3',w.shape[0],w.shape[1]
    w = w - np.multiply(sg,eta)



    #Input
    #w  : (K+1)x1 dimensional weight vector before update 
    #sg : gradient of the calculated weight vector using stochastic gradient descent
    #eta: Learning rate

    #Output
    #w  : Updated weight vector
    
    return w
    
def LinReg_SGD(XTrain, yTrain, XTest, yTest):
    
    #Stochastic Gradient Descent Algorithm function

    #Input
    #XTrain : Nx(K+1) numpy matrix containing N number of K+1 dimensional training features
    #yTrain : Nx1 numpy vector containing the actual output for the training features
    #XTest  : nx(K+1) numpy matrix containing n number of K+1 dimensional test features
    #yTest  : nx1 numpy vector containing the actual output for the test features
    num_feature = np.size(XTrain,1)
    w = np.ones([num_feature,1])

    w = np.multiply(0.5,w)
    ytrain = np.matrix(yTrain)
    ytrain = np.transpose(ytrain)

    yTest = np.matrix(yTest)
    yTest = np.transpose(yTest)
    # initial value in w are all 0.5
    N = np.size(XTrain,0)
    trainLoss = np.zeros([100,1])
    testLoss = np.zeros([100,1])
    Iter = 1
    for iter in range(100):
        # trainLoss_inside = np.zeros([N,1])
        for j in range(N):
            #eta = np.true_divide(0.5, np.sqrt((iter + 1)*(j + 1)))
            eta = np.true_divide(0.5, np.sqrt(Iter))
            Iter = Iter + 1
            XTrain_sample = XTrain[j,:]
            XTrain_sample = np.matrix(XTrain_sample)
            yTrain_sample = yTrain[j]

            sg = LinReg_CalcSG(XTrain_sample, yTrain_sample, w)
            sg = np.matrix(sg)
            sg = np.transpose(sg)
            w = LinReg_UpdateParams(w, sg, eta)
            # trainLoss_inside[j] = LinReg_CalcObj(XTrain_sample, yTrain_sample, w)
                # the updated new w
        # rainLoss[iter] = np.sum(trainLoss_inside)
        trainLoss[iter] = LinReg_CalcObj(XTrain, ytrain, w)
        trainLoss = np.array(trainLoss)
        testLoss[iter] = LinReg_CalcObj(XTest, yTest, w)
        testLoss = np.array(testLoss)
        w = np.matrix(w)
    print trainLoss
    print testLoss
    subbb = trainLoss - testLoss
    mini = np.min(subbb)
    ind = np.where(subbb==mini)
    print subbb
    print ind


    #Output
    #w    : Updated Weight vector after completing the stochastic gradient descent
    #trainLoss : vector of training loss values at each epoch
    #testLoss : vector of test loss values at each epoch
    

    
    return (w, trainLoss, testLoss)
    
def plot():     # This function's results should be returned via gradescope and will not be evaluated in autolab.
    
    return None
    

    
    