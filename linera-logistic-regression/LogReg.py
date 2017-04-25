import os
import math
import numpy as np


def LogReg_ReadInputs(filepath):
    
    #function that reads all four of the Logistic Regression csv files and outputs
    #them as such
    #Input
    #filepath : The path where all the four csv files are stored.

    #output 
    #XTrain : Nx(K+1) numpy matrix containing N number of K+1 dimensional training features
    XTrain = np.genfromtxt(os.path.join(filepath, 'LogReg_XTrain.csv'), delimiter=',')
    #yTrain : Nx1 numpy vector containing the actual output for the training features
    yTrain = np.genfromtxt(os.path.join(filepath, 'LogReg_yTrain.csv'), delimiter=',')
    #XTest  : nx(K+1) numpy matrix containing n number of K+1 dimensional testing features
    XTest = np.genfromtxt(os.path.join(filepath, 'LogReg_XTest.csv'), delimiter=',')
    #yTest  : nx1 numpy vector containing the actual output for the testing features
    yTest = np.genfromtxt(os.path.join(filepath, 'LogReg_yTest.csv'), delimiter=',')


    N = np.size(XTrain,0)
    N_one = np.ones([N,1])
    n = np.size(XTest,0)
    n_one = np.ones([n,1])
    XTrain = np.concatenate((N_one,XTrain),axis=1)
    XTest = np.concatenate((n_one,XTest),axis=1)

    return (XTrain, yTrain, XTest, yTest)
    
def LogReg_CalcObj(X, y, w):

    
    #function that outputs the conditional log likelihood we want to maximize.
    n = np.size(X,0)
    cll = np.zeros([n,1])
    for i in range(n):
        yi = y[i,:]
        xi = X[i,:]
        u_w = -np.dot(xi,w)
        tmp = 1 + np.exp(u_w)
        hwx = np.true_divide(1,tmp)
        cll[i,:] = yi * np.log(hwx) + (1 - yi) * np.log(1 - hwx)
    cll = np.sum(cll)
    #Input
    #w      : numpy weight vector of appropriate dimensions initialized to 0.5
    #AND EITHER
    #XTrain : Nx(K+1) numpy matrix containing N number of K+1 dimensional training features
    #yTrain : Nx1 numpy vector containing the actual output for the training features
    #OR
    #XTest  : nx(K+1) numpy matrix containing n number of K+1 dimensional testing features
    #yTest  : nx1 numpy vector containing the actual output for the testing features

    #Output
    #cll   : The conditional log likelihood we want to maximize
    
    cll =  np.true_divide(cll,n)

    return cll
    
def LogReg_CalcSG(x, y, w):
    
    #Function that calculates and returns the stochastic gradient value using a
    #particular data point (x, y).

    #Input
    #x : 1x(K+1) dimensional feature point
    #y : Actual output for the input x
    #w : weight vector
    u_w = -np.dot(x,w)
    tmp = 1 + np.exp(u_w)
    h_w = np.true_divide(1,tmp)

    sg_tmp = y - h_w
    sg = np.dot(sg_tmp,x)
    #Output
    #sg : gradient of the weight vector
    

    
    return sg
        
def LogReg_UpdateParams(w, sg, eta):
    
    #Function which takes in your weight vector w, the stochastic gradient
    #value sg and a learning constant eta and returns an updated weight vector w.

    #Input
    #w  : weight vector before update 
    #sg : gradient of the calculated weight vector using stochastic gradient ascent
    #eta: Learning rate
    w = w + np.multiply(sg, eta)
    #Output
    #w  : Updated weight vector
    
    return w
    
def LogReg_PredictLabels(X, y, w):
    
    #Function that returns the value of the predicted y along with the number of
    #errors between your predictions and the true yTest values

    #Input
    #w : weight vector 
    #AND EITHER
    #XTest : nx(K+1) numpy matrix containing m number of d dimensional testing features
    #yTest : nx1 numpy vector containing the actual output for the testing features
    #OR
    #XTrain : Nx(K+1) numpy matrix containing N number of K+1 dimensional training features
    #yTrain : Nx1 numpy vector containing the actual output for the training features
    
    #Output
    #yPred : An nx1 vector of the predicted labels for yTest/yTrain
    #perMiscl : The percentage of y's misclassified

    u_w = -np.dot(X,w)
    tmp = 1 + np.exp(u_w)
    yPred = np.true_divide(1,tmp)
    ind1 = np.where(yPred > 0.5)
    ind0 = np.where(yPred <= 0.5)
    yPred[ind1] = 1
    yPred[ind0] = 0



    ind_right = (yPred==y)
    num_right = np.sum(ind_right)
    PerRight = np.true_divide(num_right,len(y))
    PerMiscl = 1 - PerRight
    return (yPred, PerMiscl)    

def LogReg_SGA(XTrain, yTrain, XTest, yTest):
    
    #Stochastic Gradient Ascent Algorithm function

    # ytrain = np.matrix(yTrain)
    # ytrain = np.transpose(ytrain)

    ytest = np.matrix(yTest)
    ytest = np.transpose(ytest)


    num_feature = np.size(XTrain, 1)
    N = np.size(XTrain,0)
    w = np.ones([num_feature, 1])
    w = np.multiply(0.5, w)
    a = np.floor((5*N)/200)
    trainPerMiscl = np.zeros([a,1])
    testPerMiscl = np.zeros([a,1])
    iter = 1
    for i in range(5):
        for j in range(N):
            eta = np.true_divide(0.5, np.sqrt(iter))
            XTrain_sample = XTrain[j, :]
            XTrain_sample = np.matrix(XTrain_sample)
            yTrain_sample = yTrain[j]
            iter = iter + 1

            sg = LogReg_CalcSG(XTrain_sample, yTrain_sample, w)
            sg = np.matrix(sg)
            sg = np.transpose(sg)
            w = LogReg_UpdateParams(w, sg, eta)
            ind = np.floor( iter / 200)
            if ind<73:
                _,trainPerMiscl[ind,:] = LogReg_PredictLabels(XTrain, yTrain, w)
                _,testPerMiscl[ind, :] = LogReg_PredictLabels(XTest, ytest, w)
    #Input
    #XTrain : Nx(K+1) numpy matrix containing N number of K+1 dimensional training features
    #yTrain : Nx1 numpy vector containing the actual output for the training features
    #XTest  : nx(K+1) numpy matrix containing n number of K+1 dimensional testing features
    #yTest  : nx1 numpy vector containing the actual output for the testing features

    #Output
    #w             : final weight vector
    #trainPerMiscl : a vector of percentages of misclassifications on your training data at every 200 gradient descent iterations
    #testPerMiscl  : a vector of percentages of misclassifications on your testing data at every 200 gradient descent iterations
    #yPred         : a vector of your predictions for yTest using your final w
    yPred,_ = LogReg_PredictLabels(XTest, yTest, w)
   # print trainPerMiscl,'trainsize',trainPerMiscl.shape[0]
   # print testPerMiscl,'testsize',testPerMiscl.shape[0]
    print trainPerMiscl
    print testPerMiscl
    return (w, trainPerMiscl, testPerMiscl, yPred)
    
def plot():     # This function's results should be returned via gradescope and will not be evaluated in autolab.
    
    return None