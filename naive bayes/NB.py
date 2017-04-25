import math
import numpy as np
import numpy.matlib


# The logProd function takes a vector of numbers in logspace
# (i.e., x[i] = log p[i]) and returns the product of those numbers in logspace
# (i.e., logProd(x) = log(product_i p[i]))
def logProd(x):
    ## Inputs ##
    # x - 1D numpy ndarray
    ## Outputs ##
    # log_product - float
    sum = np.sum(x)
    log_product = np.float64(sum)
    return log_product


# The NB_XGivenY function takes a training set XTrain and yTrain and
# Beta parameters alpha and beta, then returns a matrix containing
# MAP estimates of theta_yw for all words w and class labels y
def NB_XGivenY(XTrain, yTrain, alpha, beta):
    ## Inputs ##
    # XTrain - (n by V) numpy ndarray
    # yTrain - 1D numpy ndarray of length n
    # alpha - float
    # beta - float

    ## Outputs ##
    # D - (2 by V) numpy ndarray
    v = np.size(XTrain, 1)
    num_y1 = sum(yTrain)
    num_y0 = len(yTrain) - num_y1
    # number of zeros and ones in yTtrain

    theta_0w = np.zeros(v)
    theta_1w = np.zeros(v)
    # put zero to all elements in D

    index_yequal0 = np.where(yTrain == 0)
    index_yequal1 = np.where(yTrain == 1)
    # index for y equal to zero and one

    for j in range(v):
        num_0w = np.sum(XTrain[index_yequal0, j])
        theta_0w[j] = np.true_divide(((alpha - 1) + num_0w), ((alpha - 1) + (beta - 1) + num_y0))
        num_1w = np.sum(XTrain[index_yequal1, j])
        theta_1w[j] = np.true_divide(((alpha - 1) + num_1w), ((alpha - 1) + (beta - 1) + num_y1))
        # put theta_map into D

    theta_0w = numpy.matlib.repmat(theta_0w, 1, 1)
    theta_1w = numpy.matlib.repmat(theta_1w, 1, 1)

    D = np.concatenate((theta_0w, theta_1w), axis=0)


    return D


# The NB_YPrior function takes a set of training labels yTrain and
# returns the prior probability for class label 0
def NB_YPrior(yTrain):
    ## Inputs ##
    # yTrain - 1D numpy ndarray of length n

    ## Outputs ##
    # p - float
    num_y0 = len(yTrain) - np.sum(yTrain)
    p = np.true_divide(num_y0, len(yTrain))
    #p = np.float64(p)
    return p


# The NB_Classify function takes a matrix of MAP estimates for theta_yw,
# the prior probability for class 0, and uses these estimates to classify
# a test set.
def NB_Classify(D, p, XTest):
    ## Inputs ##
    # D - (2 by V) numpy ndarray
    # p - float
    # XTest - (m by V) numpy ndarray

    ## Outputs ##
    # yHat - 1D numpy ndarray of length m
    m = np.size(XTest, 0)

    #for ii in range(m):
    #   XTest[ii,:]=np.true_divide(XTest[ii,:],sum(XTest[ii,:]))
	#ind=np.where(XTest==0)
    #XTest[ind]=1
    #XTest=np.log(XTest)

    yHat = np.zeros(m)

    D0=np.sum(D[0,:])
    D1=np.sum(D[1,:])
    D=np.true_divide(D,[[D0],[D1]])
    #D=np.multiply(D,0.00001)
    D=np.log(D)
    # prioreco=numpy.matlib.repmat(p,1,1)
    # prioronion=numpy.matlib.repmat((1-p),1,1)
    for i in range(m):
        #qd0 = np.add(D[0, :], XTest[i, :])
        qd0 = np.multiply(D[0, :], XTest[i, :])
        # qd0=np.concatenate((qd0, prioreco), axis=1)
        #qd0 = np.log(qd0)
        #qd1 = np.add(D[1, :], XTest[i, :])
        qd1 = np.multiply(D[1, :], XTest[i, :])
        # qd1=np.concatenate((qd1, prioronion), axis=1)
        #qd1 = np.log(qd1)

        if (logProd(qd0) + (np.log(p))) > (logProd(qd1) + (np.log(1 - p))):
            yHat[i] = 0
        else:
            yHat[i] = 1
    yHat = np.transpose(yHat)
    # yHat = np.ones(XTest.shape[0])
    return yHat


# The classificationError function takes two 1D arrays of class labels
# and returns the proportion of entries that disagree
def classificationError(yHat, yTruth):
    ## Inputs ##
    # yHat - 1D numpy ndarray of length m
    # yTruth - 1D numpy ndarray of length m

    ## Outputs ##
    # error - float
    deno = np.size(yHat, 0)
    c = (yHat == yTruth)
    nume = np.sum(c)
    error = np.true_divide(nume, deno)
    error = 1 - error
    return error
