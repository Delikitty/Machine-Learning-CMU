import numpy as np
import math
import math
import numpy as np
import numpy.matlib


########################################################################
#######  you should maintain the  return type in starter codes   #######
########################################################################


def perceptron_predict(w, x):
    # Input:
    W = np.matrix(w)
    X = np.matrix(x)
    # X_t = np.transpose(X)
    w_t = np.transpose(W)
    z = np.dot(X, w_t)

    if z > 0:
        y_hat = 1
    else:
        y_hat = -1
    # w is the weight vector (d,),  1-d array
    #   x is feature values for test example (d,), 1-d array
    # Output:
    #   the predicted label for x, scalar -1 or 1
    return y_hat


def perceptron_train(w0, XTrain, yTrain, num_epoch):
    # Input:

    W0 = np.matrix(w0)
    w = W0
    # XTrain = np.matrix(XTrain)
    i_num = XTrain.shape[0]
    for j in range(num_epoch):
        for i in range(i_num):
            x = XTrain[i, :]
            y = yTrain[i]
            y_hat = perceptron_predict(w, x)
            if y_hat != y:
                w = w + np.multiply(y, x)
    w = np.transpose(w)
    w = np.array(w)
    w = np.reshape(w, [785, ])

    #   w0 is the initial weight vector (d,), 1-d array
    #   XTrain is feature values of training examples (n,d), 2-d array
    #   yTrain is labels of training examples (n,), 1-d array
    #   num_epoch is the number of times to go through the data, scalar
    # Output:
    #   the trained weight vector, (d,), 1-d array
    return w


def RBF_kernel(X1, X2, sigma):
    # Input:
    #   X1 is a feature matrix (n,d), 2-d array or 1-d array (d,) when n = 1
    #   X2 is a feature matrix (m,d), 2-d array or 1-d array (d,) when m = 1
    #   sigma is the parameter $\sigma$ in RBF function, scalar
    # Output:
    #   K is a kernel matrix (n,m), 2-d array

    # ----------------------------------------------------------------------------------------------
    # Special notes: numpy will automatically convert one column/row of a 2d array to 1d array
    #                which is  unexpected in the implementation
    #                make sure you always return a 2-d array even n = 1 or m = 1
    #                your implementation should work when X1, X2 are either 2d array or 1d array
    #                we provide you with some starter codes to make your life easier
    # ---------------------------------------------------------------------------------------------
    if len(X1.shape) == 2:
        n = X1.shape[0]
    else:
        n = 1
        X1 = np.reshape(X1, (1, X1.shape[0]))
    if len(X2.shape) == 2:
        m = X2.shape[0]
    else:
        m = 1
        X2 = np.reshape(X2, (1, X2.shape[0]))

    #  num_i = X1.shape[0]
    #  num_j = X2.shape[0]
    #  K = np.zeros([num_i, num_j])
    X2 = np.repeat(X2,n,axis = 0)

    #  for i in range(num_i):
    #  x1 = X1[i, :]
    #for j in range(num_j):
        # x2 = X2[j, :]

    tmp = np.multiply(np.subtract(X1,X2), np.subtract(X1, X2))

    tmp = np.true_divide(-np.sum(tmp,axis = 1), 2 * sigma * sigma)

    #  K[i, :] = np.exp(tmp)
    K = np.exp(tmp)
    K = np.reshape(K,(n,m))

    return K


def kernel_perceptron_predict(a, XTrain, yTrain, x, sigma):
    # Input:
    a = np.matrix(a)
    x = np.matrix(x)
    i_num = XTrain.shape[0]  # n
    K = RBF_kernel(XTrain, x, sigma)
    K  = np.matrix(K)
    K = np.transpose(K)
    tmp = np.multiply(a, yTrain)
    # print tmp.shape
    tmp = np.multiply(tmp, K)
    z = np.sum(tmp)
    if z > 0:
        ypred = 1
    else:
        ypred = -1
    # a is the counting vector (n,),  1-d array
    #   XTrain is feature values of training examples (n,d), 2-d array
    #   yTrain is labels of training examples (n,), 1-d array
    #   x is feature values for test example (d,), 1-d array
    #   sigma is the parameter $\sigma$ in RBF function, scalar
    # Output:
    #   the predicted label for x, scalar -1 or 1
    return ypred


def kernel_perceptron_train(a0, XTrain, yTrain, num_epoch, sigma):
    # Input:

    #A0 = np.matrix(a0)
    #a = A0
    # XTrain = np.matrix(XTrain)
    i_num = XTrain.shape[0]
    for j in range(num_epoch):
        for i in range(i_num):
            x = XTrain[i, :]
            y = yTrain[i]
            ypred = kernel_perceptron_predict(a0, XTrain, yTrain, x, sigma)
            if ypred != y:
                a0[i] = a0[i] + 1
            #else:
               # a[i] = 0

    # a = np.transpose(a)
    a0 = np.array(a0)
    #a = np.reshape(a, (600, ))
    #   a0 is the initial counting vector (n,), 1-d array
    #   XTrain is feature values of training examples (n,d), 2-d array
    #   yTrain is labels of training examples (n,), 1-d array
    #   num_epoch is the number of times to go through the data, scalar
    #   sigma is the parameter $\sigma$ in RBF function, scalar
    # Output:
    #   the trained counting vector, (n,), 1-d array
    return a0