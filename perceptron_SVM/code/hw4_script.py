import os
import csv
import numpy as np
import perceptron
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import perceptron
import math
import numpy as np
import numpy.matlib

# Point to data directory here
# By default, we are pointing to '../data/'
data_dir = os.path.join('..','data')

# Load numeric data files into numpy arrays
XTrain = np.genfromtxt(os.path.join(data_dir, 'XTrain.csv'), delimiter=',')
yTrain = np.genfromtxt(os.path.join(data_dir, 'yTrain.csv'), delimiter=',')
XTest = np.genfromtxt(os.path.join(data_dir, 'XTest.csv'), delimiter=',')
yTest = np.genfromtxt(os.path.join(data_dir, 'yTest.csv'), delimiter=',')
#yy = np.matrix(yTest)
# print yTest.shape
#print yy.shape[0],yy.shape[1]



#EX1

num_epoch = 10
w0 = np.zeros([785,])
w = perceptron.perceptron_train(w0, XTrain, yTrain, num_epoch)
y_hat = np.zeros([600,])
for i in range(600):
    y_hat[i] = perceptron.perceptron_predict(w,XTest[i,:])
ind = (yTest!=y_hat)
ind = np.sum(ind)
print ind
per = np.true_divide(ind,600)
print per


#EX2
#sigma = 1000
#num_epoch = 2
#a0 = np.zeros([600,])
#a = perceptron.kernel_perceptron_train(a0, XTrain, yTrain, num_epoch, sigma)
#ypred = np.zeros([600,])
#for i in range(600):
#    x = XTest[i,:]
#    ypred[i] = perceptron.kernel_perceptron_predict(a, XTrain, yTrain, x, sigma)
#print ypred.shape
#ind = (yTest!=ypred)
#ind = np.sum(ind)
#print ind
#per = np.true_divide(ind,600)
#print per






#a = perceptron.kernel_perceptron_train(a0, XTrain, yTrain, num_epoch, sigma)
# pred = perceptron.kernel_perceptron_predict(a, XTrain, yTrain, x, sigma)
#print a

# Visualize the image
# idx = 0
# datapoint = XTrain[idx, 1:]
# plt.imshow(datapoint.reshape((28,28), order = 'F'), cmap='gray')
# plt.show()

# TODO: Test perceptron_predict function, defined in perceptron.py

# TODO: Test perceptron_train function, defined in perceptron.py

# TODO: Test RBF_kernel function, defined in perceptron.py

# TODO: Test kernel_perceptron_predict function, defined in perceptron.py

# TODO: Test kernel_perceptron_train function, defined in perceptron.py

# TODO: Run experiments outlined in HW4 PDF