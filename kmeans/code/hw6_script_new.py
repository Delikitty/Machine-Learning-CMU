import os
import csv
import numpy as np
from sklearn.decomposition import PCA
import kmeans_new

# Point to data directory here
# By default, we are pointing to '../data/'
data_dir = os.path.join('..','data')

# Load numeric data files into numpy arrays
X = np.genfromtxt(os.path.join(data_dir, 'kmeans_test_data.csv'), delimiter=',')

#n = X.shape[0]
#d = X.shape[1]

# C = np.random.random(size=18)
# C = np.reshape(C,(9,2))


# QnA 30
'''
X = np.array([[1,1],[1.5,2],[3,4],[5,7],[3.5,5],[4.5,5],[3.5,4.5]])
C = np.array([[1,1],[5,7]])
#C = np.array([[1,1],[1,2],[1,3],[2,1],[1,2],[2,3],[3,1],[2,2],[3,3]])
C = np.float32(C)

a = kmeans.update_assignments(X,C)
print a

CC = kmeans.update_centers(X, C, a)
print CC
'''


# (C,a) = kmeans.lloyd_iteration(X, C)

# for k in (1~10), min obj value?
# for k in range(20):
  #  print k


# QnA 28 29
mnist_X = np.genfromtxt(os.path.join(data_dir, 'mnist_data.csv'), delimiter=',')
# (2876,784)
#n_component = 784
pca = PCA(n_components=5)
reduced = pca.fit_transform(mnist_X)
(best_C, best_a, best_obj) = kmeans_new.kmeans_cluster( reduced , 3 , 'fixed', 1)
print best_obj


#(best_C, best_a, best_obj) = kmeans.kmeans_cluster(X, 1, 'fixed', 10)
#print best_obj

# (best_C, best_a, best_obj) = kmeans.kmeans_cluster(X, 1, 'fixed', 10)
# print best_obj

# TODO: Test update_assignments function, defined in kmeans.py

# TODO: Test update_centers function, defined in kmeans.py

# TODO: Test lloyd_iteration function, defined in kmeans.py

# TODO: Test kmeans_obj function, defined in kmeans.py

# TODO: Run experiments outlined in HW6 PDF

# For question 9 and 10
# from sklearn.decomposition import PCA
# mnist_X = np.genfromtxt(os.path.join(data_dir, 'mnist_data.csv'), delimiter=',')