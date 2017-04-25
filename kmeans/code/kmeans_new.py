import numpy as np
import math

########################################################################
#######  you should maintain the  return type in starter codes   #######
########################################################################

def update_assignments(X, C):
  # Input:
  #   X is the data matrix (n, d),  2-d array
  #   C is the cluster centers (k, d), 2-d array
  # Output:
  #   a is the cluster assignments (n,), 1-d array

  a = np.zeros(X.shape[0])
  # a is a (n,) 1d array
  n = X.shape[0]
  k = C.shape[0]
  for j in range(n):
    xj = X[j,:]
    xj = np.matrix(xj)
    xjrepeat = np.repeat(xj,k,0)
    xjrepeat = np.array(xjrepeat)
    # xjrepeat is a k*d array
    tmp = xjrepeat - C
    tmp = np.multiply(tmp,tmp)
    tmp = np.sum(tmp,1)
    # mini = np.min(tmp)
    indcenter= np.argmin(tmp) # np.where(tmp==mini)
    a[j] = indcenter

  '''
  for i in range(k):
    ci = C[i,:]
    cirepeat = np.repeat(ci,n,axis=0)
    # xjrepeat is a k*d array
    tmp = cirepeat - X
    tmp = np.multiply(tmp,tmp)
    tmp = np.sum(tmp,1)
    mini = np.min(tmp)
    ind = np.where(tmp==mini)
    a[ind] = i
  '''

  return a

def update_centers(X, C, a):
  # Input:
  #   X is the data matrix (n, d),  2-d array
  #   C is the cluster centers (k, d), 2-d array
  #   a is the cluster assignments (n,), 1-d array
  # Output:
  #   C is the new cluster centers (k, d), 2-d array
  k = C.shape[0]
  for i in range(k):

    '''
    ci = C[i,:]
    ci = np.multiply(ci,ci)
    ci = np.sum(ci)
    ci = np.sqrt(ci)
    '''
    # find the index that which sample of X in ci
    indi = np.where(a==i)
    xi = X[indi,:]

    totalxi = np.sum(xi,0)
    totalxi = np.sum(totalxi, 0)
    num_xi = xi.shape[1] # * xi.shape[1]
    # totalxi = np.rint(np.true_divide(totalxi,100)) * 100
    # num_xi = np.rint(np.true_divide(num_xi,100)) * 100
    '''if num_xi==0:
      num_xi = 100
    C[i,:] = np.true_divide(totalxi,num_xi)'''
    if num_xi!=0:
      C[i,:] = (np.true_divide(totalxi,num_xi))

  return C



def lloyd_iteration(X, C):
  # Input:
  #   X is the data matrix (n, d),  2-d array
  #   C is the initial cluster centers (k, d), 2-d array
  # Output:
  #   C is the cluster centers (k, d), 2-d array
  #   a is the cluster assignments (n,), 1-d array


  a = np.zeros(X.shape[0])
  change = 1
  #a_new = update_assignments(X, C)
  #change = np.abs(a_new - a)

  while np.sum(change) != 0:
    a_new = a
    a = update_assignments(X, C)
    C = update_centers(X, C, a)
    # C_new = update_centers(X, C, a)
    change = np.abs(a_new - a)


  return (C, a)

def kmeans_obj(X, C, a):
  # Input:
  #   X is the data matrix (n, d),  2-d array
  #   C is the cluster centers (k, d), 2-d array
  #   a is the cluster assignments (n,), 1-d array
  # Output:
  #   obj is the k-means objective of the provided clustering, scalar, float
  n = X.shape[0]
  k = C.shape[0]
  tmp_now = 0
  for i in range(k):
    ci = C[i,:]
    indi = np.where(a==i)
    x = X[indi,:]
    tmp = x - ci
    tmp_new = np.sum(np.multiply(tmp,tmp))
    sumobj = tmp_new + tmp_now
    tmp_now = sumobj

  obj = tmp_now

  return obj


########################################################################
#######          DO NOT MODIFY, BUT YOU SHOULD UNDERSTAND        #######
########################################################################

# kmeans_cluster will be used in the experiments, it is available after you
# have implemented lloyd_iteration and kmeans_obj.

def kmeans_cluster(X, k, init, num_restarts):
  n = X.shape[0]
  # Variables for keeping track of the best clustering so far
  best_C = None
  best_a = None
  best_obj = np.inf
  for i in range(num_restarts):
    if init == "random":
      perm = np.random.permutation(range(n))
      C = np.copy(X[perm[0:k]])
    elif init == "kmeans++":
      C = kmpp_init(X, k)
    elif init == "fixed":
      C = np.copy(X[0:k])
    else:
      print "No such module"
    # Run the Lloyd iteration until convergence
    (C, a) = lloyd_iteration(X, C)
    # Compute the objective value
    obj = kmeans_obj(X, C, a)
    if obj < best_obj:
      best_C = C
      best_a = a
      best_obj = obj
  return (best_C, best_a, best_obj)



########################################################################
#######     DO NOT MODIFY, READ ONLY IF YOU ARE INTERESTED       #######
########################################################################

def kmpp_init(X, k):
  n = X.shape[0]
  sq_distances = np.ones(n)
  center_ixs = list()
  for j in range(k):
    # Choose a new center index using D^2 weighting
    ix = discrete_sample(sq_distances)
    # Update the squared distances for all points
    deltas = X - X[ix]
    for i in range(n):
      sq_dist_to_ix = np.power(np.linalg.norm(deltas[i], 2), 2)
      sq_distances[i] = min(sq_distances[i], sq_dist_to_ix)
    # Append this center to the list of centers
    center_ixs.append(ix)
  # Output the chosen centers
  C = X[center_ixs]
  return np.copy(C)


def discrete_sample(weights):
  total = np.sum(weights)
  t = np.random.rand() * total
  p = 0.0
  for i in range(len(weights)):
    p = p + weights[i];
    if p > t:
      ix = i
      break
  return ix