import numpy as np
np.random.seed(1234)

def random_posdef(n):
  A = np.random.rand(n, n)
  return np.dot(A, A.transpose())

# Parameter initialization ###
K = 3
pi = [1.0/K for i in range(K)]
means = [[0,0] for i in range(K)]
covs = [random_posdef(2) for i in range(K)]
##############################
