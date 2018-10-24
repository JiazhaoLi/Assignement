import numpy as np
import math
from scipy.stats import multivariate_normal
import matplotlib as mpl
import matplotlib.pyplot as plt

#np.random.seed(1257) #5
#np.random.seed(1292) #10
np.random.seed(1234)
def random_posdef(n):
  A = np.random.rand(n, n)
  return np.dot(A, A.transpose())

# Parameter initialization ###
K = 3
pi = [1.0/K for i in range(K)]
means = [[0,0] for i in range(K)]
covs = [random_posdef(2) for i in range(K)]
c = np.load("gmm_data.npy") #1000*2

for _ in range(50):
    summ = 0
    for j in range(K):
        norm = multivariate_normal(means[j], covs[j])
        summ += pi[j]*norm.pdf(c)
    for j in range(K):
        norm = multivariate_normal(means[j], covs[j])
        tau = pi[j]*norm.pdf(c)
        gamma = tau / summ

        means[j] = np.dot(gamma, c)/sum(gamma)
        covs[j] = np.dot(gamma * (c - means[j]).T, (c - means[j])) / sum(gamma)
        pi[j] = sum(gamma) / 1000

#print(pi)
#print(means)
#print(covs)

x1 = np.linspace(-1, 6, 500)
x2 = np.linspace(0, 8, 500)
X, Y = np.meshgrid(x1, x2)
x_len = x1.shape[0]
y_len = x2.shape[0]
Z = np.zeros((x_len, y_len))

for jj in range(K):
    norm = multivariate_normal(means[jj], covs[jj])
    for i in range(x_len):
        for j in range(y_len):
            Z[j][i] = norm.pdf([x1[i],x2[j]])
    cs = plt.contour(x1, x2, Z)
    plt.clabel(cs, inline=0.1, fontsize=10)

plt.scatter(c[:,0], c[:,1],s= 1, c = "r",label='data')
plt.show()
