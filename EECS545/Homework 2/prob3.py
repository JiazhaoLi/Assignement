import numpy as np


# we defined a class for sequential bayesian learner
class bayesian_linear_regression(object):

    # initialized with covariance matrix(sigma), mean vector(mu) and prior(beta)
    def __init__(self,sigma,mu,beta):
        self.sigma = sigma
        self.mu = mu
        self.beta = beta

    # you need to implement the update function
    # when received additional design matrix phi and continuous label t
    def update(self,phi,t):
        pass


def data_generator(size,scale):
    x = np.random.uniform(low=-3, high=3, size=size)
    rand = np.random.normal(0, scale=scale, size=size)
    y = 0.5 * x - 0.3 + rand
    phi = np.array([[x[i], 1] for i in range(x.shape[0])])
    t = y
    return phi, t


def main():
    # initialization
    alpha = 2
    sigma_0 = np.diag(1.0/alpha*np.ones([2]))
    mu_0 = np.zeros([2])
    beta = 1.0
    blr_learner = bayesian_linear_regression(sigma_0, mu_0, beta=beta)

    num_episodes = 10

    for epi in range(num_episodes):
        phi, t = data_generator(1,1.0/beta)
        blr_learner.update(phi,t)

main()


