import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

from test_data import TestData

class Two2DGaussians(TestData):

    """ 
    2 events, modeled as 2D gaussians with different means
    N = total # of scenes
    beta = variance of each Gaussian component
    """
    def __init__(self, N = 25, beta = 0.1):
        pi = np.array([0.4, 0.6])
        mus = [[1, 1], [-1, -1]]
        x = np.zeros((N, 2), dtype=np.float32)
        for n in range(N/2):
            x[n, :] = np.random.multivariate_normal(mus[0], np.eye(2) * beta)
        for n in range(N/2, N):
            x[n, :] = np.random.multivariate_normal(mus[1], np.eye(2) * beta)
           
        self.D = 2
        self.X = x
        self.y = np.concatenate([np.zeros((N/2), dtype=int), np.ones((N/2), dtype=int)])

class TwoAlternating2DGaussians(TestData):

    """ 
    2 alternating events, modeled as gaussians with different means
    N = total # of scenes
    beta = variance of each Gaussian component
    """
    def __init__(self, N=100, beta=0.1):
	pi = np.array([0.4, 0.6])
	mus = [[1, 1], [-1, -1]]
	stds = np.ones((2, 2)) * beta
	x = np.zeros((N, 2), dtype=np.float32)
	for n in range(N/4):
	    x[n, :] = np.random.multivariate_normal(mus[0], np.diag(stds[0]))
	for n in range(N/4, N/2):
	    x[n, :] = np.random.multivariate_normal(mus[1], np.diag(stds[1]))
	for n in range(N/2, N/4*3):
	    x[n, :] = np.random.multivariate_normal(mus[0], np.diag(stds[0]))
	for n in range(N/4*3, N):
	    x[n, :] = np.random.multivariate_normal(mus[1], np.diag(stds[1]))

	self.D = 2	    
        self.X = x
        self.y = np.concatenate([np.zeros(N/4, dtype=int), np.ones(N/4, dtype=int), np.zeros(N/4, dtype=int), np.ones(N/4, dtype=int)])


class TwoLinearDynamicalSystems(TestData):

    """
    Two 2D LDSs where event corresponds to jumps in space.
    N = total # of scenes
    beta = variance of Gaussians
    """
    def __init__(self, N=100, beta=0.01):
	pi = np.array([0.4, 0.6])
	bs = [[1, 1], [-1, -1]]  # starting point!
	ws = [np.array([[1, 0], [.5, 2]]), np.array([[-1, .2], [-.2, 1]])]
	stds = np.eye(2) * beta
	x = np.zeros((N, 2), dtype=np.float32)
	
	x[0, :] = np.random.multivariate_normal(bs[0], stds)
	for n in range(1, N/2):
	    _x = x[n-1, :] + np.array([0.05, 0.05])
	    x[n, :] = np.random.multivariate_normal(_x, stds)
	    
	x[N/2, :] = np.random.multivariate_normal(bs[1], stds)
	for n in range(N/2+1, N):
	    _x = x[n-1, :] + np.array([-0.05, -0.05])
	    x[n, :] = np.random.multivariate_normal(_x, stds)
        
        self.D = 2
        self.X = x
        self.y = np.concatenate([np.zeros(N/2, dtype=int), np.ones(N/2, dtype=int)])

