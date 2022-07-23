import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D #This is for 3d scatter plots.
import math
import random
import functools
import scipy.stats as ss

def E_step(pi, mu, sigma, X):
    '''
    X - (800,3)
    sigma - list of (3,3), k length
    pi - list of real number, k length
    mu - list of (3,), k length
    
    r - (800,4) every row represents rik for 4 centroids
    '''
    # YOUR CODE HERE
    m,n = X.shape
    k = len(pi)
    # calculate responsibilities
    numerator = np.zeros((m,k))
    for i in range(k):
        # calculate responsibilities for each muti-guassian model
        numerator[:,i] = pi[i] * ss.multivariate_normal.pdf(X,mu[i],sigma[i])
    r = numerator/(np.sum(numerator,axis=1)).reshape(-1,1)  # (800,4)
    
    
    return r

def M_step(r, X):
    '''
    r - rik (800,4)
    X - X (800,3)
    '''
    # YOUR CODE HERE
    m,n = X.shape
    k = r.shape[1]
    Nk = np.sum(r,axis=0)
    ## calculate for mu
    mu = []
    for i in range(k):
        # calculate mu for the ith centroids
        mu.append(np.sum(r[:,i].reshape(-1,1)*X,axis=0)/Nk[i])
    
    ## calculate for sigma
    sigma = []
    for i in range(k):
        # calculate covariance mat for the ith centroids
        sigma_temp = np.zeros((n,n))
        sigma_temp += (r[:,i].reshape(1,-1) * (X-mu[i].reshape(1,-1)).T)@(X-mu[i].reshape(1,-1))
        sigma.append(sigma_temp/Nk[i])
    
    ## calculate for pi
    pi = Nk/m
    
    
    return mu,sigma,pi

x = np.array([[1,0],
                [-1,0],
                [0,1],
                [0,-1]])
mu = [np.array([0,1]),np.array([1,0])]
pi = [0.5,0.5]
sigma = [np.array([[0.1,0],[0,0.1]]),np.array([[0.1,0],[0,0.1]])]
# E step
gamma = E_step(pi,mu,sigma,x)
# M step
mu, Sigma, pi = M_step(gamma,x)
print('\n',f"gamma:\n {gamma}   \n mu:\n{mu}    \n sigma:\n{Sigma}  \n pi:{pi}")