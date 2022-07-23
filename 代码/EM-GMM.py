import numpy as np 
import matplotlib.pyplot as plt 
import math
from sklearn.mixture import GaussianMixture

def normal_distribution(x, mu, sigma2):
    return 1/np.sqrt(2 * np.pi* sigma2) * np.exp(- np.square(x- mu) /(2* sigma2) )

def E_step(x, p, mu, sigma2,N,K):
    gamma = np.zeros((N,K))
    for k in range(K):
        gamma[:,k] = p[k] * normal_distribution(x,mu[k],sigma2[k])
    gamma = gamma/np.sum(gamma,axis=1)[:,None]
    return gamma

def M_step(x, gamma, N, K):
    N_k = np.sum(gamma,axis=0,keepdims=1).T
    mu_new = gamma.T@x[:,None]/N_k
    sigma_new = np.sum(gamma.T*(x - mu_new)**2, axis=1)[:,None]/N_k
    pi_new = N_k/N
    return mu_new, sigma_new, pi_new


def sample_from_GMM(numbers, p, mu, sigma2):
    samples = np.zeros(numbers)
    classes = np.random.choice(
                                a = np.arange(0,3), 
                                size=numbers, 
                                p=p)
    for idx,class_ in enumerate(classes):
        samples[idx] =  np.random.normal(
            mu[class_],
            sigma2[class_],
            size=1)
    return samples

def pdf(x, p,mu, sigma):
    prob_density = np.zeros((x.shape))
    for i in range(len(mu)):
        prob_density += p[i]*normal_distribution(x, mu[i], sigma[i])
    return prob_density
    
def loglikelihood(x, p, mu, sigma2):
    log = np.sum(np.log(pdf(x, p,mu, sigma2)))
    return log

def convergence(prev_loss, later_loss, tol = 1e-7):
    if abs(prev_loss - later_loss) < tol:
        return True
    else:
        return False

if __name__ == '__main__':
    p = np.array([0.2, 0.3, 0.5])
    mu = np.array([-1,1,3])
    sigma = np.array([.4, .6, 1.5])
    data = sample_from_GMM\
        (2000, p, mu, np.sqrt(sigma))
    # initialisation
    N = data.shape[0]
    K = 3
    pi_ = np.ones((K,1))/K # (K,1)
    mu_ = np.array([[-1],[0],[1]]) # (K,1)
    sigma2_ = np.array([.1,.1,.1])
    max_iter = 1600
    log = []
    for iter in range(max_iter):
        if iter < 2 or not convergence(log[0],log[1]):
            gamma = E_step(data, pi_, mu_, sigma2_,N,K)
            mu_,sigma2_, pi_ = M_step(data,gamma,N,K)
            log.append(loglikelihood(data, pi_, mu_, sigma2_))
        else:
            break
        if iter > 2:
            log = log[-2:]
    print(iter)
    print("pi: ",pi_)
    print("mu: ",mu_)
    print("sigma: ", np.sqrt(sigma2_))
    
  