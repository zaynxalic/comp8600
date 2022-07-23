import numpy as np 
import matplotlib.pyplot as plt 
import math

def normal_distribution(x, mu, sigma2):
    return 1/math.sqrt(2 * math.pi* sigma2)\
         * np.exp(- (x- mu)**2 /(2*sigma2) )

def pdf(x, p,mu, sigma2):
    prob_density = np.zeros((x.shape))
    for i in range(len(mu)):
        prob_density += p[i]* \
            normal_distribution(x, mu[i], sigma2[i])
    return prob_density

def sample_from_GMM(numbers, p, mu, sigma):
    samples = np.zeros(numbers)
    classes = np.random.choice(
            a = np.arange(0,3), 
            size=numbers, 
            p=p)
    for idx,class_ in enumerate(classes):
        samples[idx] =  np.random.normal(
            mu[class_],
            sigma[class_],
            size=1)
    return samples
    
if __name__ == '__main__':
    k = 3
    p = np.array([0.2, 0.3, 0.5])
    mu = np.array([-1,1,3])
    sigma = np.array([.4, .6, 1.5])
    sigma2 = np.power(sigma,2)
    data = np.linspace(-2,7,2000)
    y = pdf(data, p, mu,sigma2 )
    samples = sample_from_GMM\
        (2000, p, mu, sigma)
    plt.figure(figsize= (8,4))
    plt.plot(data,y)
    plt.xticks(np.array([-2,1,4,7])) 
    plt.savefig('2_1.pdf', bbox_inches='tight')
    plt.hist(samples,bins = 100,density=1)
    plt.savefig('2_2.pdf', bbox_inches='tight')
    plt.tight_layout()
    plt.show()  