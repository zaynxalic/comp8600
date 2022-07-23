import dataclasses
import numpy as np
from functools import lru_cache
from scipy.integrate import quad

import framework.em_code as em_code
import emm_question as emm
################################################################
##### Helper functions (DO NOT CHANGE)
################################################################

# @lru_cache  # Makes things go fast
def normalise_expontial_family(sufstat, eta):
    unnorm_prob = lambda z: np.exp(sufstat(z) @ np.array(eta))
    Z, err = quad(unnorm_prob, -np.inf, np.inf)
    return float(Z)

def exponential_family_pdf(x, sufstat, eta):
    # Input Shapes: (1,), None, (M,)
    # sufstat designates the sufficient statistic map for the exponential
    # family, taking values in (1,) to (M,).
    unnorm_prob = lambda z: np.exp(sufstat(z) @ eta)
    eta = eta.squeeze()
    Z = normalise_expontial_family(sufstat, tuple(eta))
    prob = unnorm_prob(x) / Z # Here Z = exp(-psi(eta))
    return prob

################################################################
##### EMM Question Code
################################################################

def weighted_probs(data, pi, eta, sufstat, N, K):
    # Input Shapes: (N,), (K,1), (K,m), None, None
    # Should implement pi_k * q(x_n|eta_k) for each n, k, and thus return shape
    # should be (N,K). You should use exponential_family_pdf as defined above.
    # Note: sufstat(x) = u(x).
    # Works for scalars ((1,) -> (2,)); and 1D arrays ((N,) -> (N, 2)).
    ### CODE HERE ###
    probs = np.zeros((N,K))
    for k in range(K):
        probs[:,k]  = exponential_family_pdf(data, sufstat, eta[k]).reshape(-1,1)@pi[k]
    return probs # (N, K)

def e_step_EMM(data, pi, eta, sufstat, N, K):
    # Input Shapes: (N,), (K,1), (K,m), None, None
    # Should implement gamma_nk for each n, k; and thus return shape should be (N,K).
    # Note: sufstat(x) = u(x).
    # This works for scalars ((1,) -> (2,)); and 1D arrays ((N,) -> (N, 2)).
    # It should use weighted_probs.
    ### CODE HERE ###
    gamma = weighted_probs(data, pi, eta, sufstat, N, K)/(np.sum(weighted_probs(data, pi, eta, sufstat, N, K),axis=1)).reshape(-1,1)
    return gamma # (N, K)

def m_step_EMM(data, gamma, sufstat, exp_to_nat, N, K):
    # Input Shapes: (N,D), (N,K), None, None, None
    # Should implement updates for pi, Eta, and return them in that order.
    # exp_to_nat is a function which converts the expectation parameter to
    # natural parameter. This only works dimensions (2,) -> (2,).
    # Note: sufstat(x) = u(x).
    # This works for scalars (1,) -> (2,); and 1D arrays (N,) -> (N, 2).
    # Return shapes should be (K,1), (K,m).
    ### CODE HERE ###
    lambda_k = np.zeros((K,2))
    eta_new = np.zeros((K,2))
    N_k = np.sum(gamma, axis=0)
    pi_new = (N_k/N).reshape(-1,1)
    for k in range(K):
        lambda_k[k] = sufstat(data).T @ gamma[:,k] / N_k[k]
        eta_new[k] = exp_to_nat(lambda_k[k])
    return pi_new, eta_new # (K,1), (K,m)