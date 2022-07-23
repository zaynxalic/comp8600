import numpy as np
################################################################
##### BLR Question Code
################################################################

def single_EM_iter_blr(features, targets, alpha_i, beta_i):
    # (30,2) = (n,d), (30,1) = (n,1), 1, 1
    # Given the old alpha_i and beta_i, computes expectation of latent variable w: M_n and S_n,
    # and using that computes the new alpha and beta values.
    # Should return M_n, S_n, new_alpha, new_beta in that order, with return shapes (M,1), (M,M), None, None
    ### CODE HERE ###
    N,M = features.shape
    I = np.identity(M)
    sn_inv = alpha_i * I + beta_i * features.T@features
    sn = np.linalg.inv(sn_inv)
    mn = beta_i * sn @ features.T @ targets
    e_alpha = (mn.T@mn).squeeze() + np.trace(sn)
    e_beta = ((targets - features @ mn).T @ (targets - features @ mn)).squeeze() + np.trace(features.T@features@sn)
    new_alpha = M/e_alpha
    new_beta = N/e_beta
    return mn, sn, new_alpha, new_beta # (M,1), (M,M), None, None