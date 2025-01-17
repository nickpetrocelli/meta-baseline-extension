import numpy as np
import scipy.sparse.linalg as sla

# Robust mean estimation via Gradient Descent:
#   Y. Cheng, I. Diakonikolas, R. Ge, M. Soltanolkotabi.
#   High-Dimensional Robust Mean Estimation via Gradient Descent.
#   In Proceedings of the 37th International Conference on Machine Learning (ICML), 2020.

# Ported from Matlab to Python by Nicholas Petrocelli.

# Input: X (N x d, N d-dimensinoal samples) and eps (fraction of corruption).
# Output: a hypothesis vector mu (a column vector).
# The number of iteration nItr is set to 10, which can be changed as you see fit.


def robust_mean_pgd(X, eps):

    # N = number of samples, d = dimension.
    N = np.shape(X)[0]
    d = np.shape(X)[1]
    epsN = round(eps * N)
    
    stepSz = 1.0 / N
    nItr = 10
    w = np.ones((N, 1)) / N
    #for itr = 1:nItr
    for itr in range(1, nItr):
        # Sigma_w = X' * diag(w) * X - X' * w * w' * X;
        # [u, lambda] = eigs(Sigma_w, 1);
        Xw = X.T @ w
        Sigma_w_fun = lambda v:  X.T @ (w * (X @ np.reshape(v, (d, 1)))) - Xw @ Xw.T @ np.reshape(v, (d, 1))
        # [u, lambda1] = eigs(Sigma_w_fun, d, 1)
        # https://stackoverflow.com/questions/51247998/numpy-equivalents-of-eig-and-eigs-functions-of-matlab
        Sigma_w_op = sla.LinearOperator((d, d), matvec=Sigma_w_fun)
        W, V = sla.eigs(Sigma_w_op, 1)
        lambda1 = W[0]
        u = V[:, 0]


        # Compute the gradient of spectral norm (assuming unique top eigenvalue)
        # nabla_f_w = (X * u) .* (X * u) - (w' * X * u) * X * u;
        Xu = X @ u
        Xu = np.reshape(Xu, (np.shape(Xu)[0], 1))
        # print(np.shape(Xu))
        # print(np.shape(w.T))
        # print(np.shape(w.T @ Xu))

        nabla_f_w = Xu * Xu - 2 * (w.T @ Xu) * Xu;
        old_w = w;
        w = w - stepSz * nabla_f_w / np.linalg.norm(nabla_f_w, ord=2);
        # Projecting w onto the feasible region
        w = project_onto_capped_simplex_simple(w, 1 / (N - epsN))
        
        # Use adaptive step size.
        #   If objective function decreases, take larger steps.
        #   If objective function increases, take smaller steps.
        Sigma_w_fun = lambda v: X.T @ (w * (X @ np.reshape(v, (d, 1)))) - Xw @ Xw.T @ np.reshape(v, (d, 1))
        Sigma_w_op = sla.LinearOperator((d, d), matvec=Sigma_w_fun)
        new_lambda1, _ = sla.eigs(Sigma_w_op, 1)
        #[~, new_lambda1] = eigs(Sigma_w_fun, d, 1);
        if (new_lambda1 < lambda1):
            stepSz = stepSz * 2
        else:
            stepSz = stepSz / 4
            w = old_w
        
    # TODO why is this complex?
    return np.real(np.reshape(X.T @ w, (d,)))


def project_onto_capped_simplex_simple(w, cap):
    # The projection of w onto the capped simplex is  min(max(w - t, 0), cap)  for some scalar t
    tL = np.amin(w) - 1
    tR = np.amax(w)
    for bSearch in range(1,51):
        t = (tL + tR) / 2
        if (np.sum(np.minimum(np.maximum(w - t, 0), cap)) < 1):
            tR = t
        else:
            tL = t
        
    
    return np.minimum(np.maximum(w - t, 0), cap)

