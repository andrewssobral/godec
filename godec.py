from numpy import prod, zeros, sqrt
from numpy.random import randn
from scipy.linalg import qr
from sklearn.metrics import mean_squared_error


def godec(X, rank=1, card=None, iterated_power=1, max_iter=100, tol=0.001):
    """
    GoDec - Go Decomposition (Tianyi Zhou and Dacheng Tao, 2011)
    
    The algorithm estimate the low-rank part L and the sparse part S of a matrix X = L + S + G with noise G.

    Parameters
    ----------
    X : array-like, shape (n_features, n_samples), which will be decomposed into a sparse matrix S 
        and a low-rank matrix L.
    
    rank : int >= 1, optional
        The rank of low-rank matrix. The default is 1.
    
    card : int >= 0, optional
        The cardinality of the sparse matrix. The default is None (number of array elements in X).
    
    iterated_power : int >= 1, optional
        Number of iterations for the power method, increasing it lead to better accuracy and more time cost. The default is 1.
    
    max_iter : int >= 0, optional
        Maximum number of iterations to be run. The default is 100.
    
    tol : float >= 0, optional
        Tolerance for stopping criteria. The default is 0.001.

    Returns
    -------
    L : array-like, low-rank matrix.
    
    S : array-like, sparse matrix.

    LS : array-like, reconstruction matrix.
    
    RMSE : root-mean-square error.
    
    References
    ----------
    Zhou, T. and Tao, D. "GoDec: Randomized Lo-rank & Sparse Matrix Decomposition in Noisy Case", ICML 2011.
    """
    iter = 1
    RMSE = []
    card = prod(X.shape) if card is None else card
    
    X = X.T if(X.shape[0] < X.shape[1]) else X
    m, n = X.shape
    
    # Initialization of L and S
    L = X
    S = zeros(X.shape)
    LS = zeros(X.shape)
    
    while True:
        # Update of L
        Y2 = randn(n, rank)
        for i in range(iterated_power):
            Y1 = L.dot(Y2)
            Y2 = L.T.dot(Y1)
        Q, R = qr(Y2, mode='economic')
        L_new = (L.dot(Q)).dot(Q.T)
        
        # Update of S
        T = L - L_new + S
        L = L_new
        T_vec = T.reshape(-1)
        S_vec = S.reshape(-1)
        idx = abs(T_vec).argsort()[::-1]
        S_vec[idx[:card]] = T_vec[idx[:card]]
        S = S_vec.reshape(S.shape)
        
        # Reconstruction
        LS = L + S
        
        # Stopping criteria
        error = sqrt(mean_squared_error(X, LS))
        RMSE.append(error)
        
        print("iter: ", iter, "error: ", error)
        if (error <= tol) or (iter >= max_iter):
            break
        else:
            iter = iter + 1

    return L, S, LS, RMSE
