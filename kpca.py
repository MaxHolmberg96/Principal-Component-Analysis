from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
import numpy as np

def kernelPCA(X, kernel="rbf", desired_dims=2, gamma=15):
    import numpy as np
    
    if kernel == "rbf":
        #kernel = rbf
        kernel = rbf_kernel_pca
    else:
        raise Exception("Kernel type: " + kernel + " is not supported")
        return

    
    
    #K = kernel(X, gamma=15, n_components=2)
    #N = len(X)
    #oneN = np.ones((N, N)) / N
    #normalized_K = K - oneN.dot(K) - K.dot(oneN) + oneN.dot(K).dot(oneN)
    #evalues, evectors = np.linalg.eig(normalized_K)
    # Sort eigenvalues first so we can normalize the eigenvectors
    #evalues = evalues[(-evalues).argsort()]
    #for i in range(len(evectors)):
    #    evectors[i] = evectors[i] / np.sqrt(evalues[i])
    # sort with largest first, these are the principal components
    #evectors = evectors[(-evalues).argsort()]
    #projection = K.T.dot(evectors[:, :desired_dims])
    #projection = normalized_K.T.dot(evectors[:, :desired_dims])
    #return projection
    return rbf_kernel_pca(X, gamma=gamma, n_components=desired_dims)

def rbf(X, l=10):
    import numpy as np
    
    n = len(X)
    K = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            dot = np.dot(X[i] - X[j], X[i] - X[j])
            K[i, j] = np.exp(-l*dot)
    return K

def rbf_kernel_pca(X, gamma, n_components):
    """
    RBF kernel PCA implementation.    
    Parameters
    ------------
    X: {NumPy ndarray}, shape = [n_examples, n_features]  
    gamma: float
        Tuning parameter of the RBF kernel    
    n_components: int
        Number of principal components to return    
    Returns
    ------------
    X_pc: {NumPy ndarray}, shape = [n_examples, k_features]
        Projected dataset   
    """
    # Calculate pairwise squared Euclidean distances
    # in the MxN dimensional dataset.
    sq_dists = pdist(X, 'sqeuclidean')    
    # Convert pairwise distances into a square matrix.
    mat_sq_dists = squareform(sq_dists)    
    # Compute the symmetric kernel matrix.
    K = exp(-gamma * mat_sq_dists)    
    # Center the kernel matrix.
    N = K.shape[0]
    one_n = np.ones((N,N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)    
    # Obtaining eigenpairs from the centered kernel matrix
    # scipy.linalg.eigh returns them in ascending order
    eigvals, eigvecs = eigh(K)
    eigvals, eigvecs = eigvals[::-1], eigvecs[:, ::-1]    
    # Collect the top k eigenvectors (projected examples)
    X_pc = np.column_stack([eigvecs[:, i]
                           for i in range(n_components)])    
    return X_pc