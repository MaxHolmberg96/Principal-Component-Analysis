def kernelPCA(X, kernel="rbf", desired_dims=2):
    import numpy as np
    
    if kernel == "rbf":
        kernel = rbf
    else:
        raise Exception("Kernel type: " + kernel + " is not supported")
        return
    
    K = kernel(X)
    N = len(X)
    oneN = np.ones((N, N)) / N
    normalized_K = K - oneN.dot(K) - K.dot(oneN) + oneN.dot(K).dot(oneN)
    evalues, evectors = np.linalg.eig(normalized_K)
    # Sort eigenvalues first so we can normalize the eigenvectors
    evalues = evalues[(-evalues).argsort()]
    for i in range(len(evectors)):
        evectors[i] = evectors[i] / np.sqrt(evalues[i])
    # sort with largest first, these are the principal components
    evectors = evectors[(-evalues).argsort()]
    projection = K.T.dot(evectors[:, :desired_dims])
    return projection

def rbf(X, l=10):
    import numpy as np
    
    n = len(X)
    K = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            dot = np.dot(X[i] - X[j], X[i] - X[j])
            K[i, j] = np.exp(-l*dot)
    return K