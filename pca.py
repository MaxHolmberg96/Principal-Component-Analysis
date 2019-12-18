def PCA(X):
    import numpy as np
    N = len(X)
    q = len(X[0])
    # Center the features around zero
    for i in range(q):
        X[:, i] = X[:, i] - np.mean(X[:, i])
    cov = X.T.dot(X) / N
    eigenValues, eigenVectors = np.linalg.eig(cov)
    sortedEigenVectors = eigenVectors[(-eigenValues).argsort()]
    sortedEigenValues = eigenValues[(-eigenValues).argsort()]
    for i in range(len(sortedEigenVectors)):
        sortedEigenVectors[i] = sortedEigenVectors[i] / np.linalg.norm(sortedEigenVectors[i])
    return X.dot(sortedEigenVectors)
    