import numpy as np
from scipy.optimize import fmin_cg
from scipy import optimize
import matplotlib.pyplot as plt

"""
args = (2, 3, 7, 8, 9, 10)  # parameter values

def f(x, *args):
    u, v = x
    a, b, c, d, e, f = args
    return a*u**2 + b*u*v + c*v**2 + d*u + e*v + f
def gradf(x, *args):
    u, v = x
    a, b, c, d, e, f = args
    gu = 2*a*u + b*v + d     # u-component of the gradient
    gv = b*u + 2*c*v + e     # v-component of the gradient
    return np.asarray((gu, gv))


x0 = np.asarray((0, 0))  # Initial guess.

res1 = optimize.fmin_cg(f, x0, fprime=None, args=args)

print(res1)
"""



def fK(x, *args):
    # x n x q latent variable matrix, q is our chosen dimension
    # *args = [Y,kernel function]
    # L from the paper
    #print("fK called")
    D,N,Y,kernel_function = args
    x = np.reshape(x,(100,2))
    K = kernel_function(x)
    """
    print(f"x {x.shape}")
    print(f"K {K.shape}")
    print(f"Y {Y.shape}")
    """
    result = -1*D*N/2 * np.log(2*np.pi)-D/2*np.log(np.linalg.det(K))-1/2*np.trace(np.linalg.inv(K).dot(np.dot(Y,Y.T)))
    return -1 * result

def rbf(X, l=10):
    import numpy as np
    
    n = len(X)
    K = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            dot = np.dot(X[i] - X[j], X[i] - X[j])
            K[i, j] = np.exp(-l*dot)
    return K

def solver_callback(x):
    global iter
    #print(x)
    print(f"solver_callback iter {iter}")
    iter = iter + 1

iter = 1
N = 100
Y = np.load("/Users/samling/Desktop/KTH/AML_project/pca/data.npz")['Y']
D = 12

q = 2
cov_matrix = np.identity(q)
y_cov_matrix = np.identity(D)
x0 = np.random.multivariate_normal(np.zeros((q)), cov_matrix,N)
#Y = np.random.multivariate_normal(np.zeros((D)), y_cov_matrix,N)
args_K = (D,N,Y,rbf)

print(f"x0 {x0.shape}")
print(f"Y {Y.shape}")
res2 = optimize.fmin_cg(fK, x0, fprime=None, args=args_K, callback=solver_callback, maxiter=10)

result = np.reshape(res2, (N,2))

a,b = result.T

plt.scatter(a,b)
plt.show()