import numpy as np
from scipy.optimize import fmin_cg
from scipy import optimize
import matplotlib.pyplot as plt
import matplotlib
import time

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
    global start_time
    global args_K
    #print(x)
    print(f"solver_callback iter {iter}")
    print(f"seconds passed {time.time() - start_time}")

    if(iter % 20 == 0):
        save_string = "result" + str(iter) + ".npy"
        current_loglike = fK(x, *args_K)
        print(f"current loglike: {current_loglike}")
        np.save(save_string, x)
    iter = iter + 1

iter = 0
N = 100
#Y = np.load("/Users/samling/Desktop/KTH/AML_project/pca/data.npz")['Y']
Y = np.load("data.npz")['Y']

D = 12

q = 2
cov_matrix = np.identity(q)
y_cov_matrix = np.identity(D)
#x0 = np.random.multivariate_normal(np.zeros((q)), cov_matrix,N)
x0 = np.load("result.npy")
#Y = np.random.multivariate_normal(np.zeros((D)), y_cov_matrix,N)
args_K = (D,N,Y,rbf)

print(f"x0 {x0.shape}")
print(f"Y {Y.shape}")
start_time = time.time()
res2 = optimize.fmin_cg(fK, x0, fprime=None, args=args_K, callback=solver_callback, maxiter=0)
finish_time = time.time()
print(f"total run time {finish_time - start_time}")
np.save("result.npy",res2)
result = np.reshape(res2, (N,2))

a,b = result.T
print(f"a {a.shape}")
print(f"b {b.shape}")
colors = ['red','green','blue']
labels = np.load("data.npz")["labels"]
plt.scatter(a,b,c=labels, cmap=matplotlib.colors.ListedColormap(colors))
plt.show()