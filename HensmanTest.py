# -*- coding: utf-8 -*-
# Copyright 2009 James Hensman
# Licensed under the Gnu General Public license, see COPYING
 
import numpy as np
import pylab
from PCA_EM import PCA_EM
import kernels
import GP
from scipy import optimize
import matplotlib.pyplot as plt
import matplotlib
#import MLP
 
 
class GPLVM:
    """ TODO: this should inherrit a GP, not contain an instance of it..."""
    def __init__(self,Y,dim):
        self.Xdim = dim
        self.N,self.Ydim = Y.shape
         
        """Use PCA to initalise the problem. Uses EM version in this case..."""
        myPCA_EM = PCA_EM(Y,dim)
        myPCA_EM.learn(100)
        X = np.array(myPCA_EM.m_Z)
        
        self.GP = GP.GP(X,Y)#choose particular kernel here if so desired.
     
    def learn(self,niters):
        for i in range(niters):
            self.optimise_latents()
            self.optimise_GP_kernel()
             
    def optimise_GP_kernel(self):
        """optimisation of the GP's kernel parameters"""
        self.GP.find_kernel_params()
        print(self.GP.marginal(), 0.5*np.sum(np.square(self.GP.X)))
     
    def ll(self,xx,i):
        """The log likelihood function - used when changing the ith latent variable to xx"""
        self.GP.X[i] = xx
        self.GP.update()
        return -self.GP.marginal()+ 0.5*np.sum(np.square(xx))
     
    def ll_grad(self,xx,i):
        """the gradient of the likelihood function for us in optimisation"""
        self.GP.X[i] = xx
        self.GP.update()
        self.GP.update_grad()
        matrix_grads = [self.GP.kernel.gradients_wrt_data(self.GP.X,i,jj) for jj in range(self.GP.Xdim)]
        grads = [-0.5*np.trace(np.dot(self.GP.alphalphK,e)) for e in matrix_grads]
        return np.array(grads) + xx
         
    def optimise_latents(self):
        """Direct optimisation of the latents variables."""
        xtemp = np.zeros(self.GP.X.shape)
        for i,yy in enumerate(self.GP.Y):
            original_x = self.GP.X[i].copy()
            #xopt = optimize.fmin(self.ll,self.GP.X[i],disp=True,args=(i,))
            xopt = optimize.fmin_cg(self.ll,self.GP.X[i],fprime=self.ll_grad,disp=True,args=(i,))
            self.GP.X[i] = original_x
            xtemp[i] = xopt
        self.GP.X = xtemp.copy()
         
         

 

if __name__=="__main__":
    N = 20
    colours = np.arange(N)#something to colour the dots with...
    theta = np.linspace(2,6,N)
    old_Y = np.vstack((np.sin(theta)*(1+theta),np.cos(theta)*theta)).T
    old_Y += 0.1*np.random.randn(N,2)
    Y = np.load("data.npz")['Y']

    thetanorm = (theta-theta.mean())/theta.std()
     
    xlin = np.linspace(-1,1,1000).reshape(1000,1)
     
    myGPLVM = GPLVM(Y,2)
    #GPLVMC(Y,1,nhidden=3)
     
    def plot_current():
        pylab.figure()
        ax = pylab.axes([0.05,0.8,0.9,0.15])
        pylab.scatter(myGPLVM.GP.X[:,0]/myGPLVM.GP.X.std(),np.zeros(N),40,colours)
        pylab.scatter(thetanorm,np.ones(N)/2,40,colours)
        pylab.yticks([]);pylab.ylim(-0.5,1)
        ax = pylab.axes([0.05,0.05,0.9,0.7])
        pylab.scatter(Y[:,0],Y[:,1],40,colours)
        Y_pred = myGPLVM.GP.predict(xlin)[0]
        pylab.plot(Y_pred[:,0],Y_pred[:,1],'b')
     
    class callback:
        def __init__(self,print_interval):
            self.counter = 0
            self.print_interval = print_interval
        def __call__(self,w):
            self.counter +=1
            if not self.counter%self.print_interval:
                print(self.counter, 'iterations, cost: ',myGPLVM.GP.get_params())
                plot_current()
                 
    cb = callback(100)
             
    #myGPLVM.learn(callback=cb)
    myGPLVM.learn(5)
    #plot_current()
    result = myGPLVM.GP.X
    a,b = result.T
    print(f"a {a.shape}")
    print(f"b {b.shape}")

    colors = ['red','green','blue']
    labels = np.load("data.npz")["labels"]
    plt.scatter(a,b,c=labels, cmap=matplotlib.colors.ListedColormap(colors))
    plt.show()
     
    pylab.show()
         
         
         
        