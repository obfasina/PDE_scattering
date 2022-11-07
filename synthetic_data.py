
# Script for generating synthetic PDE data

import numpy as np
#import matplotlib.pyplot as plt

#Start with non-homogenous Poisson equation with quadratic source

#Problem domain

# Geometric domain: x from [0,10], y from [0,5] 
# Eigenvalue domain: n = {0,1,....10}, m = {0,1,...10}
# Source is f(x,y) = -x^2 - y^2: 

# Should have 100 10 x 10 meshes where each mesh correson

#Defining eigenfunctions, source, and eigenvalue

def sourced(varint):


    if varint == 0:
        f = lambda x, y : (x**3 + y)
    if varint == 1:
        f = lambda x, y : (x**2 + y**2)
    if varint == 2:
        f = lambda x, y : (y-x)
    if varint == 3:
        f = lambda x, y : ((4*x)-y)
    if varint == 4:
        f = lambda x, y : (-x**2 - y**2)
    if varint == 5:
        f = lambda x, y : (x*y)
    if varint == 6:
        f = lambda x, y : (np.exp(-x + y))
    if varint == 7:
        f = lambda x, y : (np.exp(-y + x)*-x)
    if varint == 8:
        f = lambda x, y : (x**2 + y)
    if varint == 9:
        f = lambda x, y : (x+ (y*x) )



    return f



#Data generation(returns solution mesh)
def datagen(x,y,ev_m,ev_n,en,src):

    """
     x = input x coordinate
     y = input y coordinate 
     ev_m = eigenvalues for x - basis of eigenfunctions
     ev_n = eigenvalues for y - basis of eigenfunctions
     en = specify which mesh you want (by specifying the eigenvalue)

    """

    def X(x,m,xvec):
        return np.sin( ((m+1)*np.pi*x) /len(xvec) )

    def Y(y,n,yvec):
        return np.sin( ((n+1)*np.pi*y) /len(yvec) ) 

    def Evalue(x,y,m,n,xvec,yvec):

        #Defining constant
        C = -4/ (len(xvec)*len(yvec)*( ((m + 1)**2*np.pi**2/len(xvec)**2) + ((n + 1)**2*np.pi**2/len(yvec)**2) ))

        #summing over space
        interm = []
        for i in range(len(xvec)):
            for j in range(len(yvec)):

                interm.append(src(x,y)*X(x,m,xvec)*Y(y,n,yvec))

        sumint = np.sum(interm)

        return C*sumint

    def Solution(x,y,m,n,xvec,yvec):

        return Evalue(x,y,m,n,xvec,yvec)*X(x,m,xvec)*Y(y,n,yvec)

    eval_meshlist = []
    for a in range(len(ev_m)):
        for b in range(len(ev_n)):

            mesh = []
            for i in range(len(x)):
                for j in range(len(y)):
                    mesh.append(Solution(x[i],y[j],ev_m[a],ev_n[b],x,y))

            eval_meshlist.append(mesh)
        

    test = np.array(eval_meshlist[en]).reshape((100,1)) # returns vector of solutions
    ntest = test.reshape((10,10)) # returns mesh of solutions
    return test



