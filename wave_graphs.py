from __future__ import print_function
from mshr import *
from fenics import *
import numpy as np
import fenics
import sys
#from mshr import *
import matplotlib.pyplot as plt
import dill
import pickle
import argparse
import pandas as pd



Lap = np.load("Joyce_data002_Lap_pca_100.npy")
graph = np.load("Joyce_data002_graph_pca_100.npy")
eivec = np.load("Joyce_data002_eivec_pca_100.npy")

print(Lap.shape)
print(graph.shape)
print(eivec.shape)

Deivec = np.diag(eivec)
print(graph.flatten()[:].shape)

#sys.exit()

# Create mesh and define function space
nx = ny = 99
#mesh = RectangleMesh(Point(0,0),Point(1,1),nx, ny)
mesh = UnitSquareMesh(nx, ny)

V = FunctionSpace(mesh, 'P', 1)
coord = V.tabulate_dof_coordinates()

def boundary(x, on_boundary):
    return on_boundary
bc = DirichletBC(V, Constant(0), boundary)

# Define initial value
u_n = interpolate(Constant(0), V)
u_n_n = interpolate(Constant(0),V) #Set equal to values of Adjacency matrix
print(u_n_n.vector()[:].shape)
print(graph.flatten()[:].shape)
u_n_n.vector()[:1000000] = graph.flatten()[:]


c = interpolate(Constant(0),V)
c.vector()[:1000000] = Deivec.flatten()[:]
#c = 5

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
dt = 1E-1
F = u*v*dx + dt*dt*c*c * dot(grad(u),grad(v))*dx - (2*u_n - u_n_n)*v*dx
a, L = lhs(F), rhs(F)


# Time-stepping
u = Function(V)
t = 0
field_val = []
num_steps = 10
for n in range(num_steps):
    
    # Update current time
    #t += dt
    # Compute solution
    solve(a == L, u, bc)

    # Update previous solution
    u_n_n.assign(u_n)
    u_n.assign(u)

    #print(u_n.vector().get_local())
    field_val.append(u_n.vector().get_local())
    

    
print(field_val[0].shape)

plt.figure()
plt.title("Field Values of Graph for 10 different time steps")
for i in range(len(field_val)):
    plt.scatter(np.arange(len(field_val[0])),field_val[i])
    plt.ylabel("Field Value")
    plt.xlabel("Nodes")


for i in range(len(field_val)):

    plt.figure()
    plt.title("Graph(T=" + str(i + 1) + ") Field Values")
    plt.scatter(coord[:,0],coord[:,1],c=field_val[i],vmin = np.min(field_val[-1]), vmax = np.max(field_val[-1]) )
    plt.xlabel(" X (spatial position)")
    plt.ylabel(" Y (spatial position)")
    plt.colorbar()