

#import fenics
from fenics import *
from fenics_adjoint import *
import torch
import torch_fenics
import numpy as np
from dolfin import *
import pickle


import argparse
p = argparse.ArgumentParser()
p.add_argument('--new_data',action='store_true',help='Specify whether or not to generate new data')
opt = p.parse_args()


class Heat(torch_fenics.FEniCSModule):
    # Construct variables which can be in the constructor
    def __init__(self):
        # Call super constructor
        super().__init__()

        #Define constants for equation

        self.T = 2.0            # final time
        self.num_steps = 10     # number of time steps
        self.dt = self.T / self.num_steps # time step size
        #alpha = 3          # parameter alpha
        #beta = 1.2         # parameter beta

        # Create mesh/function space
        nx = ny = 8
        mesh = UnitSquareMesh(nx,ny)
<<<<<<< HEAD
        print(mesh)
        sys.exit()
=======
>>>>>>> eb5bdba03f8b6f6ba1a0db07f67847778baac663
        self.V = FunctionSpace(mesh, 'P', 1)
        mcoord = self.V.tabulate_dof_coordinates()         

        #Save coordinate data
        with open("/home/dami/Inverse_GNN/FEM_output/fenics_coord","wb") as fa:
            pickle.dump(mcoord,fa)




    def solve(self, beta, alpha):


        # Create trial and test functions + define variation problem
        u = TrialFunction(self.V)
        v = TestFunction(self.V)

        #Define boundary condition and variational form
        f = Constant(beta - 2 - 2*alpha) # source term
        u_D = Expression('1 + x[0]*x[0] + alpha*x[1]*x[1] + beta*t',
                degree=2, alpha=alpha, beta=beta, t=0)

        u_n = interpolate(u_D, self.V)

        #Specifying PDE to be solved
        F = u*v*dx + self.dt*dot(grad(u), grad(v))*dx - (u_n + self.dt*f)*v*dx
        a, L = lhs(F), rhs(F) 

        

        def boundary(x, on_boundary):
            return on_boundary

        bc = DirichletBC(self.V, u_D, boundary)

        # Solve the Heat equation
        u = Function(self.V)
        t = 0
        
        for n in range(self.num_steps):

            # Update current time
            t += self.dt
            u_D.t = t

            # Compute solution
            solve(a == L, u, bc)

            # Update previous solution
            u_n.assign(u)

        # Return the solution
        return u_n

    def input_templates(self):
        # Declare templates for the inputs to Poisson.solve
        return Constant(0), Constant(0)

    


"""
# Testing 
heat = Heat()
N = 10
cx = torch.rand(N, 1, requires_grad=True, dtype=torch.float64)
cy = torch.rand(N, 1, requires_grad=True, dtype=torch.float64)

cxx = torch.tensor([[1.2],[4],[1.2],[1.2],[1.2]],requires_grad=True,dtype=torch.float64)
cyy = torch.tensor([[3],[3],[3],[3],[3]],requires_grad=True,dtype=torch.float64)

print("rand function",cx)
print("cx function",cxx)

print("size check",cxx.size())
u = heat(cxx,cyy)
print(np.shape(u))
print(u[2])
j = u.sum()
print(j.backward())

#Note; u returns matrix of size [num paramters x solution grid]
"""


# Generate solution data
if opt.new_data == True: #Generate initial solution mesh

    #Define paramter space
    ialph = np.repeat(1.3,500)
    ibeta = np.arange(500)
    palph = [ [x] for x in ialph]
    pbeta = [ [x] for x in ibeta]
    talph = torch.squeeze(torch.tensor([palph],dtype=torch.float64),0)#,requires_grad=True,dtype=torch.float64)
    tbeta = torch.squeeze(torch.tensor([pbeta],dtype=torch.float64),0)#,requires_grad=True,dtype=torch.float64)
<<<<<<< HEAD
    print(tbeta.size())
    print(type(tbeta))
=======

>>>>>>> eb5bdba03f8b6f6ba1a0db07f67847778baac663

    #Define PDE class and save data
    heat = Heat()
    u = heat(talph,tbeta).numpy()
    with open("/home/dami/Inverse_GNN/FEM_output/fenics_sol","wb") as fb:
        pickle.dump(u,fb)
    with open("/home/dami/Inverse_GNN/FEM_output/fenics_lab","wb") as fc:
        pickle.dump(tbeta.numpy(),fc)


if opt.new_data == False: #Generate solution mesh during optimization
    GNNout = np.load("GNN_output.npy")
    # Get graph-averaged paramter value
    nGNNout = []
    bs = 100 # batch size
    nnodes = 81
    for i in range(bs):
        a = i*nnodes
        b = (i+1)*nnodes
        nGNNout.append(np.mean(GNNout[a:a+b]))
    

    #Define paramter space
    ibeta = np.arange(bs)
    palph = [ [x] for x in nGNNout]
    pbeta = [ [x] for x in ibeta]
    talph = torch.squeeze(torch.tensor([palph],requires_grad = True, dtype=torch.float64),0)
    tbeta = torch.squeeze(torch.tensor([pbeta],requires_grad = True, dtype=torch.float64),0)
<<<<<<< HEAD
    print(type(tbeta))
    print(tbeta.size())
    #Define PDE class and save data
    heat = Heat()
    u = heat(talph,tbeta)
    print("Solution size:",u.size())
    print(u)
    J = u.sum()
    print(J.backward())
    with open("/home/dami/Inverse_GNN/FEM_output/fenics_optsol","wb") as fb:
        pickle.dump(u,fb)




    
=======


    #Define PDE class and save data
    heat = Heat()
    u = heat(talph,tbeta)
    with open("/home/dami/Inverse_GNN/FEM_output/fenics_optsol","wb") as fb:
        pickle.dump(u,fb)
>>>>>>> eb5bdba03f8b6f6ba1a0db07f67847778baac663
