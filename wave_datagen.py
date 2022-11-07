
<<<<<<< HEAD
#import fenics
from fenics import *
=======

#import fenics
from fenics import *
from fenics_adjoint import *
>>>>>>> eb5bdba03f8b6f6ba1a0db07f67847778baac663
import torch
import torch_fenics
import torch_fenics.numpy_fenics
import numpy as np
from dolfin import *
<<<<<<< HEAD
from fenics_adjoint import *
=======
>>>>>>> eb5bdba03f8b6f6ba1a0db07f67847778baac663
import pickle
import sympy as sym
import pandas as pd
import os
<<<<<<< HEAD
import dill
=======
>>>>>>> eb5bdba03f8b6f6ba1a0db07f67847778baac663


import argparse
p = argparse.ArgumentParser()
p.add_argument('--new_data',action='store_true',help='Specify whether or not to generate new data')
<<<<<<< HEAD
p.add_argument('--genpred_field',action='store_true',help='Generate field values given predicted paramters/ICs')
opt = p.parse_args()

cwd = os.getcwd()
=======
opt = p.parse_args()

>>>>>>> eb5bdba03f8b6f6ba1a0db07f67847778baac663

class Wave(torch_fenics.FEniCSModule):
    # Construct variables which can be in the constructor


    def __init__(self):
        # Call super constructor
        super().__init__()

        #Define time settings
        self.T = 30            # final time
<<<<<<< HEAD
        self.num_steps = 30  # number of time steps
=======
        self.num_steps = 30     # number of time steps
>>>>>>> eb5bdba03f8b6f6ba1a0db07f67847778baac663
        self.dt = self.T / self.num_steps # time step size

        # Create mesh/function space
        nx = ny = 50
        mesh = RectangleMesh(Point(0,0),Point(1,1),nx,ny)


        self.V = FunctionSpace(mesh, 'P', 1)
        mcoord = self.V.tabulate_dof_coordinates()         

        #Save coordinate data
<<<<<<< HEAD
        path = '/FEM_output/fenics_coord_wave'
        npath = cwd + path
        with open(npath,"wb") as fa:
            pickle.dump(mcoord,fa)
  

    def solve(self, beta, timer):
=======
        with open("/home/dami/Inverse_GNN/FEM_output/fenics_coord_wave","wb") as fa:
            pickle.dump(mcoord,fa)
  

    def solve(self, beta):
>>>>>>> eb5bdba03f8b6f6ba1a0db07f67847778baac663
        
        # Create trial and test functions spaces
        u = TrialFunction(self.V)
        v = TestFunction(self.V)

        #Define initial conditions

        #initial field array
        u_n_n = Function(self.V)
<<<<<<< HEAD
        u_n_n_store = Function(self.V)
        #u_n_n.vector()[:] = torch_fenics.numpy_fenics.fenics_to_numpy(beta)[:]
        #u_n_n = beta
        u_n_n = interpolate(Constant(5),self.V)
=======
        u_n_n.vector()[:] = torch_fenics.numpy_fenics.fenics_to_numpy(beta)[:]
        #u_n_n = interpolate(Constant(0),self.V)
>>>>>>> eb5bdba03f8b6f6ba1a0db07f67847778baac663

        #Velocity paramter
        #c = Function(self.V)
        #c.vector()[:] = torch_fenics.numpy_fenics.fenics_to_numpy(beta)[:]
<<<<<<< HEAD
        #c = interpolate(Constant(0),self.V)
 
        #first order derivative
        u_n = interpolate(Constant(5),self.V)
        

        #Specifying PDE to be solved
        #F = u*v*dx + self.dt*self.dt*c*c * dot(grad(u),grad(v))*dx - (2*u_n - u_n_n)*v*dx
        F = u*v*dx + self.dt*self.dt*beta*beta * dot(grad(u),grad(v))*dx - (2*u_n - u_n_n)*v*dx

=======
        c = interpolate(Constant(0),self.V)
 
        #first order derivative
        u_n = interpolate(Constant(0),self.V)

        #Specifying PDE to be solved
        F = u*v*dx + self.dt*self.dt*c*c * dot(grad(u),grad(v))*dx - (2*u_n - u_n_n)*v*dx
>>>>>>> eb5bdba03f8b6f6ba1a0db07f67847778baac663
        a, L = lhs(F), rhs(F) 

        #Define boundary condition
        def boundary(x, on_boundary):
            return on_boundary
<<<<<<< HEAD

        bc = DirichletBC(self.V, Constant(0), boundary)
        #bc = DirichletBC(self.V, beta, boundary)
        #bc = DirichletBC(self.V, beta, 'on_boundary')
=======
        bc = DirichletBC(self.V, Constant(0), boundary)
>>>>>>> eb5bdba03f8b6f6ba1a0db07f67847778baac663

        # Solve the Wave equation
        u = Function(self.V)
        time_u = np.expand_dims(torch_fenics.numpy_fenics.fenics_to_numpy(u.vector()),axis=1)
        list_timevol = []
        t = 0
<<<<<<< HEAD
        ntimer = int(torch_fenics.numpy_fenics.fenics_to_numpy(timer)[:][0]) # defining time point to extract during optimizztion
        
        for n in range(self.num_steps):

            
            # Update current time
            t += self.dt
=======


        for n in range(self.num_steps):

            # Update current time
            t += self.dt

>>>>>>> eb5bdba03f8b6f6ba1a0db07f67847778baac663
            # Compute solution
            solve(a == L, u, bc)

            # Update previous solution
            u_n_n.assign(u_n)
            u_n.assign(u)

<<<<<<< HEAD
            if t == ntimer:# only store field value at correct time index
                u_n_n_store.assign(u) #stores computed field value at current time step

            print(n)
            #save solution mesh at each time step
            time_u = np.append(time_u,np.expand_dims(torch_fenics.numpy_fenics.fenics_to_numpy(u_n.vector()),axis=1),axis=1)
            
              
=======
            #save solution mesh at each time step
            time_u = np.append(time_u,np.expand_dims(torch_fenics.numpy_fenics.fenics_to_numpy(u_n.vector()),axis=1),axis=1)
            
            
        
>>>>>>> eb5bdba03f8b6f6ba1a0db07f67847778baac663
       
        #save data file
        f = open('fenics_sol.dat','ab')
        np.savetxt(f,time_u[:,1:]) 
        np.loadtxt('fenics_sol.dat') #reloading is required to preserve dimensionality - not sure why           

<<<<<<< HEAD


        # Return the solution

        return u_n_n_store

    def input_templates(self):
        # Declare template for fenics to use
        return Function(self.V), Constant(0)#Constant(np.repeat(0,2601)), Constant(0)

=======
        # Return the solution
        return u_n

    def input_templates(self):
        # Declare template for fenics to use
        return Constant(np.repeat(0,2601))
>>>>>>> eb5bdba03f8b6f6ba1a0db07f67847778baac663




<<<<<<< HEAD
if opt.genpred_field == False: #Should probably change, easy way to let script skip over part of fenics being used for trainning model

    # Generate solution data
    if opt.new_data == True: #Generate initial solution mesh

        #Define gaussian initial condition or velocity paramater 
        nnodes = 2601 #This number is known from examinning solution mesh a priori in fenics
        incondp = []
        trainsamp = 40


        #Load coordinates 
        path = '/FEM_output/fenics_coord_wave'
        npath = cwd + path
        with open(npath,"rb") as fe:
            coord = pickle.load(fe)
        print(coord.shape)
    
        alpharange = np.linspace(2,6,num=trainsamp)
        for i in range(trainsamp):
            #initmesh = np.random.normal(loc=100,scale=1.0,size=nnodes) # initial condition with 1000 random values
            val = np.exp(-alpharange[i]*coord[:,0] -alpharange[i]*coord[:,1])
            incondp.append(val)

    
        np.save("paramter_IC_values.npy",incondp)
        


        paramest = torch.tensor(np.array(incondp),dtype=torch.float64)
        path = '/FEM_output/fenics_lab_wave'
        npath = cwd + path
        with open(npath,"wb") as fc:
            pickle.dump(paramest.numpy(),fc)
        
        
        
        #Simulate PDE and save data
        wave = Wave()
        time = torch.empty([trainsamp,1],dtype=torch.float64) # pass dummy time variable
        print(paramest.size())
        u= wave(paramest,time).numpy()
        mesh = np.loadtxt('fenics_sol.dat')
        print("paramter estimation",paramest.shape)
        print("solution mesh",mesh.shape)
        os.system("mv fenics_sol.dat fenics_sol_wave.dat") # Start with new file everytime
        
        

        #CFL check
        
        """
        
        #Load meshes and compute MSE
        CFL_thirty = np.loadtxt('fenics_sol_wave_CFL_30.dat')
        CFL_threehundred = np.loadtxt('fenics_sol_wave_CFL_300.dat')
        CFL_threethousand = np.loadtxt('fenics_sol_wave_CFL_3000.dat')
        CFL_thirtythousand = np.loadtxt('fenics_sol_wave_CFL_30000.dat')

        MSE_CFL_one = np.mean(np.abs(CFL_thirty[:,15].squeeze() - CFL_threehundred[:,150].squeeze()))
        MSE_CFL_two = np.mean(np.abs(CFL_threehundred[:,150].squeeze() - CFL_threethousand[:,1500].squeeze()))
        MSE_CFL_three = np.mean(np.abs(CFL_threehundred[:,1500].squeeze() - CFL_threethousand[:,3000].squeeze()))
        print(CFL_thirty[1290:1310,15])
        print(CFL_threehundred[1290:1310,150])
        print(CFL_threethousand[1290:1310,1500])
        print(CFL_thirtythousand[1290:1310,1500])
        print(MSE_CFL_one)
        print(MSE_CFL_two)
        print(MSE_CFL_three)

        """
        
        


    if opt.new_data == False: #Generate solution mesh during optimization

        

        #GNNout = np.load("GNN_output.npy").squeeze()
        GNNout = torch.load("GNN_output.pt")
        #print("Optimization mesh shape",GNNout.shape) # this array should be (# of nodes + time index) if just predicting from one time step
        f = open("time.txt","r")
        rtime = f.read()
        f.close()
        nnodes = 2601
        #incond = GNNout[0:nnodes] # Grab all nodes for given prediction

        #if GNNout[-1] == 0:
        #    autoregressive = True
        #if GNNout[-1] == 1:
        #    autoregressive = False

        #Load coordinates 

        path = '/FEM_output/fenics_coord_wave'
        npath = cwd + path
        with open(npath,"rb") as fe:
            coord = pickle.load(fe)

        """
        #If inputing predicted paramters/initial conditions for multiple time steps:
        nnodes = 2601
        npred = int(GNNout.shape[0]/nnodes) #number of predictions from GNN
        findx = 0
        sindx = nnodes
        time = torch.empty([1,1],dtype=torch.float64)
        pred_field = torch.empty([int(nnodes*npred)])
        print(pred_field.size())
    
        for i in range(npred):

            paramest = torch.tensor(np.expand_dims(GNNout[findx:sindx],0),dtype=torch.float64,requires_grad=True)
            #Define PDE class and save data
            wave = Wave()
            time[:,:] = i
            #print(paramest.size())
            u = wave(paramest,time)
            pred_field[findx:sindx] = u.squeeze()
            findx = findx + nnodes
            sindx = sindx + nnodes
        
        """
        autoregressive = True
        if autoregressive:
            #Autoregoressive training
            #Define PDE class and format GNN output
            wave = Wave()
            time = torch.empty([1,1])
            #time[:,:] = torch.tensor(GNNout[-2]+ 1) #get field value at next time step
            time[:,:] = torch.tensor(int(rtime))

            time = time.to(torch.double)
            #incondp = torch.tensor(np.expand_dims(incond,0),dtype=torch.float64,requires_grad=True)
            #incondp = GNNout
            
            #incondpdat = torch.unsqueeze(incondp.to(torch.double).cpu()[:,0],1)
            a = GNNout[0,0]
            print(a)
            f = torch.tensor([[a]],requires_grad=True, dtype=torch.float64)
            print(f)
            print(f.size())
       
            pred_field = wave(f,time)
            J = pred_field.sum()
            J.backward()
            print(f.grad)
       
            #J.backward()
         
        

        else:
            #Neural Operator
            #Define PDE class and format GNN output
            ntimesteps = 30
            findx = 0
            sindx = nnodes
            pred_field = torch.empty([int(nnodes*ntimesteps)])
            for j in range(ntimesteps):
                wave = Wave()
                time = torch.empty([1,1])
                time[:,:] = torch.tensor(j+1) #get field values at all time steps
                time = time.to(torch.double)
                incondp = torch.tensor(np.expand_dims(incond,0),dtype=torch.float64,requires_grad=True)
                pred_field[findx:sindx] = wave(incondp,time).squeeze()
                findx = findx + nnodes
                sindx = sindx + nnodes
            

        #Save field value
        path = '/FEM_output/fenics_sol_wave_opt'
        npath = cwd + path
        with open(npath,"wb") as fe:
            pickle.dump(pred_field,fe)
        os.system("rm fenics_sol.dat")

        #os.system("mv fenics_sol.dat fenics_sol_wave_opt.dat") # Numpy array of field values

if opt.genpred_field == True:

    # Testing predicted parameters for hold out set
    file = "predwavelocityguess.npy"
    pred = np.load(file)
    print(pred.shape)

    #for i in range(pred.shape[0]):

    incond = pred
    wave = Wave()
    time = torch.empty([1,1])
    time = time.to(torch.double)
    incondp = torch.tensor(np.expand_dims(incond,0),dtype=torch.float64,requires_grad=True)
    pred_field = wave(incondp,time).squeeze()

    os.system("mv fenics_sol.dat guess_sol.dat") # Start with new file everytime
   
    #fieldvalGT = np.loadtxt("gtruth_wavelocity_solGS.dat")
    #print(fieldvalGT.shape)
=======
# Generate solution data
if opt.new_data == True: #Generate initial solution mesh


    #Define gaussian initial condition or velocity paramater 
    nnodes = 2601 #This number is known from examinning solution mesh a priori in fenics
    incondp = []
    trainsamp = 10000
    for i in range(trainsamp):
        vecrand = np.random.normal(size=nnodes)
        incondp.append(vecrand)
    paramest = torch.tensor(np.array(incondp),dtype=torch.float64)

    #Simulate PDE and save data
    wave = Wave()
    u= wave(paramest).numpy()
    mesh = np.loadtxt('fenics_sol.dat')
    print("paramter estimation",paramest.shape)
    print("solution mesh",mesh.shape)

    with open("/home/dami/Inverse_GNN/FEM_output/fenics_lab_wave","wb") as fc:
        pickle.dump(paramest.numpy(),fc)
    os.system("mv fenics_sol.dat fenics_sol_wave.dat") # Start with new file everytime


if opt.new_data == False: #Generate solution mesh during optimization

    GNNout = np.load("GNN_output.npy").squeeze()
    print("Optimization mesh shape",GNNout.shape)

    nnodes = 2601
    bs = int(GNNout.shape[0]/nnodes) #batch size
   
    findx = 0
    sindx = nnodes
    incondp = []

    for i in range(bs):
        incondp.append(GNNout[findx:sindx])
        findx = findx + nnodes
        sindx = sindx + nnodes

    paramest = torch.tensor(np.array(incondp),dtype=torch.float64)

    #Define PDE class and save data
    wave = Wave()
    u = wave(paramest).numpy()
    os.system("mv fenics_sol.dat fenics_sol_wave_opt.dat") # Start with new file everytime
>>>>>>> eb5bdba03f8b6f6ba1a0db07f67847778baac663
