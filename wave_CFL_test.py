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

#p = argparse.ArgumentParser()
#p.add_argument('--gendata',action='store_true',help='Specify whether or not to generate new data')
#p.add_argument('--analyzedata',action='store_true',help='Call to analyze output of network')
#opt = p.parse_args()


gendata = False
analyzedata=True

if gendata == True:

    num_param = 10
    num_steps = 50 
    sol_time = []
    incondpload = np.load("gauss_polygon_IC.npy")
    dt= 1E-7
    num_coord_ellipse = 197
    gauss_ellipse_sol = np.empty((num_param,num_steps,num_coord_ellipse))

    for i in range(num_param):

        incondp = incondpload[i]

        #T = 5            # final time
        #num_steps = num_steps -1   # number of time steps
        #dt = T / num_steps # time step size
        dt = 1E-1


        # Create mesh and define function space
        #nx = ny = 50
        #mesh = RectangleMesh(Point(0,0),Point(1,1),nx, ny)
        domain_vertices = [Point(0.0,0.7),Point(0.0,-0.35),Point(1.0,-1.0),Point(2.5,0.0),Point(1.0,1.0)]#,Point(0.7,-1.0)]
        domain = Polygon(domain_vertices)
        mesh = generate_mesh(domain,10)
        File('polygon.xml') << mesh
        mesh = Mesh('polygon.xml')
        
        domain = Ellipse(Point(0,0),2,3)
        mesh = generate_mesh(domain,16)
        File('ellipse.xml') << mesh
        mesh = Mesh('ellipse.xml')



        V = FunctionSpace(mesh, 'P', 1)
        coord = V.tabulate_dof_coordinates()

        def boundary(x, on_boundary):
            return on_boundary
        bc = DirichletBC(V, Constant(0), boundary)

        # Define initial value
        u_n = interpolate(Constant(10), V)
        u_n_n = interpolate(Constant(5),V)
        #beta = 5
        
        
        c = interpolate(Constant(0),V)
        c.vector()[:] = incondp.squeeze()[:]
        #c = 5

        # Define variational problem
        u = TrialFunction(V)
        v = TestFunction(V)
        F = u*v*dx + dt*dt*c*c * dot(grad(u),grad(v))*dx - (2*u_n - u_n_n)*v*dx
        a, L = lhs(F), rhs(F)


        # Time-stepping
        u = Function(V)
        t = 0
        field_val = []
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
            
        field_val = np.array(field_val)
        gauss_ellipse_sol[i,:,:] = field_val
        


    #print(gauss_ellipse_sol.shape)
    #print(gauss_ellipse_sol[8,40,:])
    #np.save("poly_polygon_sol.npy",gauss_ellipse_sol)


    plt.figure()
    plt.title(" Gaussian IC 1")
    plt.scatter(coord[:,0],coord[:,1],c = incondpload[1])
    plt.colorbar()

    plt.figure()
    plt.title("Gaussian IC 4")
    plt.scatter(coord[:,0],coord[:,1],c = incondpload[4])
    plt.colorbar()

    plt.figure()
    plt.title("Gaussian IC 7")
    plt.scatter(coord[:,0],coord[:,1],c = incondpload[7])
    plt.colorbar()

    plt.figure()
    plt.title("Gaussian Field value (T=25,IC=1)")
    plt.scatter(coord[:,0],coord[:,1],c = gauss_ellipse_sol[1,24,:].squeeze())
    plt.colorbar()

    plt.figure()
    plt.title("IC 4")
    plt.title("Gaussian Field value (T=25,IC=4)")
    plt.scatter(coord[:,0],coord[:,1],c = gauss_ellipse_sol[4,24,:].squeeze())
    plt.colorbar()

    plt.figure()
    plt.title("Gaussian Field value (T=25,IC=7)")
    plt.scatter(coord[:,0],coord[:,1],c = gauss_ellipse_sol[7,24,:].squeeze())
    plt.colorbar()

    # Normalize color scale

    plt.figure()
    plt.title(" Gaussian IC 1 (Same scale) ")
    plt.scatter(coord[:,0],coord[:,1],c = incondpload[1],vmin = np.min(incondpload[1]), vmax = np.max(incondpload[1]) )
    plt.colorbar()

    plt.figure()
    plt.title("Gaussian IC 4 (Same scale) ")
    plt.scatter(coord[:,0],coord[:,1],c = incondpload[4],vmin=np.min(incondpload[1]) , vmax = np.max(incondpload[1]))
    plt.colorbar()

    plt.figure()
    plt.title("Gaussian IC 7 (Same scale) ")
    plt.scatter(coord[:,0],coord[:,1],c = incondpload[7], vmin = np.min(incondpload[1]) , vmax = np.max(incondpload[1]))
    plt.colorbar()

    plt.figure()
    plt.title("Gaussian Field value (T=25,IC=1) (Same scale) ")
    plt.scatter(coord[:,0],coord[:,1],c = gauss_ellipse_sol[1,24,:].squeeze(), vmin = np.min(gauss_ellipse_sol[1,24,:].squeeze()), vmax = np.max(gauss_ellipse_sol[1,24,:].squeeze()) )
    plt.colorbar()

    plt.figure()
    plt.title("IC 4")
    plt.title("Gaussian Field value (T=25,IC=4) (Same scale) ")
    plt.scatter(coord[:,0],coord[:,1],c = gauss_ellipse_sol[4,24,:].squeeze(), vmin = np.min(gauss_ellipse_sol[1,24,:].squeeze()),vmax = np.max(gauss_ellipse_sol[1,24,:].squeeze()) )
    plt.colorbar()

    plt.figure()
    plt.title("Gaussian Field value (T=25,IC=7) (Same scale) ")
    plt.scatter(coord[:,0],coord[:,1],c = gauss_ellipse_sol[7,24,:].squeeze(), vmin = np.min(gauss_ellipse_sol[1,24,:].squeeze()), vmax = np.max(gauss_ellipse_sol[1,24,:].squeeze()) )
    plt.colorbar()



    """
    #CFL Check
    vecdiff = np.abs(sol_time[-1] - sol_time[-2])
    vecmean = np.abs(np.mean(sol_time[-1]))
    relative = np.linalg.norm(vecdiff,2)/vecmean
    print("Percent Error",relative*100)
    print(sol_time[-1])

    """


    #Number of different wave velocity parameters
    trainsamp = 10 
    alpharange = np.random.normal(1,1,trainsamp)
    betarange = np.random.normal(1,1,trainsamp)
    gammarange = np.random.normal(80,20,trainsamp)


    #Gaussians
    gauss = 1
    if gauss == 0:
        incondp = []
        for i in range(trainsamp):

            pcenterx = np.random.normal(0,np.max(coord[:,0]),1)
            pcentery = np.random.normal(0,np.max(coord[:,1]),1)
            ncoordx = np.abs(coord[:,0] - pcenterx)
            ncoordy = np.abs(coord[:,1] - pcentery)
            ncoord = np.concatenate((ncoordx.reshape(coord.shape[0],1),ncoordy.reshape(coord.shape[0],1)),axis=1)

            val = gammarange[i]*np.exp(-alpharange[i]*(ncoord[:,0] + ncoord[:,1]))
            incondp.append(np.abs(val))

    #Polynomials
    else:
        incondp = []
        for i in range(trainsamp):
            #initmesh = np.random.normal(loc=100,scale=1.0,size=nnodes) # initial condition with 1000 random values

            pcenterx = np.random.normal(0,np.max(coord[:,0]),1)
            pcentery = np.random.normal(0,np.max(coord[:,1]),1)
            ncoordx = np.abs(coord[:,0] - pcenterx)
            ncoordy = np.abs(coord[:,1] - pcentery)
            ncoord = np.concatenate((ncoordx.reshape(coord.shape[0],1),ncoordy.reshape(coord.shape[0],1)),axis=1)


            val = (1 + (alpharange[i]*ncoord[:,0]**2) + (betarange[i]*ncoord[:,1]**2))
            incondp.append(np.abs(val))


#print("Difference")
#print(incondp[0])
#print(np.max(incondp[1]))
#print(np.linalg.norm(np.abs(incondp[0] - incondp[2]),2)/np.mean(incondp[2]))
#print(alpharange)


#print(sol_time[-1])
#np.save("poly_polygon_IC.npy",np.array(incondp))
#np.save("poly_polygon_coord.npy",coord)
#np.save("Gauss_ellipse_sol.npy",np.array(sol_time[-1]))


#Define coordinate mesh grid (for visualization)
#x = np.linspace(np.min(coord[:,0]),np.max(coord[:,0]),25)
#y = np.linspace(np.min(coord[:,1]),np.max(coord[:,1]),25)
#xx, yy = np.meshgrid(x,y)



if analyzedata == True:


    #load data from wandb
    NNoutputpred = pd.read_csv('/Users/oluwadamilolafasina/Downloads/wandb_export_2022-11-01T11_27_37.166-07_00.csv').to_numpy()
    NNoutputgt = pd.read_csv('/Users/oluwadamilolafasina/Downloads/wandb_export_2022-11-01T11_25_37.109-07_00.csv').to_numpy()
    waveparamgt = NNoutputgt[:,0]  
    waveparampred = NNoutputpred[:,0]  
    print(waveparamgt)
    print(waveparampred)


    #Predicted


    #run simulation
    dt = 1E-1
    num_steps = 50
    # Create mesh and define function space
    #nx = ny = 50
    #mesh = RectangleMesh(Point(0,0),Point(1,1),nx, ny)

    if np.sum(np.isnan(waveparampred)) > 0:

        domain_vertices = [Point(0.0,0.7),Point(0.0,-0.35),Point(1.0,-1.0),Point(2.5,0.0),Point(1.0,1.0)]#,Point(0.7,-1.0)]
        domain = Polygon(domain_vertices)
        mesh = generate_mesh(domain,10)
        File('polygon.xml') << mesh
        mesh = Mesh('polygon.xml')
        wave = waveparampred[:197]

        V = FunctionSpace(mesh, 'P', 1)
        coord = V.tabulate_dof_coordinates()
       


    if np.sum(np.isnan(waveparampred)) == 0:

        domain = Ellipse(Point(0,0),2,3)
        mesh = generate_mesh(domain,16)
        File('ellipse.xml') << mesh
        mesh = Mesh('ellipse.xml')

        V = FunctionSpace(mesh, 'P', 1)
        coord = V.tabulate_dof_coordinates()

      


    V = FunctionSpace(mesh, 'P', 1)
    coord = V.tabulate_dof_coordinates()

    def boundary(x, on_boundary):
        return on_boundary
    bc = DirichletBC(V, Constant(0), boundary)

    # Define initial value
    u_n = interpolate(Constant(10), V)
    u_n_n = interpolate(Constant(5),V)
    #beta = 5


    c = interpolate(Constant(0),V)
    c.vector()[:] = wave.squeeze()[:]
    #c = 5

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    F = u*v*dx + dt*dt*c*c * dot(grad(u),grad(v))*dx - (2*u_n - u_n_n)*v*dx
    a, L = lhs(F), rhs(F)


    # Time-stepping
    u = Function(V)
    t = 0
    field_val = []
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
        
    pred_field = np.array(field_val)



    #Ground Truth

    #run simulation
    dt = 1E-1
    num_steps = 50
    # Create mesh and define function space
    #nx = ny = 50
    #mesh = RectangleMesh(Point(0,0),Point(1,1),nx, ny)

    if np.sum(np.isnan(waveparampred)) > 0:

        domain_vertices = [Point(0.0,0.7),Point(0.0,-0.35),Point(1.0,-1.0),Point(2.5,0.0),Point(1.0,1.0)]#,Point(0.7,-1.0)]
        domain = Polygon(domain_vertices)
        mesh = generate_mesh(domain,10)
        File('polygon.xml') << mesh
        mesh = Mesh('polygon.xml')
        wave = waveparampred[:197]

        V = FunctionSpace(mesh, 'P', 1)
        coord = V.tabulate_dof_coordinates()
       


    if np.sum(np.isnan(waveparampred)) == 0:

        domain = Ellipse(Point(0,0),2,3)
        mesh = generate_mesh(domain,16)
        File('ellipse.xml') << mesh
        mesh = Mesh('ellipse.xml')

        V = FunctionSpace(mesh, 'P', 1)
        coord = V.tabulate_dof_coordinates()

      


    V = FunctionSpace(mesh, 'P', 1)
    coord = V.tabulate_dof_coordinates()

    def boundary(x, on_boundary):
        return on_boundary
    bc = DirichletBC(V, Constant(0), boundary)

    # Define initial value
    u_n = interpolate(Constant(10), V)
    u_n_n = interpolate(Constant(5),V)
    #beta = 5


    c = interpolate(Constant(0),V)
    c.vector()[:] = wave.squeeze()[:]
    #c = 5

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    F = u*v*dx + dt*dt*c*c * dot(grad(u),grad(v))*dx - (2*u_n - u_n_n)*v*dx
    a, L = lhs(F), rhs(F)


    # Time-stepping
    u = Function(V)
    t = 0
    field_val = []
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
        
    gt_field = np.array(field_val)


plt.figure()
plt.title(" Predicted Wave Velocity")
plt.scatter(coord[:,0],coord[:,1],c = waveparamgt[:197])#,vmin = np.min(incondpload[1]), vmax = np.max(incondpload[1]) )
plt.colorbar()

plt.figure()
plt.title(" Ground Truth Wave Velocity ")
plt.scatter(coord[:,0],coord[:,1],c = waveparampred[:197])#,vmin = np.min(incondpload[1]), vmax = np.max(incondpload[1]) )
plt.colorbar()


plt.figure()
plt.title(" Predicted Field Value (T=25)")
plt.scatter(coord[:,0],coord[:,1],c = pred_field[2,:])#,vmin = np.min(incondpload[1]), vmax = np.max(incondpload[1]) )
plt.colorbar()

plt.figure()
plt.title(" Ground Truth Field Value (T=25)")
plt.scatter(coord[:,0],coord[:,1],c = gt_field[2,:])#,vmin = np.min(incondpload[1]), vmax = np.max(incondpload[1]) )
plt.colorbar()

