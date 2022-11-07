# 


import torch
from torch import nn
#from sympy import solve_triangulated
from torch_geometric.nn import MessagePassing, TopKPooling
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops, add_self_loops
from synthetic_data import datagen#, sourced
import numpy as np
#import matplotlib.pyplot as plt
from Granola_GNN import NodeClassifier
from torch_geometric.loader import DataLoader
import sys
import pickle
import os
from torch.autograd import Variable
import wandb
import fenics
from torch_cluster import knn_graph
import pandas as pd
import scipy
from fenics import *
import torch_fenics
import torch_fenics.numpy_fenics
from dolfin import *
from fenics_adjoint import *
import sympy as sym
import dill



# Setting up arguments 
import argparse
p = argparse.ArgumentParser()
p.add_argument('--GS',action='store_true',help='Specify whether or not to include geometric scattering')
p.add_argument('--FEM_loss',action='store_true',help='Specify whether or not to include FEM loss in GNN training')
p.add_argument('--wandb',action='store_true',help='Specify wheter or not to track data on wandb')
p.add_argument('--new_run_data',action='store_true',help='Specify whether or not to generate new data')
p.add_argument('--load_old_graphs',action='store_true',help='load saved data graphs')
opt = p.parse_args()

sparsemeasure = False

cwd = os.getcwd()



class Wave(torch_fenics.FEniCSModule):
    # Construct variables which can be in the constructor


    def __init__(self):
        # Call super constructor
        super().__init__()

        #Define time settings
        self.T = 30            # final time
        self.num_steps = 30  # number of time steps
        self.dt = self.T / self.num_steps # time step size

        # Create mesh/function space
        nx = ny = 50
        mesh = RectangleMesh(Point(0,0),Point(1,1),nx,ny)


        self.V = FunctionSpace(mesh, 'P', 1)
        mcoord = self.V.tabulate_dof_coordinates()         

        #Save coordinate data
        path = '/FEM_output/fenics_coord_wave'
        npath = cwd + path
        with open(npath,"wb") as fa:
            pickle.dump(mcoord,fa)
  

    def solve(self, beta, timer):
        
        # Create trial and test functions spaces
        u = TrialFunction(self.V)
        v = TestFunction(self.V)
        #Define initial conditions

        #initial condition
        u_n_n = Function(self.V)
        u_n_n_store = Function(self.V)
        u_n_n = interpolate(Constant(5),self.V)
        #first order derivative
        u_n = interpolate(Constant(5),self.V)

        #u_n_n.vector()[:] = torch_fenics.numpy_fenics.fenics_to_numpy(beta)[:]
        #u_n_n = interpolate(beta,self.V)
        #u_n_n.vector() = beta

        #Velocity paramter
        #c = Function(self.V)
        #c.vector()[:] = torch_fenics.numpy_fenics.fenics_to_numpy(beta)[:]
        #c = interpolate(Constant(0),self.V)

        #Specifying PDE to be solved
        F = u*v*dx + self.dt*self.dt*beta*beta * dot(grad(u),grad(v))*dx - (2*u_n - u_n_n)*v*dx
        a, L = lhs(F), rhs(F) 

        #Define boundary condition
        def boundary(x, on_boundary):
            return on_boundary

        bc = DirichletBC(self.V, Constant(0), boundary)
        #bc = DirichletBC(self.V, beta, boundary)
        #bc = DirichletBC(self.V, Constant(0), 'on_boundary')

        # Solve the Wave equation
        u = Function(self.V)
        time_u = np.expand_dims(torch_fenics.numpy_fenics.fenics_to_numpy(u.vector()),axis=1)
        list_timevol = []
        t = 0
        ntimer = int(torch_fenics.numpy_fenics.fenics_to_numpy(timer)[:][0]) # defining time point to extract during optimizztion
        
        for n in range(self.num_steps):

            
            # Update current time
            t += self.dt
            # Compute solution
            solve(a == L, u, bc)

            # Update previous solution
            u_n_n.assign(u_n)
            u_n.assign(u)

            if t == ntimer:# only store field value at correct time index
                u_n_n_store.assign(u) #stores computed field value at current time step

            #save solution mesh at each time step (only useful for generating time steps)
            time_u = np.append(time_u,np.expand_dims(torch_fenics.numpy_fenics.fenics_to_numpy(u_n.vector()),axis=1),axis=1)
            
              
        #save data file
        #f = open('fenics_sol_debug.dat','ab')
        #np.savetxt(f,time_u[:,1:]) 
        #np.loadtxt('fenics_sol_debug.dat') #reloading is required to preserve dimensionality - not sure why           


        # Return the solution
        return u_n_n_store
 
    def input_templates(self):
        # Declare template for fenics to use
        return Function(self.V), Constant(0)




#Save process ID (so that I can kill later if I want)
f = open("PID.txt","w")
f.write(str(os.getpid()))
f.close()

#Wandb option
if opt.wandb == True:
    wandb.init(project="wave_equation_nFEM_model_velocity_9_27")


#Defining Pytorch geometric graph
def graphgen(gdata,glabel,gcoord):

    "gdata = node solution values; gcoord = node coordinates; glabel = node labels"

    #Defining node features
    nodefeats = torch.tensor(gdata,dtype=torch.float)
    nodelabel = torch.tensor(glabel,dtype=torch.float)
    
    #Define edge index using KNN
    edges = knn_graph(torch.tensor(gdata),k=4)

    #Define Graph
    graph = Data(x=nodefeats,y=nodelabel,edge_index=edges,pos=gcoord)

    return graph


#Define and format data properties
nnodes = 2601
nobs = 40
ntimesteps = 30
ngraphs = nobs * ntimesteps 
nfeat = 3 #solution value + number of coordinates (x,y)
nsparse = 500


if opt.new_run_data == True:

    #Generate training data
    os.system("python wave_datagen.py --new_data")

    #Loading/formatting training data
    print("Time to play:")
    print("Loading lists of data")
    data = np.loadtxt("fenics_sol_wave.dat")

    path = '/FEM_output/fenics_coord_wave'
    npath = cwd + path 
    with open(npath,"rb") as fe:
        coord = pickle.load(fe)

    path = '/FEM_output/fenics_lab_wave'
    npath = cwd + path 
    with open(npath,"rb") as ff:
        lab = pickle.load(ff)

else:

    #Loading/formatting training data
    print("Time to play:")
    print("Loading lists of data")
    data = np.loadtxt("fenics_sol_wave.dat")

    path = '/FEM_output/coordinate_data'
    npath = cwd + path 

    with open(npath,"rb") as fe:
        coord = pickle.load(fe)

    path = '/FEM_output/fenics_lab_wave'
    npath = cwd + path 

    with open(npath,"rb") as ff:
        lab = pickle.load(ff)

print("Coordinates",coord.shape)

#Converting raw data to standard format: [graphs,features,nodes]
nmat = np.empty((ngraphs,nfeat,nnodes))
print(nmat.shape)
print(data.shape)
findx = 0
sindx = nnodes

#For loop that just seperates original 2D matrix of [nodes x features] to 3D matrix of [ngraphs x features x nodes]
for i in range(nobs):

    hold = data[findx:sindx,:].T
    findx = findx + nnodes
    sindx = sindx + nnodes
    
    c = (i*ntimesteps)

    #print("Observation number",i)
    for j in range(ntimesteps):
        
        thold = np.concatenate((np.expand_dims(hold[j,:],0),coord.T),axis=0)
        nmat[c,:,:] = np.expand_dims(thold,0)
        c = c + 1

#print("Graph data shape",nmat.shape) #Should be [graphs, features, nodes]
maxind = []
minind = []
#Check graph data:
for i in range(nmat.shape[0]):
    maxind.append(np.max(nmat[i,0,:].squeeze()))
    minind.append(np.min(nmat[i,0,:].squeeze()))

print(nmat.shape)
np.save("GNN_input_graph_wave.npy",nmat)


#New way of loading data

sol = np.load("/home/dami/Inverse_GNN/FEM_output/gauss_ellipse_sol.npy")
coord = np.load("/home/dami/Inverse_GNN/FEM_output/gauss_ellipse_coord.npy")
IC = np.load("/home/dami/Inverse_GNN/FEM_output/gauss_ellipse_IC.npy")
print(sol.shape)
print(IC.shape)
print(coord.shape)
sys.exit()


"""
## Need to manually shuyffle the data, but by keep order of time in each observation 
ind = np.arange(nobs-1) #Shuffling based on first index
np.random.shuffle(ind)
shfmat = np.empty((ngraphs,3,nnodes))
nfindx = 0
nsindx = ntimesteps
for i in ind:

    findx = i*ntimesteps
    sindx = findx + ntimesteps
    shfmat[nfindx:nsindx,0,:] = nmat[findx:sindx,0,:]
    nfindx = nfindx + ntimesteps
    nsindx = nsindx + ntimesteps

shfmat[:,1:2,:] = nmat[:,1:2,:] #copy over coordinates
nmat = shfmat
np.save('GNN_input_graph_wave_shuffle.npy',nmat)

"""

#Defining sparse measurement by randomly sampling nodes
snodes = np.random.choice(np.arange(2601),size=500,replace=False)
np.save("snodesindices.npy",snodes)
time = np.arange(2,30,2) 
stime = np.sort(np.insert(time,0,[30,31]))




if opt.load_old_graphs == False:

    #Generating graphs
    if opt.GS == False: # Option if we do not want spectral information

        #ind = np.arange(ngraphs)#Shuffled indices
        #np.random.shuffle(ind)
        c = 0
        pygraphs = []
        for j in range(nobs):
            
            c = j*ntimesteps

            for i in range(ntimesteps):


                if sparsemeasure == True:

                    nmesh = nmat[:,np.array(stime,dtype=np.intp),:]
                    nnmesh = nmesh[:,:,np.array(snodes,dtype=np.intp)]
                    print("sparse measurement mesh shape:",nnmesh.shape)

                    inpnodefeat = nnmesh[ind[i],:,:].squeeze().T
                    inpnodelab = lab[ind[i],:].T.squeeze()
                    inpcoord = coord[np.array(snodes,dtype=np.intp),:]

                    print("Graph Number:",i)
                    pygraphs.append(graphgen(inpnodefeat,inpnodelab[snodes],inpcoord))

                else: 

                    inpnodefeat = nmat[c,:,:].squeeze().T
                    inpnodelab = lab[j,:].T.squeeze()
                    inpcoord = coord

                    print("Graph Number:",c)
                    pygraphs.append(graphgen(inpnodefeat,inpnodelab,inpcoord))

                c = c + 1               

        #Saving graph data

        path = '/datagraphs'
        npath = cwd + path 
        with open(npath,"wb") as fg:
            pickle.dump(pygraphs,fg)


    #Add spectral information
    if opt.GS == True: #Option if we want spectral information
        #NOTE: sparsity is handled implicitly in GST script

        # Encode geometric scattering information
        os.system("python apply_GS_wave.py")

        #Load GS information
        path = '/FEM_output/gspdata_wave'
        npath = cwd + path
        with open(npath,"rb") as fh:
            GSTmat = np.swapaxes(pickle.load(fh),1,2)

        c = 0
        pygraphs = []

        for j in range(nobs):
            c = j * ntimesteps
            for i in range(ntimesteps):

                #sparse measurment for node features is taken care of in GST
                if sparsemeasure == True:

                    inpnodefeat = GSTmat[ind[i],:,:].squeeze().T
                    inpnodelab = lab[ind[i],:].T.squeeze() 
                    inpcoord = coord[np.array(snodes,dtype=np.intp),:]

                    print("Graph Number:",i)
                    pygraphs.append(graphgen(inpnodefeat,inpnodelab,inpcoord))

                else:

                    inpnodefeat = GSTmat[c,:,:].squeeze().T
                    inpnodelab = lab[j,:].T.squeeze() #sparse measurment for labels has been taken care of 
                    inpcoord = coord

                    print("Graph Number:",c)
                    pygraphs.append(graphgen(inpnodefeat,inpnodelab,inpcoord))

                c = c + 1

        #Saving graph data

        path = '/datagraphs'
        npath = cwd + path 
        with open(npath,"wb") as fg:
            pickle.dump(pygraphs,fg)


else: #Just load old graphs

    path = '/datagraphs'
    npath = cwd + path 
    with open(npath,"rb") as ff:
        pygraphs = pickle.load(ff)

# Computations for upsampling of GNN's


#get pairwise distnace between all nodes
distmat = scipy.spatial.distance_matrix(coord,coord,p=2)
# Load sampled indeces
sampindeces = np.load("snodesindices.npy")
#Get nodes that weren't sparsely sampled 
nsampindeces = []
for i in range(2601):
    if i not in list(sampindeces):
        nsampindeces.append(i)

#Get distances between unsampled and sampled nodes
ndistmat = distmat[:,sampindeces]

#Get four nearest neighbors for each unsampled node
nearestnodesMAT = np.argsort(ndistmat,axis=1)[:,:4]# get four nearest nodes for all datapoints (sorted indeces of original sampled list)
#print(nearestnodesMAT.shape)


#Define data loaders
#train/validation/test split (60/20/20) [NOTE: Data has already been shuffled] in data loaders
print("graph data",pygraphs[0])
print("graph data shape:",len(pygraphs))

#for i in range(ngraphs):
#    print(pygraphs[i].y)

#print(pygraphs)
#sys.exit()


def shuffle_graphs(sgraph,shfindx):


    tind = np.arange(40)
    np.random.shuffle(tind)
    ind = tind*ntimesteps

    trainind = int(nobs*.6)
    vadind = int(nobs*.2)
    testind = int(nobs*.2)

    if shfindx == 0: #First epoch for shuffling, select hold out data


        traingraphs = []
        for j in range(trainind):
            traingraphs.append(pygraphs[ind[j]:ind[j]+ntimesteps])

        vadgraphs = []
        for k in range(vadind):
            j = j + 1
            vadgraphs.append(pygraphs[ind[trainind]:ind[trainind]+ntimesteps])

        testgraphs = []
        for l in range(testind):    
            j = j + 1
            testgraphs.append(pygraphs[ind[trainind]:ind[trainind]+ntimesteps])

    else:

        traingraphs = []
        for j in range(trainind):
            traingraphs.append(pygraphs[ind[j]:ind[j]+ntimesteps])

        vadgraphs = []
        for k in range(vadind):
            j = j + 1
            vadgraphs.append(pygraphs[ind[trainind]:ind[trainind]+ntimesteps])

        testgraphs = 0

    return traingraphs, vadgraphs, testgraphs


sys.exit()

#Split data into batches
bs = int(ntimesteps*1) 
trainloader = DataLoader(pygraphs[:int(round(ngraphs*.6))],batch_size=bs,shuffle=False)
vadloader = DataLoader(pygraphs[ int(round(ngraphs*.6)): int(round(ngraphs*.8))],batch_size=bs,shuffle=False)
testloader = DataLoader(pygraphs[int(round(ngraphs*.8)):],batch_size=bs,shuffle=False)


# Script for training GNN 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if opt.GS == True:

    input_features = pygraphs[0].x.shape[1] #solution mesh features + coordinates + GST features
    hidden_features = 128
    out_features = 1

else:
    input_features = pygraphs[0].x.shape[1] #solution mesh features + coordinates
    print(input_features)
    hidden_features = 128
    out_features = 1

if sparsemeasure == True:
    in_nodes = int(nsparse * bs)
    out_nodes = int(nnodes * bs)
else:
    in_nodes = int(nnodes*bs)
    out_nodes = int(nnodes*bs)

model = NodeClassifier(input_features = input_features, hidden_features = hidden_features, out_features = out_features,in_nodes = in_nodes,out_nodes = out_nodes) 

#f = open("model_weights.txt","w")
#f.write(str(model.lin1.weight))
#f.close()

optimizer = torch.optim.Adam(model.parameters(),lr=0.00001)
criterion = torch.nn.MSELoss()
model = model.float().to(device)
lamda = 1 # regularization paramter
snodes = np.load("snodesindices.npy")



def upsample_nodes(output):

    #output = one batch
    #ninit = Variable(torch.tensor(np.arange(2601)),requires_grad=True)
    tdistmat = Variable(torch.tensor(ndistmat),requires_grad=True)
    ninit = torch.tensor(np.arange(2601))
    for i in range(2601):

        if i not in list(sampindeces):
            x = nearestnodesMAT[i,:] # get indeces of original sampled indeces vector
            sol = output[x]
            loc = tdistmat[i,x]
            ninit[i] = torch.dot(sol,loc.to(device).to(torch.float))
        else:
            ninit[i] = output[np.argwhere(sampindeces == i)]  

    return ninit



def train(fenicsopt,ep,regp,etraindat):

    model.train()  
    loss_all = 0
    c = 0 # Counter for number of batches
    lossbatch = [] #stores loss for each batch
    for data in etraindat: #NOTE: not using a data loader

        """
        #Generate model output
        data = data.to(device)
        print(data)
        sys.exit()
        model.train()   
        optimizer.zero_grad()
        #print(pygraphs[0])
        out = model(data.x, data.edge_index,data.pos) 
        #sys.exit()
        

        #Formatting output 
    
        nnout = out.squeeze()
   
        #print(dir(nnout))
        #sys.exit()
        #nnout = bsol.to(device).to(torch.double).squeeze()
        gtout = data.y

        print("Training Observation Number:",c)
        """

        if opt.FEM_loss: #and ep % 20 == 0 and ep > 700: #adjust weights based on solution mesh loss every 10 epochs

        
            
            autoregressive = True
            if autoregressive: 
                #npredval = int(ntimesteps-1)
                npredval = 3
                a = 0
            else:
                npredval = ntimesteps
                a = 1

            
            loss_p_batch = []
 
          
            #store_grad = torch.empty([out.size()[0],1])
            a = 1
            for j in range(npredval): # backpropagating for predicted values at each time step

            
                timeselect = a*7 #(Selecting the 7th, 14th, and 21st time-step)

                #Generate model output at current and next time steps
                curdata = data[timeselect].to(device)
                datanext = data[timeselect+1].to(device)
                out = model(curdata.x, curdata.edge_index,curdata.pos)
                print(out)
                sys.exit()
                nextpdata = model(datanext.x, datanext.edge_index,datanext.pos)  

            
                #Call to FEM Solver
                wave = Wave() #define class instance
                time = torch.empty([1,1]) #Define time index
                time[:,:] = torch.tensor(timeselect)
                time = time.to(torch.double)
                FEM_input= out.T.cpu().double()
                f1 = torch.autograd.Variable(FEM_input, requires_grad=True)
                pred_field = wave(f1,time)
          

                # Compute loss 
                #print(pred_field.shape)
                #print(nextpdata.shape)


                varonesum = torch.sum(pred_field.T.squeeze().to(device).to(torch.double))
                vartwosum = torch.sum(nextpdata[:,0].squeeze().to(torch.double) )
                varonesqrsum = torch.sum(pred_field.T.squeeze().to(device).to(torch.double)**2)
                vartwosqrsum = torch.sum(nextpdata[:,0].squeeze().to(torch.double)**2)
                varprod = torch.sum(pred_field.T.squeeze().to(device).to(torch.double) * nextpdata[:,0].squeeze().to(torch.double))
                n = pred_field.T.squeeze().to(device).to(torch.double).size()[0]
                
                num = n*varprod - (varonesum*vartwosum)
                dnm = torch.sqrt( (n*varonesqrsum - varonesum**2)*(n*vartwosqrsum - vartwosum**2)  )
                loss_u = 1 - torch.abs(num/dnm)
                #print(loss_u)
                
                
                #loss_u = criterion( pred_field.squeeze().to(device).to(torch.double), nextpdata[:,0].to(torch.double)) 
                #num =  torch.sum(  pred_field.T.squeeze().to(device).to(torch.double) - torch.mean(pred_field.T.squeeze().to(device).to(torch.double))  *  nextpdata[:,0].squeeze().to(torch.double) - torch.mean(nextpdata[:,0].squeeze().to(torch.double))    )
                #dnm = torch.sqrt(  torch.sum((pred_field.T.squeeze().to(device).to(torch.double) - torch.mean(pred_field.T.squeeze().to(device).to(torch.double)))**2) * torch.sum(  (nextpdata[:,0].squeeze().to(torch.double) - torch.mean(nextpdata[:,0].squeeze().to(torch.double)))**2  ) )
                #tloss = num/dnm
                #loss_u = 1 - torch.abs(tloss)
                #print(loss_u)


                #Compute graidents 
                loss_u.backward() #compute gradients for FEM output w.r.t to input
                out.backward(f1.grad.T.to(device)) #compute Jacobian vector product: [GNN outpu w.r.t GNN input][FEM output w.r.t FEM input]
                
                #optimizer step
                optimizer.step() 
                optimizer.zero_grad() #zero gradients for next iterations
                a = a + 1 #update time step

    
        else:

            #Computing loss
            loss = criterion(nnout, gtout)
            loss_all += loss.item() # adds loss for each batch
            print(model.lin1.weight.grad)
            loss.backward()
            print(model.lin1.weight.grad)
            optimizer.step()
            sys.exit()

        c = c + 1 # update count each batch
        lossbatch.append(loss_u.cpu().detach().numpy()) #stores loss for each batch


    meantrainloss = np.mean(np.array(lossbatch))
    print("Average Train Loss",meantrainloss)

    if opt.wandb == True:
        wandb.log({"Average Train Loss:":meantrainloss})
    
        
    return meantrainloss # reporting average loss per batch for each epoch



def validate(fenicsopt,ep,regp,evadata):

    loss_all = 0
    c = 0# counter for number of batches
    lossbatch = []
    for data in evadata:



        #print("Validation Observation Number:",c)


        if opt.FEM_loss: #and ep % 20 == 0 and ep > 700: #adjust weights based on solution mesh loss every 10 epochs


            autoregressive = True
            if autoregressive: 
                #npredval = int(ntimesteps-1)
                npredval = 3
                a = 0
            else:
                npredval = ntimesteps
                a = 1

            
            loss_p_batch = []
            npredval = 1

            timeselect = np.random.randint(1,28) #selecting random index to put through FEM solver
            rfindx = int(nnodes*timeselect)
            rsindx = int(nnodes*(timeselect+1))

            a = 1
            for j in range(npredval):
        
                timeselect = a*7
                #Generate model output
                curdata = data[timeselect].to(device)
                datanext = data[timeselect+1].to(device)
                out = model(curdata.x, curdata.edge_index,curdata.pos)
                nextpdata = model(datanext.x, datanext.edge_index,datanext.pos)  
            

                #Call to FEM solver
                wave = Wave() #define class instance
                time = torch.empty([1,1]) #Define time index
                time[:,:] = torch.tensor(timeselect)
                time = time.to(torch.double)
                FEM_input = out.T.cpu().double()
                f1 = torch.autograd.Variable(FEM_input, requires_grad=True)
                pred_field = wave(f1,time)
                
                #extracting loss and computing gradients
                #loss_u = criterion( pred_field.squeeze().to(device).to(torch.double), nextpdata[:,0].to(torch.double))  

                #Gather components for coefficent computation

                varonesum = torch.sum(pred_field.T.squeeze().to(device).to(torch.double))
                vartwosum = torch.sum(nextpdata[:,0].squeeze().to(torch.double) )
                varonesqrsum = torch.sum(pred_field.T.squeeze().to(device).to(torch.double)**2)
                vartwosqrsum = torch.sum(nextpdata[:,0].squeeze().to(torch.double)**2)
                varprod = torch.sum(pred_field.T.squeeze().to(device).to(torch.double) * nextpdata[:,0].squeeze().to(torch.double))
                n = pred_field.T.squeeze().to(device).to(torch.double).size()[0]
                
                num = n*varprod - (varonesum*vartwosum)
                dnm = torch.sqrt( (n*varonesqrsum - varonesum**2)*(n*vartwosqrsum - vartwosum**2)  )
                loss_u = 1 - torch.abs(num/dnm)
                #print(loss_u)


                #num =  torch.sum(  (pred_field.T.squeeze().to(device).to(torch.double) - torch.mean(pred_field.T.squeeze().to(device).to(torch.double)))  *  (nextpdata[:,0].squeeze().to(torch.double) - torch.mean(nextpdata[:,0].squeeze().to(torch.double)))    )
                #dnm = torch.sqrt(  torch.sum((pred_field.T.squeeze().to(device).to(torch.double) - torch.mean(pred_field.T.squeeze().to(device).to(torch.double)))**2) * torch.sum(  (nextpdata[:,0].squeeze().to(torch.double) - torch.mean(nextpdata[:,0].squeeze().to(torch.double)))**2  ) )
                #tloss = num/dnm
                #loss_u = 1 - torch.abs(tloss)

                loss_u.backward()
                out.backward(f1.grad.T.to(device))
                optimizer.step() 
                optimizer.zero_grad()
                a = a + 1



        else:
            # Computing loss
            nnout = out.squeeze().to(torch.double)
            ny = data.y.to(torch.double)
            loss = criterion( nnout, ny)
            loss_all += loss.item() # adds loss for each batch
            print(model1.lin.weight.grad)
            loss.backward()
            print(model1.lin.weight.grad)
            optimizer.step()

        lossbatch.append(loss_u.cpu().detach().numpy())

        c = c + 1 #update count for each batch
    
    print("Average Validation Loss",np.mean(np.array(lossbatch)))

    meanvadloss = np.mean(np.array(lossbatch))
    print("Average Vad Loss",meanvadloss)

    if opt.wandb == True:
        wandb.log({"Average Vad Loss:":meanvadloss})

    return loss_all/c #report average batch loss for each epoch


svad = []
svtrain = []

traindat, vadat, testdat = shuffle_graphs(pygraphs,0)
for epoch in range(1,50):

    print("Epoch",epoch)

    #FEM Loss option
    if opt.FEM_loss == True:
        fenicsopt=True
    else:
        fenicsopt=False

    #Shuffle data

    if epoch > 1:
        traindat, vadat, garbage = shuffle_graphs(pygraphs,1) #Don't want to shuffle test data(data leakage)
        
    #Call to train/validation
    train_loss = train(fenicsopt,epoch,lamda,traindat)
    validate_loss = validate(fenicsopt,epoch,lamda,vadat)

    #f = open("model_weights.txt","a")
    #f.write(str(model.lin1.weight))
    #f.close()

    svtrain.append(train_loss)
    svad.append(validate_loss)

    if epoch % 10 == 0: #Print every 10 epochs
        print("Epoch:",epoch, " Train Loss:",train_loss," Validation Loss:",validate_loss)

        


#Testing model with MSE
tloss = 0
MSE = []
pred_testfin = np.empty((nnodes, int(nobs*.2)))
gt_testfin = np.empty((nnodes, int(nobs*.2)))


c = 0
for data in testdat:


    timeselect = np.random.randint(0,28) #selecting random index to store paramters
    inpdata = data[timeselect].to(device)
    model.eval()
    out = model(inpdata.x,inpdata.edge_index,inpdata.pos)

    #Formatting output 
    nnout = out.squeeze()
    gtout = inpdata.y

    pred_testfin[:,c] = nnout.cpu().detach().numpy()# for given observation, randomly select time step and store its predicted value
    gt_testfin[:,c] = gtout.cpu().detach().numpy()
    tloss += criterion(nnout,gtout)

    c = c + 1 #update batch number


#Printing predictions
df_gt = pd.DataFrame(gt_testfin)
print("Ground Truth",df_gt)
df_pred = pd.DataFrame(pred_testfin)
print("Predicted",df_pred)


#Save model (locally and WandB)
path = '/Models/wave_velocity_50ep_9_27_he_128.pth'
npath = cwd + path
torch.save(model.state_dict(), npath)
if opt.wandb == True:
    artifact = wandb.Artifact('wave_velocity_50ep_9_27_GS_he128',type='model')
    artifact.add_file(npath)
    wandb.log_artifact(artifact)


#Save data (locally and WandB)
if opt.wandb == True:
    wandb.log({"pred_params":df_pred})
    wandb.log({"gt_params":df_gt})
   

