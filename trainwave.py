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
from varname import nameof
from apply_GS_wave import get_scattering

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


    def __init__(self,shape):
        # Call super constructor
        super().__init__()


        # Create mesh/function space
        if shape == 'ellipse':
            mesh = Mesh('ellipse.xml') #load ellipse mesh
        if shape == 'polygon':
            mesh = Mesh('polygon.xml')

        self.V = FunctionSpace(mesh, 'P', 1)
        mcoord = self.V.tabulate_dof_coordinates()   
            


    def solve(self, beta, gamma,omega):

        #gamma = previous time step
        #beta = wave velocity parameter
        #omega = two time steps back
        
        # Create trial and test functions spaces
        u = TrialFunction(self.V)
        v = TestFunction(self.V)
        #Define initial conditions

        #initial condition
        #u_n_n = Function(self.V)
        u_n_n_store = Function(self.V)
        #u_n_n = interpolate(Constant(5),self.V)
        #first order derivative
        #u_n = interpolate(Constant(5),self.V)

        #u_n_n.vector()[:] = torch_fenics.numpy_fenics.fenics_to_numpy(beta)[:]
        #u_n_n = interpolate(beta,self.V)
        #u_n_n.vector() = beta

        dt = 1E-1 #time step

        #Specifying PDE to be solved
        F = u*v*dx + dt*dt*beta*beta * dot(grad(u),grad(v))*dx - (2*gamma - omega)*v*dx
        a, L = lhs(F), rhs(F) 

        #Define boundary condition
        def boundary(x, on_boundary):
            return on_boundary
        bc = DirichletBC(self.V, Constant(0), boundary)

        # Solve the Wave equation
        u = Function(self.V)

        
        for n in range(1):

            # Compute solution
            solve(a == L, u, bc)

            # Update previous solution
            omega.assign(gamma)
            gamma.assign(u)

            #if t == ntimer:# only store field value at correct time index
            u_n_n_store.assign(u) #stores computed field value at current time step


        # Return the solution
        return u_n_n_store
 
    def input_templates(self):
        # Declare template for fenics to use
        return Function(self.V), Function(self.V), Function(self.V)#, Constant(0)




#Save process ID (so that I can kill later if I want)
f = open("PID.txt","w")
f.write(str(os.getpid()))
f.close()

#Wandb option
if opt.wandb == True:
    wandb.init(project="wave_debug_11_1")


#Utility functions
def graphgen(gdata,glabel,gcoord):

    "gdata = node features-> shape: [# nodes, dimension of node features]"
    "gcoord = node coordinates -> shape: [# nodes, dimension of coordinates]"
    "glabel = node labels -> shape: [# nodes, dimension of label]"

    "Construct pytorch geometric graph"

    #Defining node features
    nodefeats = torch.tensor(gdata,dtype=torch.float)
    nodelabel = torch.tensor(glabel,dtype=torch.float)
    
    #Define edge index using KNN
    edges = knn_graph(torch.tensor(nodefeats.T),k=4)
    #Define Graph
    graph = Data(x=nodefeats.T,y=nodelabel,edge_index=edges,pos=gcoord.T)

    return graph



def format_data(sol,coord,x):

    #formats data to: [ngraphs, nfeatures, nnodes]
    meshmat = np.empty((sol.shape[0]*sol.shape[1],3,sol.shape[2]))

    for i in range(sol.shape[0]): #loops through different IC's
        for j in range(sol.shape[1]): #loops through each mesh

            meshmat[j,0,:] = sol[None,None,i,j,:]
            meshmat[j,1:3,:] = coord[None,:,:]    

    np.save('GNN_input_' + x ,meshmat) 

    return meshmat



def create_graph_data(nmat,IC,time):

    tgraphs = [] #create graphs from dataset

    a = 0 #counter for timesteps
    c = 0 #counter for initial condition
    for i in range(nmat.shape[0]):
        
        tgraphs.append(graphgen(nmat[i,:,:].squeeze(),IC[c,:].T,nmat[i,1:3,:].squeeze()))
        a = a + 1

        if a == time: #if true, go to next IC
            a = 0
            c = c + 1

    return tgraphs


def train_test_split(graphs):

    "Function for shuffling and splitting data into train/test/validation"
    #NOTE nobs != ngraphs 

    ind = np.arange(nobs)*ntime
    np.random.shuffle(ind)
    trainperc = 0.6
    vadperc = 0.8

    trainbatch = []
    vadbatch = []
    testbatch = []

    #ind = random indeices for time steps starting at t=0
    #ntime = number of time stes

    for i in range(nobs):

        #Goes through each observation and gets the next t time steps defining that observation

        if i < int(trainperc*nobs):
            trainbatch.append(graphs[ ind[i]:ind[i] + ntime  ])

        if int(trainperc*nobs) <= i < int(vadperc*nobs): 
            vadbatch.append(graphs[ ind[i]:ind[i] + ntime  ])

        if i >= int(vadperc*nobs):
            testbatch.append(graphs[ ind[i]:ind[i] + ntime  ])

    return trainbatch, vadbatch, testbatch



#load data for each wave equation formulation
sol_poly_ellipse = np.load("/home/dami/Inverse_GNN/FEM_output/poly_ellipse_sol.npy")
coord_poly_ellipse = np.load("/home/dami/Inverse_GNN/FEM_output/poly_ellipse_coord.npy").T
IC_poly_ellipse = np.load("/home/dami/Inverse_GNN/FEM_output/poly_ellipse_IC.npy")

sol_gauss_ellipse = np.load("/home/dami/Inverse_GNN/FEM_output/gauss_ellipse_sol.npy")
coord_gauss_ellipse = np.load("/home/dami/Inverse_GNN/FEM_output/gauss_ellipse_coord.npy").T
IC_gauss_ellipse = np.load("/home/dami/Inverse_GNN/FEM_output/gauss_ellipse_IC.npy")

sol_poly_polygon = np.load("/home/dami/Inverse_GNN/FEM_output/poly_polygon_sol.npy")
coord_poly_polygon = np.load("/home/dami/Inverse_GNN/FEM_output/poly_polygon_coord.npy").T
IC_poly_polygon = np.load("/home/dami/Inverse_GNN/FEM_output/poly_polygon_IC.npy")

sol_gauss_polygon = np.load("/home/dami/Inverse_GNN/FEM_output/gauss_polygon_sol.npy")
coord_gauss_polygon = np.load("/home/dami/Inverse_GNN/FEM_output/gauss_polygon_coord.npy").T
IC_gauss_polygon = np.load("/home/dami/Inverse_GNN/FEM_output/gauss_polygon_IC.npy")



#get properties of mesh data
nnodes = sol_gauss_ellipse.shape[2]
nfeat = 3
ngraphs = (sol_gauss_ellipse.shape[0] * sol_gauss_ellipse.shape[1])
ntime = sol_gauss_ellipse.shape[1] 
nobs = sol_gauss_ellipse.shape[0] * 4


name_a = nameof(sol_poly_ellipse)
name_b = nameof(sol_gauss_ellipse)
name_c = nameof(sol_poly_polygon)
name_d = nameof(sol_gauss_polygon)


#New way of loading data

if opt.load_old_graphs == False:

    fdata_a = format_data(sol_poly_ellipse,coord_poly_ellipse,name_a)
    fdata_b = format_data(sol_gauss_ellipse,coord_gauss_ellipse,name_b)
    fdata_c = format_data(sol_poly_polygon,coord_poly_polygon,name_c)
    fdata_d = format_data(sol_gauss_polygon,coord_gauss_polygon,name_d)
    
    if opt.GS == True:
        fdata_a = get_scattering(fdata_a)
        fdata_b = get_scattering(fdata_b)
        fdata_c = get_scattering(fdata_c)
        fdata_d = get_scattering(fdata_d)

    pygraphs_a = create_graph_data(fdata_a,IC_poly_ellipse,ntime)
    pygraphs_b = create_graph_data(fdata_b,IC_gauss_ellipse,ntime)
    pygraphs_c = create_graph_data(fdata_c,IC_poly_polygon,ntime)
    pygraphs_d = create_graph_data(fdata_d,IC_gauss_polygon,ntime)

    pygraphs_t = [pygraphs_a,pygraphs_b,pygraphs_c,pygraphs_d]

    pygraphs = []
    #convert into singular contiguious graph
    for i in range(len(pygraphs_t)):
        for j in range(len(pygraphs_t[0])): 
            pygraphs.append(pygraphs_t[i][j]) 


    #Save graph data
    path = '/datagraphs'
    npath = cwd + path 
    with open(npath,"wb") as fg:
        pickle.dump(pygraphs_t,fg)

else:
    #load old graph data
    path = '/datagraphs'
    npath = cwd + path 
    with open(npath,"rb") as ff:
        pygraphs = pickle.load(ff)




#Split data into batches
trainloader, vadloader, testloader = train_test_split(pygraphs)


# Define Graph network
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if opt.GS == True:

    input_features = pygraphs[0].x.shape[1] #solution mesh features + coordinates + GST features
    hidden_features = 128
    out_features = 1

else:
    input_features = pygraphs[0].x.shape[1] #solution mesh features + coordinates
    hidden_features = 128
    out_features = 1

print("Input features",input_features)
model = NodeClassifier(input_features = input_features, hidden_features = hidden_features, out_features = out_features) 

#f = open("model_weights.txt","w")
#f.write(str(model.lin1.weight))
#f.close()

optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
criterion = torch.nn.MSELoss()
model = model.float().to(device)
lamda = 1 # regularization paramter




def train(ep,regp,etraindat):

    model.train()  
    loss_all = 0
    c = 0 # Counter for number of batches
    lossbatch = [] #stores loss for each batch
    
    for data in etraindat: #going through each batch of data

        nnodes = data[0].x.shape[0] #discern shape
        npredval = 30
      
        
        #store_grad = torch.empty([out.size()[0],1])
        a = 10
        for j in range(npredval): # backpropagating for predicted values at each time step within batch

        
            timeselect = a #(Selecting the 7th, 14th, and 21st time-step)

            #Generate model output at current and next time steps
            curdata = data[timeselect].to(device) 
            out = model(curdata.x, curdata.edge_index,curdata.pos)
            

            #get other solution data at other time steps
            tmintwo = data[timeselect-2].to(device)
            tminone = data[timeselect-1].to(device)
            nextpdata = torch.unsqueeze(data[timeselect+1].x[:,0].to(device),0) #Ground truth
           
        
            #Call to FEM Solver
            #time = torch.empty([1,1]) #Define time index
            #time[:,:] = torch.tensor(timeselect)
            #time = time.to(torch.double)


            FEM_input= out.T.cpu().double() 
            tminone = torch.unsqueeze(tminone.x[:,0].cpu().double(),0)
            tmintwo = torch.unsqueeze(tmintwo.x[:,0].cpu().double(),0)
            waveparam = torch.autograd.Variable(FEM_input, requires_grad=True)
            tminonefem = torch.autograd.Variable(tminone, requires_grad=True)
            tmintwofem = torch.autograd.Variable(tmintwo, requires_grad=True)

            if nnodes == 419:
                wave = Wave('ellipse') #define class instance
            if nnodes == 197:
                wave = Wave('polygon')

            pred_field = wave(waveparam,tminonefem,tmintwofem) # get prediction
        

            # Compute loss 
            loss_u = criterion( pred_field.squeeze().to(device).to(torch.double), nextpdata.to(torch.double)) 
            #print("Loss",loss_u)

            #Compute graidents 
            loss_u.backward() #compute gradients for FEM output w.r.t to input
            out.backward(waveparam.grad.T.to(device)) #compute Jacobian vector product: [GNN output w.r.t GNN input][FEM output w.r.t FEM input]
        
            
            #optimizer step
            optimizer.step() 
            optimizer.zero_grad() #zero gradients for next iterations
            a = a + 1 #update time step

    


        c = c + 1 # update count each batch
        lossbatch.append(loss_u.cpu().detach().numpy()) #stores loss for each batch


    meantrainloss = np.mean(np.array(lossbatch))
    print("Average Train Loss",meantrainloss)

    if opt.wandb == True:
        wandb.log({"Average Train Loss:":meantrainloss})
    
        
    return meantrainloss # reporting average loss per batch for each epoch



def validate(ep,regp,evadata):

 
    loss_all = 0
    c = 0 # Counter for number of batches
    lossbatch = [] #stores loss for each batch
    
    for data in evadata: #going through each batch of data

        nnodes = data[0].x.shape[0] #discern shape
        npredval = 30
      
        
        #store_grad = torch.empty([out.size()[0],1])
        a = 10
        for j in range(npredval): # backpropagating for predicted values at each time step within batch

        
            timeselect = a #(Selecting the 7th, 14th, and 21st time-step)

            #Generate model output at current and next time steps
            curdata = data[timeselect].to(device) 
            out = model(curdata.x, curdata.edge_index,curdata.pos)
            

            #get other solution data at other time steps
            tmintwo = data[timeselect-2].to(device)
            tminone = data[timeselect-1].to(device)
            nextpdata = torch.unsqueeze(data[timeselect+1].x[:,0].to(device),0) #Ground truth
           
        
            #Call to FEM Solver
            #time = torch.empty([1,1]) #Define time index
            #time[:,:] = torch.tensor(timeselect)
            #time = time.to(torch.double)


            FEM_input= out.T.cpu().double() 
            tminone = torch.unsqueeze(tminone.x[:,0].cpu().double(),0)
            tmintwo = torch.unsqueeze(tmintwo.x[:,0].cpu().double(),0)
            waveparam = torch.autograd.Variable(FEM_input, requires_grad=True)
            tminonefem = torch.autograd.Variable(tminone, requires_grad=True)
            tmintwofem = torch.autograd.Variable(tmintwo, requires_grad=True)

            if nnodes == 419:
                wave = Wave('ellipse') #define class instance
            if nnodes == 197:
                wave = Wave('polygon')
            pred_field = wave(waveparam,tminonefem,tmintwofem) # get prediction


            # Compute loss 
            loss_u = criterion( pred_field.squeeze().to(device).to(torch.double), nextpdata.to(torch.double)) 
            #print("Loss",loss_u)

            a = a + 1 #update time step

    


        c = c + 1 # update count each batch
        lossbatch.append(loss_u.cpu().detach().numpy()) #stores loss for each batch


    meanvadloss = np.mean(np.array(lossbatch))
    print("Average Validation Loss",meanvadloss)

    if opt.wandb == True:
        wandb.log({"Average Validation Loss:":meanvadloss})

    return meanvadloss #report average batch loss for each epoch






svad = []
svtrain = []

for epoch in range(1,100):

    print("Epoch",epoch)

    #FEM Loss option
    if opt.FEM_loss == True:
        fenicsopt=True
    else:
        fenicsopt=False

    #Shuffle data
        
    #Call to train/validation
    np.random.shuffle(trainloader)
    train_loss = train(epoch,lamda,trainloader)
    np.random.shuffle(vadloader)
    validate_loss = validate(epoch,lamda,vadloader)

    #f = open("model_weights.txt","a")
    #f.write(str(model.lin1.weight))
    #f.close()

    svtrain.append(train_loss)
    svad.append(validate_loss)

    if epoch % 10 == 0: #Print every 10 epochs
        print("Epoch:",epoch, " Train Loss:",train_loss," Validation Loss:",validate_loss)



#Testing model and store predictions
tloss = 0
MSE = []
pred_testfin = np.empty((nnodes, int(nobs*.2)))
gt_testfin = np.empty((nnodes, int(nobs*.2)))

pred_testfin = []
gt_testfin = []

c = 0
for data in testloader:


    timeselect = np.random.randint(0,ntime) #selecting random index to predict parameter
    inpdata = data[timeselect].to(device)
    model.eval()
    out = model(inpdata.x,inpdata.edge_index,inpdata.pos)

    #Formatting output 
    nnout = out.squeeze()
    gtout = inpdata.y

    #pred_testfin[:,c] = nnout.cpu().detach().numpy() #store predicted wave velocity for given absoervation
    #gt_testfin[:,c] = gtout.cpu().detach().numpy()

    pred_testfin.append(nnout.cpu().detach().numpy() )
    gt_testfin.append(gtout.cpu().detach().numpy() )


    tloss += criterion(nnout,gtout)

    c = c + 1 #update batch number


#Printing predictions
df_gt = pd.DataFrame(gt_testfin)
df_gt = df_gt.T
print("Ground Truth",df_gt)
df_pred = pd.DataFrame(pred_testfin)
df_pred = df_pred.T
print("Predicted",df_pred)


#Save model (locally and WandB)
path = '/Models/wave_irregularshapes.pth'
npath = cwd + path
torch.save(model.state_dict(), npath)
if opt.wandb == True:
    artifact = wandb.Artifact('wave_irregularshapes',type='model')
    artifact.add_file(npath)
    wandb.log_artifact(artifact)


#Save data (locally and WandB)
if opt.wandb == True:
    wandb.log({"pred_params":df_pred})
    wandb.log({"gt_params":df_gt})
   

