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
<<<<<<< HEAD
import scipy
=======

>>>>>>> eb5bdba03f8b6f6ba1a0db07f67847778baac663

# Setting up arguments 
import argparse
p = argparse.ArgumentParser()
p.add_argument('--GS',action='store_true',help='Specify whether or not to include geometric scattering')
p.add_argument('--FEM_loss',action='store_true',help='Specify whether or not to include FEM loss in GNN training')
p.add_argument('--wandb',action='store_true',help='Specify wheter or not to track data on wandb')
p.add_argument('--new_run_data',action='store_true',help='Specify whether or not to generate new data')
p.add_argument('--load_old_graphs',action='store_true',help='load saved data graphs')
opt = p.parse_args()


#Wandb option
if opt.wandb == True:
    wandb.init(project="wave_equation_exp")


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
<<<<<<< HEAD
ngraphs = 1000
=======
ngraphs = 10000
>>>>>>> eb5bdba03f8b6f6ba1a0db07f67847778baac663
nfeat = 32 #number of time steps + number of coordinates (x,y)
nsparse = 500


if opt.new_run_data == True:
    #Generate training data
    os.system("python wave_datagen.py --new_data")

    #Loading/formatting training data
    print("Time to play:")
    print("Loading lists of data")
    data = np.loadtxt("fenics_sol_wave.dat")
    with open("/home/dami/Inverse_GNN/FEM_output/fenics_coord_wave","rb") as fe:
        coord = pickle.load(fe)

    with open("/home/dami/Inverse_GNN/FEM_output/fenics_lab_wave","rb") as ff:
        lab = pickle.load(ff)

else:

    #Loading/formatting training data
    print("Time to play:")
    print("Loading lists of data")
    data = np.loadtxt("fenics_sol_wave.dat")
    with open("/home/dami/Inverse_GNN/FEM_output/fenics_coord_wave","rb") as fe:
        coord = pickle.load(fe)

    with open("/home/dami/Inverse_GNN/FEM_output/fenics_lab_wave","rb") as ff:
        lab = pickle.load(ff)



#Converting raw data to standard format: [graphs,features,nodes]
nmat = np.empty((ngraphs,nfeat,nnodes))
findx = 0
sindx = nnodes

for i in range(ngraphs):

    hold = data[findx:sindx,:].T
    thold = np.concatenate((hold,coord.T),axis=0)
    nmat[i,:,:] = np.expand_dims(thold,0)

    findx = findx + nnodes
    sindx = sindx + nnodes

print("Graph data shape",nmat.shape) #Should be [graphs, features, nodes]

#Collecting sparse measurement
<<<<<<< HEAD
snodes = np.random.choice(np.arange(2601),size=500,replace=False)
=======
snodes = np.random.randint(0,nnodes,size=nsparse)
>>>>>>> eb5bdba03f8b6f6ba1a0db07f67847778baac663
np.save("snodesindices.npy",snodes)
time = np.arange(2,30,2) 
stime = np.sort(np.insert(time,0,[30,31]))

if opt.load_old_graphs == False:

    #Generating graphs
    if opt.GS == False: # Option if we do not want spectral information

        nmesh = nmat[:,np.array(stime,dtype=np.intp),:]
        nnmesh = nmesh[:,:,np.array(snodes,dtype=np.intp)]
        print("sparse measurement mesh shape:",nnmesh.shape)

        ind = np.arange(ngraphs)#Shuffled indices
        np.random.shuffle(ind)

        pygraphs = []
        for i in range(ngraphs):

            inpnodefeat = nnmesh[ind[i],:,:].squeeze().T
<<<<<<< HEAD
            print(i)
            print(lab.shape)
            print(ind.shape)
=======
>>>>>>> eb5bdba03f8b6f6ba1a0db07f67847778baac663
            inpnodelab = lab[ind[i],:].T.squeeze()
            inpcoord = coord[np.array(snodes,dtype=np.intp),:]

            print("Graph Number:",i)
<<<<<<< HEAD
            pygraphs.append(graphgen(inpnodefeat,inpnodelab[snodes],inpcoord))
=======
            pygraphs.append(graphgen(inpnodefeat,inpnodelab,inpcoord))
>>>>>>> eb5bdba03f8b6f6ba1a0db07f67847778baac663

        #Saving graph data
        with open("/home/dami/Inverse_GNN/datagraphs","wb") as fg:
            pickle.dump(pygraphs,fg)


    #Add spectral information
    if opt.GS == True: #Option if we want spectral information
        #NOTE: sparsity is handled implicitly in GST script

        # Encode geometric scattering information
        os.system("python apply_GS_wave.py")

        #Load GS information
        with open("/home/dami/Inverse_GNN/FEM_output/gspdata_wave","rb") as fh:
            GSTmat = np.swapaxes(pickle.load(fh),1,2)

        ind = np.arange(ngraphs)#Shuffled indices
        np.random.shuffle(ind)

        pygraphs = []
        for i in range(ngraphs):

            inpnodefeat = GSTmat[ind[i],:,:].squeeze().T
            inpnodelab = lab[ind[i],:].T.squeeze()
            inpcoord = coord[np.array(snodes,dtype=np.intp),:]

            print("Graph Number:",i)
            pygraphs.append(graphgen(inpnodefeat,inpnodelab,inpcoord))

        #Saving graph data
        with open("/home/dami/Inverse_GNN/datagraphs","wb") as fg:
            pickle.dump(pygraphs,fg)


else: #Just load old graphs

    with open("/home/dami/Inverse_GNN/datagraphs","rb") as ff:
        pygraphs = pickle.load(ff)

<<<<<<< HEAD



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
print(nearestnodesMAT.shape)


=======
>>>>>>> eb5bdba03f8b6f6ba1a0db07f67847778baac663
#Define data loaders
#train/validation/test split (60/20/20) [NOTE: Data has already been shuffled] in data loaders
bs = 10 #NOTE: only multiples of 50 samples will for batch size of 10
trainloader = DataLoader(pygraphs[:int(round(ngraphs*.6))],batch_size=bs,shuffle=True)
vadloader = DataLoader(pygraphs[ int(round(ngraphs*.6)): int(round(ngraphs*.8))],batch_size=bs,shuffle=True)
testloader = DataLoader(pygraphs[int(round(ngraphs*.8)):],batch_size=bs,shuffle=True)
print("graph data",pygraphs[0])

# Script for training GNN 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if opt.GS == True: #logic means that graphs have GST encoded

    input_features = pygraphs[0].x.shape[1] #solution mesh features + coordinates + GST features
    hidden_features = 128
    out_features = 1

else:
    input_features = pygraphs[0].x.shape[1] #solution mesh features + coordinates
    hidden_features = 128
    out_features = 1

in_nodes = int(nsparse * bs)
out_nodes = int(nnodes * bs)
model = NodeClassifier(input_features = input_features, hidden_features = hidden_features, out_features = out_features,in_nodes = in_nodes,out_nodes = out_nodes) 
optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)
criterion = torch.nn.MSELoss()
model = model.float().to(device)
lamda = 1 # regularization paramter
snodes = np.load("snodesindices.npy")


<<<<<<< HEAD
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




=======
>>>>>>> eb5bdba03f8b6f6ba1a0db07f67847778baac663
def train(fenicsopt,ep,regp):

    loss_all = 0
    c = 0 # Counter for number of batches
    for data in trainloader:

        #Generate model output
        data = data.to(device)
        model.train()   
        optimizer.zero_grad()
        out = model(data.x, data.edge_index,data.pos) 

<<<<<<< HEAD
        a = 0 
        b = 500
        c = 0
        d = 2601
        bsol = torch.empty((26010,1))
        for i in range(int(out.shape[0]/500)):
            bsol[c:d,:] = upsample_nodes(out[a:b].squeeze()).unsqueeze(1)
            c = c + 2601
            d = d + 2601
            a = a + 500
            b = b + 500
        


        #Formatting output 
        nnout = out.squeeze()
        #nnout = bsol.to(device).to(torch.double).squeeze()
        gtout = data.y
    

        if opt.FEM_loss and ep % 20 == 0 and ep > 700: #adjust weights based on solution mesh loss every 10 epochs
=======
        #Formatting output 
        nnout = out.squeeze()
        gtout = data.y
    

        if opt.FEM_loss and ep % 20 == 0:# and ep > 0: #adjust weights based on solution mesh loss every 10 epochs
>>>>>>> eb5bdba03f8b6f6ba1a0db07f67847778baac663

            regp = regp + .01
            #Call to fenics
            np.save("GNN_output.npy",out.cpu().detach().numpy())
            os.system("python wave_datagen.py")   

            #Load solution data and format
            fdata = np.loadtxt("fenics_sol_wave_opt.dat")

            findx = 0
            sindx = out_nodes
            recat = torch.empty((nsparse,14))

            for i in range(bs):
                x = fdata[findx:sindx,:]
                nx = x[np.array(snodes,dtype=np.intp),:]
                nxx = nx[:,np.array(time,dtype=np.intp)]
                recat = torch.cat((recat,torch.tensor(nxx)),dim=0)

            recatout = recat[nsparse:,:]
            

            #Computing solution loss
            loss_u = criterion( recatout.to(device).to(torch.double), data.x[:,:14].to(torch.double))

            # Computing parameter loss
            loss_p = criterion( nnout, gtout)
            loss_all += loss_p.item() # adds loss for each batch

            loss = loss_p + (regp*loss_u) #Incorporating loss from FEM solver
            loss.backward()
            optimizer.step()
            c = c + 1

        else:

            #Computing loss
            loss = criterion(nnout, gtout)
            loss_all += loss.item() # adds loss for each batch
            loss.backward()
            optimizer.step()
            c = c + 1
        
    return loss_all/c # reporting average loss per batch



def validate(fenicsopt,ep,regp):

    loss_all = 0
    c = 0 # counter for number of batches

    for data in vadloader:

        #Generating model output
        data = data.to(device)
        model.eval()
        out = model(data.x, data.edge_index,data.pos)

        #Formatting output 
        nnout = out.squeeze()
        gtout = data.y

<<<<<<< HEAD
        if fenicsopt and ep % 20 == 0 and ep > 700:
=======
        if fenicsopt and ep % 20 == 0 and ep > 0:
>>>>>>> eb5bdba03f8b6f6ba1a0db07f67847778baac663

            regp = regp + .01
            #Call to fenics
            np.save("GNN_output.npy",out.cpu().detach().numpy())
            os.system("python wave_datagen.py")         
            #Load solution data
            fdata = np.loadtxt("fenics_sol_wave_opt.dat")

            findx = 0
            sindx = out_nodes
            recat = torch.empty((nsparse,14))

            for i in range(bs):
                x = fdata[findx:sindx,:]
                nx = x[np.array(snodes,dtype=np.intp),:]
                nxx = nx[:,np.array(time,dtype=np.intp)]
                recat = torch.cat((recat,torch.tensor(nxx)),dim=0)

            recatout = recat[nsparse:,:]
            #Computing solution loss
            loss_u = criterion( recatout.to(device).to(torch.double), data.x[:,:14].to(torch.double))
            #loss_all += loss.item() # adds loss for each batch

            # Computing parametr loss
            loss_p = criterion( nnout, gtout)
            loss_all += loss_p.item() # adds loss for each batch

            loss = loss_p + (regp*loss_u) #Incorporating loss from FEM solver
            loss.backward()
            optimizer.step()
            c = c + 1

        else:
            # Computing loss
            nnout = out.squeeze().to(torch.double)
            ny = data.y.to(torch.double)
            loss = criterion( nnout, ny)
            loss_all += loss.item() # adds loss for each batch
            loss.backward()
            optimizer.step()
            c = c + 1

    return loss_all/c #normalizing by number of batches


svad = []
svtrain = []


for epoch in range(1, 1000):

    if opt.FEM_loss == True:
        fenicsopt=True
    else:
        fenicsopt=False

    train_loss = train(fenicsopt,epoch,lamda)
    validate_loss = validate(fenicsopt,epoch,lamda)

    svtrain.append(train_loss)
    svad.append(validate_loss)

    if opt.wandb == True:
        wandb.log({"train loss:":train_loss})
        wandb.log({"validation loss:":validate_loss})

    if epoch % 10 == 0: #Print every 10 epochs
        print("Epoch:",epoch, " Train Loss:",train_loss," Validation Loss:",validate_loss)

        
    #print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}) #, Validation Loss: {validate_loss:.4f}')



#Testing model with MSE
tloss = 0
MSE = []
pred_test = np.empty((nnodes, int(ngraphs*.2) ))


for data in testloader:

    data.to(device)
    model.eval()
    out = model(data.x,data.edge_index,data.pos)

    #Formatting output 
    nnout = out.squeeze()
    gtout = data.y

    bs = int(data.x.shape[0]/in_nodes) # batch size in test loader
    findx = 0
    sindx = nnodes

    #Saving MSE for each observation and storign average predicted paramters
    for i in range(bs):

        pred_test[:,i] = nnout.cpu().detach().numpy()[findx:sindx]
        MSE.append( np.sum((nnout.cpu().detach().numpy()[findx:sindx] - gtout.cpu().detach().numpy()[findx:sindx])**2) /nnodes)
        findx = findx + nnodes
        sindx = sindx + nnodes

    tloss += criterion(nnout,gtout)

GTOUT = np.expand_dims(data.y.cpu().detach().numpy()[findx:sindx],axis=1)
NNOUT = np.expand_dims(pred_test.mean(axis=1),axis=1)
PDF = pd.DataFrame(np.concatenate((GTOUT,NNOUT),axis=1),columns=['Ground_Truth','Predicted']) #Store table of average paramaters

if opt.wandb == True:
    wandb.log({"pred_gt_params":PDF})
<<<<<<< HEAD
    wandb.log({"avg_MSE":np.mean(MSE)})

print("MSE",np.mean(MSE))
np.save("MSE.npy",np.mean(MSE))
=======
    wandb.log({"avg_MSE":MSE})

print("MSE",MSE)
np.save("MSE.npy",MSE)
>>>>>>> eb5bdba03f8b6f6ba1a0db07f67847778baac663

"""
    if opt.wandb == True:

        wandb.log({"output:":nGNNout})
        wandb.log({"label:":nGNNlab})

    nGNNout = np.array(nGNNout)
    nGNNlab = np.array(nGNNlab)

    print(nGNNout.size)
    print(nGNNlab.size)

    coefdata = np.concatenate((nGNNout.reshape((nGNNout.size,1)),nGNNlab.reshape((nGNNlab.size,1))),axis=1)

    if opt.wandb == True:
        wandb.log({"coefficients":wandb.Table(data=coefdata,columns=["predictions","labels"])})

print("Average Test Loss (MSE)",tloss/len(testloader))



"""