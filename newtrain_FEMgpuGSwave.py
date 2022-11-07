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

























#Save process ID (so that I can kill later if I want)
f = open("PID.txt","w")
f.write(str(os.getpid()))
f.close()

#Wandb option
if opt.wandb == True:
    wandb.init(project="wave_equation_nFEM_model_velocity_rerun")


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

    path = '/FEM_output/fenics_coord_wave'
    npath = cwd + path 

    with open(npath,"rb") as fe:
        coord = pickle.load(fe)

    path = '/FEM_output/fenics_lab_wave'
    npath = cwd + path 

    with open(npath,"rb") as ff:
        lab = pickle.load(ff)


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


## Need to manually the data, but by keep order of time in each observation 
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
    with open("/home/dami/Inverse_GNN/datagraphs","rb") as ff:
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




#Split data into batches
bs = int(ntimesteps*1) 
trainloader = DataLoader(pygraphs[:int(round(ngraphs*.2))],batch_size=bs,shuffle=False)
vadloader = DataLoader(pygraphs[ int(round(ngraphs*.2)): int(round(ngraphs*.4))],batch_size=bs,shuffle=False)
testloader = DataLoader(pygraphs[int(round(ngraphs*.4)):],batch_size=bs,shuffle=False)


# Script for training GNN 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if opt.GS == True:

    input_features = pygraphs[0].x.shape[1] #solution mesh features + coordinates + GST features
    hidden_features = 128
    out_features = 1

else:
    input_features = pygraphs[0].x.shape[1] #solution mesh features + coordinates
    print(input_features)
    hidden_features = 10
    out_features = 1

if sparsemeasure == True:
    in_nodes = int(nsparse * bs)
    out_nodes = int(nnodes * bs)
else:
    in_nodes = int(nnodes*bs)
    out_nodes = int(nnodes*bs)

model = NodeClassifier(input_features = input_features, hidden_features = hidden_features, out_features = out_features,in_nodes = in_nodes,out_nodes = out_nodes) 

f = open("model_weights.txt","w")
f.write(str(model.lin1.weight))
f.close()

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



def train(fenicsopt,ep,regp):

    loss_all = 0
    c = 0 # Counter for number of batches
    for data in trainloader:

  
        #Generate model output
        data = data.to(device)
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

        if opt.FEM_loss: #and ep % 20 == 0 and ep > 700: #adjust weights based on solution mesh loss every 10 epochs

            findx = 0
            sindx = nnodes

            autoregressive = True
            if autoregressive: 
                #npredval = int(ntimesteps-1)
                npredval = 5
                a = 0
            else:
                npredval = ntimesteps
                a = 1

            tindx = 1
            loss_p_batch = []
            for j in range(npredval): # backpropagating for predicted values at each time step

                #NOTE: will only predicted t-1 field values for t time steps if autoregressive


                regp = regp + .01
                #print(out.size())
                #print(out)
                #Call to fenics

                #tindx determines how many time points are used
                #tindx = j + 1 #time index
                tindx = tindx + 5

            
                
                #FEMinput = np.append(out.cpu().detach().numpy()[findx:sindx,:],tindx) #Store time in second to last index
                #FEMinputa = np.append(FEMinput,a) # Store autoregressive flag in last index
                #np.save("GNN_output.npy",FEMinputa)
                FEMinputa = out[findx:sindx]
                f = open("time.txt","w")
                f.write(str(tindx))
                f.close()
                check = torch.transpose(FEMinputa,0,1)
                torch.save(check, "GNN_output.pt")
                os.system("python wave_datagen.py")   
                path = '/FEM_output/fenics_sol_wave_opt'
                npath = cwd + path 

                with open(npath,"rb") as ff:
                    fdata = pickle.load(ff)
                sys.exit()

                if sparsemeasure == True:

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

                else:

                    #loss should be between next set of nodes for autogressive so we update node indeces. Also, they need to be updated anyways for next iteration
                    findx = findx + nnodes 
                    sindx = sindx + nnodes

                    indx_sol = 0 #index of solution mesh data
                    if autoregressive:#compute loss at next time step
                        loss_u = criterion( fdata.to(device).to(torch.double), data.x[findx:sindx,indx_sol].to(torch.double))   
                    else: #compute loss at all time steps
                        loss_u = criterion( fdata.to(device).to(torch.double), data.x[:,indx_sol].to(torch.double))

                   
                #print(loss_u)
                # Computing parameter loss
                #loss_p = criterion( nnout, gtout)
                loss_all += loss_u.item() # adds loss for each batch
                #loss = loss_u #loss_p + (regp*loss_u) #Incorporating loss from FEM solver (backpropagating for each batch)
                print(model.lin1.weight.grad)
                loss_u.backward()
                print(model.lin1.weight.grad)
                optimizer.step()
                sys.exit()

                loss_p_batch.append(loss_u.cpu().detach())

            lpbatch = np.mean(np.array(loss_p_batch))
            print(lpbatch)
            if opt.wandb == True:
                wandb.log({"train loss per batch:":lpbatch})
                
    

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
        
    return loss_all/c # reporting average loss per batch for each epoch



def validate(fenicsopt,ep,regp):

    loss_all = 0
    c = 0# counter for number of batches

    for data in vadloader:

        #Generating model output
        data = data.to(device)
        model.eval()
        out = model(data.x, data.edge_index,data.pos)

        #Formatting output 
        nnout = out.squeeze()
        gtout = data.y

        print("Validation Observation Number:",c)

        
        if opt.FEM_loss:# and ep % 20 == 0 and ep > 700:

            findx = 0
            sindx = nnodes

            autoregressive = True
            if autoregressive: 
                #npredval = int(ntimesteps-1)
                npredval = 5
                a = 0
            else:
                npredval = ntimesteps
                a = 1

            tindx = 1
            for j in range(npredval): # backpropagating for predicted values at each time step

                #NOTE: will only predicted t-1 field values for t time steps if autoregressive


                regp = regp + .01
                #print(out.size())
                #print(out)
                #Call to fenics

                #time index determines
                #tindx = j + 1 #time index
                tindx = tindx + 5

                FEMinput = np.append(out.cpu().detach().numpy()[findx:sindx,:],tindx) #Store time in second to last index
                FEMinputa = np.append(FEMinput,a) # Store autoregressive flag in last index
                regp = regp + .01
                #Call to fenics
                np.save("GNN_output.npy",FEMinputa)
                os.system("python wave_datagen.py")         
                #Load solution data

                path = '/FEM_output/fenics_sol_wave_opt'
                npath = cwd + path 
                with open(npath,"rb") as ff:
                    fdata = pickle.load(ff)


                if sparsemeasure == True:

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
                
                    loss_u = criterion( recatout.to(device).to(torch.double), data.x[:,indx_sol].to(torch.double))


                    
                else:
 
                     #loss should be between next set of nodes for autogressive so we update node indeces. Also, they need to be updated anyways for next iteration
                    findx = findx + nnodes 
                    sindx = sindx + nnodes

                    indx_sol = 0 #index of solution mesh data
                    if autoregressive:#compute loss at next time step
                        loss_u = criterion( fdata.to(device).to(torch.double), data.x[findx:sindx,indx_sol].to(torch.double))   
                    else: #compute loss at all time steps
                        loss_u = criterion( fdata.to(device).to(torch.double), data.x[:,indx_sol].to(torch.double))   
                        
                    #print("Time",tindx)
                    #print("Validation Loss",loss_u)

                    if opt.wandb == True:
                        wandb.log({"validation loss:":loss_u})


            #loss_all += loss.item() # adds loss for each batch

            # Computing parametr loss
            #loss_p = criterion( nnout, gtout)
            loss_all += loss_u.item() # adds loss for each batch
            #loss = loss_u#loss_p + (regp*loss_u) #Incorporating loss from FEM solver
            print(model1.lin.weight.grad)
            loss_u.backward()
            print(model1.lin.weight.grad)
            optimizer.step()
          
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

        c = c + 1 #update count for each batch

    return loss_all/c #report average batch loss for each epoch


svad = []
svtrain = []


for epoch in range(1, 4):

    print("Epoch",epoch)

    if opt.FEM_loss == True:
        fenicsopt=True
    else:
        fenicsopt=False

    train_loss = train(fenicsopt,epoch,lamda)
    validate_loss = validate(fenicsopt,epoch,lamda)

    f = open("model_weights.txt","a")
    f.write(str(model.lin1.weight))
    f.close()

    svtrain.append(train_loss)
    svad.append(validate_loss)

    #if opt.wandb == True:
    #    wandb.log({"train loss:":train_loss})
    #    wandb.log({"validation loss:":validate_loss})

    if epoch % 10 == 0: #Print every 10 epochs
        print("Epoch:",epoch, " Train Loss:",train_loss," Validation Loss:",validate_loss)

        
    #print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}) #, Validation Loss: {validate_loss:.4f}')



#Testing model with MSE
tloss = 0
MSE = []
pred_testfin = np.empty((nnodes, int(nobs*.2)))
gt_testfin = np.empty((nnodes, int(nobs*.2)))


c = 0
for data in testloader:

    
    data.to(device)
    model.eval()
    out = model(data.x,data.edge_index,data.pos)

    #Formatting output 
    nnout = out.squeeze()
    gtout = data.y


    bs = int(data.x.shape[0]/nnodes) # batch size in test loader
    j = int(c*bs) # Counter
    findx = 0
    sindx = nnodes

    #Saving MSE for each observation and storing randomly selected predicted paramters
    timeselect = np.random.randint(0,28) #selecting random index to store paramters
    rfindx = int(nnodes*timeselect)
    rsindx = int(nnodes*(timeselect+1))
    pred_testfin[:,c] = nnout.cpu().detach().numpy()[rfindx:rsindx] #Storing predicted values
    gt_testfin[:,c] = gtout.cpu().detach().numpy()[rfindx:rsindx]
    tloss += criterion(nnout,gtout)

    c = c + 1 #update batch number

#Save data and model
df_pred = pd.DataFrame(pred_testfin)
df_gt = pd.DataFrame(gt_testfin)
print(df_gt)
print(df_pred)

np.save("nnout.npy",pred_testfin)
np.save("gtout.npy",gt_testfin)

path = '/Models/wave_velocity_20ep_GS.pth'
npath = cwd + path
torch.save(model.state_dict(), npath)
artifact = wandb.Artifact('wave_velocity_20ep_GS',type='model')
artifact.add_file(npath)
wandb.log_artifact(artifact)
#wandb.join()

#Report to WANDB
if opt.wandb == True:
    wandb.log({"pred_params":df_pred})
    wandb.log({"gt_params":df_gt})
    #wandb.log({"avg_MSE":np.mean(MSE)})

#print("MSE",np.mean(MSE))
#np.save("MSE.npy",np.mean(MSE))

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
