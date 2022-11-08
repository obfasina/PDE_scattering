
import graphScattering as GSTransform
import numpy as np
import pickle
import sys
from scipy.spatial.distance import pdist, squareform
import os
from varname import nameof

"""

#define node/graph properties
nfeat = 32
nnodes = 2601 

# import graph data 

FEMout = np.loadtxt("fenics_sol_wave.dat")
bs = int(FEMout.shape[0]/nnodes)
with open("/home/dami/Inverse_GNN/FEM_output/fenics_coord_wave","rb") as fe:
    coord = pickle.load(fe)

print(FEMout.shape)
print(coord.shape)


#Creating correct input format for Geometric Scattering Transfrom
nmat = np.empty((bs,nfeat,nnodes))
findx = 0
sindx = nnodes

for i in range(bs):

    hold = FEMout[findx:sindx,:].T
    thold = np.concatenate((hold,coord.T),axis=0)
    nmat[i,:,:] = np.expand_dims(thold,0)

    findx = findx + nnodes
    sindx = sindx + nnodes

print("Pre-graph shape",nmat.shape)

sparsemeasure = False

if sparsemeasure == True:

    #Get sparse measurement in space and time
    snodes = np.load("snodesindices.npy")
    time = np.arange(2,30,2)
    stime = np.sort(np.insert(time,0,[30,31]))

    nmesh = nmat[:,np.array(stime,dtype=np.intp),:]
    nnmesh = nmesh[:,:,np.array(snodes,dtype=np.intp)]
    nnodes = nnmesh.shape[2]


else: 
    nnmesh = nmat

"""


#Load post-processed solution mesh
#nnmesh = np.load("GNN_input_sol_gauss_ellipse.npy")
#print("Pre-GS shape",nnmesh.shape)

def get_scattering(nnmesh):

    bs = nnmesh.shape[0]
    nnodes = nnmesh.shape[2]



    #Compute adjacency matrices
    D = squareform(pdist(nnmesh[0,:,:].T))

    sigma = 3
    A = np.exp(-D**2/sigma)

    # Perform graph scattering transform which goes from graph space to euclidean space and gives GS features
    nscale = 3 # Changes the number of frequencies you can resolve
    nlayers = 2
    nmoments = 5
    GS = GSTransform.GeometricScattering(nscale,nlayers,nmoments,A) #initialize class
    transformedgraph = GS.computeTransform(np.expand_dims(nnmesh[0,:,:],0)) #output is (batch size x features x (scales * moments))
    # output is a feature augmented matrix where each feauture has now been agumented by (diffusion scattering coeffficinets)


    # Create new graph of shape [nnodes x geometric scatering features] by performing matrix multiplication
    gsmat = np.empty((bs,nnodes,transformedgraph.shape[2]))
    for i in range(bs):

        D = squareform(pdist(nnmesh[i,:,:].T))
        sigma = 3
        A = np.exp(-D**2/sigma)
        GS = GSTransform.GeometricScattering(nscale,nlayers,nmoments,A)
        transformedgraph = GS.computeTransform(np.expand_dims(nnmesh[i,:,:],0)) 
        gsmat[i,:,:] = nnmesh[i,:,:].T @ transformedgraph[0,:,:].squeeze()

    print("Input feature matrix shape:",nnmesh.shape)
    x=np.swapaxes(nnmesh,1,2)
    print("After swapping axes:",x.shape)
    print("Geometric scattering transform shape:",gsmat.shape)

    ngsmat = np.concatenate((x,gsmat),axis=-1) #Concatenating arrays 
    finalmat = np.swapaxes(ngsmat,1,2)
    np.save('GS_' + nameof(nnmesh),finalmat)
    print(finalmat.shape)

    return finalmat



"""

# Dimensionality reduce graph by removing nodes that have the least variance (just grab indeces)

varnode = []
for i in range(nm.shape[0]):
    varnode.append(np.var(nm[i,:]))

dim = 5 #number of nodes to keep
indnodes = np.argsort(varnode)[:dim]
print(indnodes)

# Invert from PC space back to data space using only PC's that explain data variance

# Construct new dim-reduced graph with less nodes (only nodes that explain variance)

"""
