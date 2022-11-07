
import GStransforms.Modules.graphScattering as GSTransform
import numpy as np
import pickle
import sys
from scipy.spatial.distance import pdist, squareform



#define node/graph properties
nfeat = 3
nnodes = 81


# import graph data 
bs = 500
with open("/home/dami/Inverse_GNN/FEM_output/fenics_sol","rb") as fh:
    fdata = pickle.load(fh)
    FEMout = np.expand_dims(fdata,1)

with open("/home/dami/Inverse_GNN/FEM_output/fenics_coord","rb") as fe:
    coord = pickle.load(fe)


<<<<<<< HEAD
print(FEMout.shape)
print(coord.shape)

=======
>>>>>>> eb5bdba03f8b6f6ba1a0db07f67847778baac663
nmat = np.empty((bs,nfeat,nnodes))
for i in range(bs):
    nmat[i,:,:] = np.concatenate((FEMout[i,:,:],coord.T),axis=0)

<<<<<<< HEAD
print(FEMout.shape)
print(nmat.shape)

sys.exti()
=======
>>>>>>> eb5bdba03f8b6f6ba1a0db07f67847778baac663


#Compute adjacency matrices
D = squareform(pdist(nmat[0,:,:].T))
sigma = 3
A = np.exp(-D**2/sigma)

# Perform graph scattering transform which goes from graph space to euclidean space and gives GS features
nscale = 5 # Changes the number of frequencies you cna resolve
nlayers = 3
nmoments = 1
GS = GSTransform.GeometricScattering(nscale,nlayers,nmoments,A) #initialize class
transformedgraph = GS.computeTransform(np.expand_dims(nmat[0,:,:],0)) #output is (batch size x features x (scales * moments))
# output is a feature augmented matrix where each feauture has now been agumented by (diffusion scattering coeffficinets)


# Create new graph of shape [nnodes x geometric scatering features] by performing matrix multiplication
gsmat = np.empty((bs,nnodes,transformedgraph.shape[2]))
for i in range(bs):

    D = squareform(pdist(nmat[i,:,:].T))
    sigma = 3
    A = np.exp(-D**2/sigma)
    GS = GSTransform.GeometricScattering(nscale,nlayers,nmoments,A)
    transformedgraph = GS.computeTransform(np.expand_dims(nmat[i,:,:],0)) 
    gsmat[i,:,:] = nmat[i,:,:].T @ transformedgraph[0,:,:].squeeze()

print("Input feature matrix shape:",nmat.shape)
x=np.swapaxes(nmat,1,2)
print("After swapping axes:",x.shape)
print("Geometric scattering transform shape:",gsmat.shape)

ngsmat = np.concatenate((x,gsmat),axis=-1) #Concatenating arrays 
print(ngsmat.shape)


with open("/home/dami/Inverse_GNN/FEM_output/gspdata","wb") as fb:
    pickle.dump(ngsmat,fb)



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

<<<<<<< HEAD
"""
=======
"""
>>>>>>> eb5bdba03f8b6f6ba1a0db07f67847778baac663
