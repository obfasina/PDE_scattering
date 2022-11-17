This package includes a neural network equipped with scattering features and a PDE solver. The PDE solver can aslo be used for synthetic data generation.
The main components of the pacakage are:

  a) The GNN module 'Granola_GNN.py'
  b) The geometric scattering script 'apply_GS_wave.py'
  c) The training script 'trainwave.py'
  d) The data generation script 'gendatawave.py'
  e) The script for running the wave equations on graphs 'wave_graphs.py'
  
To generate synthetic data on arbitrary meshes or graphs, fist  'conda create --name myenv --file gendatreq.txt' to activate the conda environment:
Run 'python genwavedata.py' to generate synthetic data on arbitrary meshes
Run 'python wave_graphs.py' to simulate wave equation on graphs (can modify file to include your own graphs)

To train GNPnet: 
Run 'conda create --name myenv --file GNPnetrun.txt' to create the conda environment 
Run 'trainwave.py' to train the network

GNPnet has a few flags:
a) --wandb will track the results on wandb
b) --load_old_graphs just loads the graphs from the previous run
c) --GS applies geometric scattering features to the graph data
