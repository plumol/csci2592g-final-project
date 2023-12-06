# from __future__ import absolute_import
import os
import sys
import glob
import pandas as pd
import math
import matplotlib.pyplot as plt
import celldancer as cd
import celldancer.cdplt as cdplt
from celldancer.cdplt import colormap
import dgl
from scipy.spatial.distance import cdist
import networkx as nx
import numpy as np

from model import build_datamodule

cell_type_u_s_path='/Users/jenniferli/Downloads/CSCI 2952G/GastrulationErythroid_cell_type_u_s.csv'
cell_type_u_s=pd.read_csv(cell_type_u_s_path)
cell_type_u_s

gene_list=['Smarca2', 'Rbms2', 'Myo1b', 'Hba-x', 'Yipf5', 'Skap1', 'Smim1', 'Nfkb1', 'Sulf2', 'Blvrb', 'Hbb-y', 'Coro2b', 'Yipf5', 'Phc2', 'Mllt3']

# def velocity(
#     cell_type_u_s,
#     gene_list=None,
#     max_epoches=200, 
#     check_val_every_n_epoch=10,
#     patience=3,
#     learning_rate=0.001,
#     dt=0.5,
#     n_neighbors=30,
#     permutation_ratio=0.125,
#     speed_up=True,
#     norm_u_s=True,
#     norm_cell_distribution=True,
#     loss_func='cosine',
#     n_jobs=-1,
#     save_path=None,
# ):

if gene_list is None:
    gene_list=list(cell_type_u_s.gene_name.drop_duplicates())
else:
    cell_type_u_s=cell_type_u_s[cell_type_u_s.gene_name.isin(gene_list)]
    all_gene_name_cell_type_u_s=list(cell_type_u_s.gene_name.drop_duplicates())
    gene_not_in_cell_type_u_s= list(set(gene_list).difference(set(all_gene_name_cell_type_u_s)))
    gene_list=list(list(set(all_gene_name_cell_type_u_s).intersection(set(gene_list))))
    if len(gene_not_in_cell_type_u_s)>0: print(gene_not_in_cell_type_u_s," not in the data cell_type_u_s")

cell_type_u_s=cell_type_u_s.reset_index(drop=True)
gene_list_buring=[list(cell_type_u_s.gene_name.drop_duplicates())[0]]
print("wtf is a buring")
print(gene_list_buring)

datamodule=build_datamodule(cell_type_u_s,False,True,0.125,True,gene_list=gene_list_buring) #type is feedData
# print("fit fit")
# print(datamodule.fit_dataset.data_fit)
# print("fit pred")
# print(datamodule.fit_dataset.data_predict)

# print("pred fit")
# print(datamodule.predict_dataset.data_fit)
# print("pred pred")
# print(datamodule.predict_dataset.data_predict)

for data_index in range(0,len(gene_list_buring)):
    print([data_index])
    selected_data = datamodule.subset([data_index])
    unsplice, splice, this_gene_name, unsplicemax, splicemax, embedding1, embedding2=selected_data.fit_dataset.__getitem__(0)
    print("selected data:")
    print(selected_data)
    print("?")
    print(unsplice.shape)
    print(splice.shape)
    print(this_gene_name)
    print(unsplicemax.shape)
    print(splicemax.shape)
    print(embedding1.shape)
    print(embedding2.shape)






# Create a graph
num_nodes = len(unsplice)
g = dgl.DGLGraph()

# Add nodes to the graph
g.add_nodes(num_nodes)

# Create pairs of values (unsplice_i, splice_i)
node_pairs = np.column_stack((unsplice, splice))

# Calculate Euclidean distances between all pairs of nodes
distances = cdist(node_pairs, node_pairs)
print(distances.shape)

# Iterate through each node and connect it to the closest nodes
for i in range(num_nodes):
    # Find the indices of the closest nodes (excluding itself)
    closest_indices = np.argsort(distances[i])[1:31]

    # Add edges between the current node and its closest neighbors
    g.add_edges(i, closest_indices)
    g.add_edges(closest_indices, i)

# Print the resulting graph
print(g)


# Convert DGL graph to NetworkX graph
nx_graph = g.to_networkx()

# Plot the graph
pos = nx.spring_layout(nx_graph, seed=0)  # You can use different layout algorithms
nx.draw(nx_graph, pos, with_labels=False, node_color='#4EB7C6', node_size=80, font_size=8, font_color='black', font_weight='normal', edge_color='#CCCCCC', linewidths=0.05)

# Show the plot
plt.show()

print("here")


# loss_df, cellDancer_df=cd.velocity(cell_type_u_s,\
#                                    gene_list=gene_list,\
#                                    permutation_ratio=0.125,\
#                                    n_jobs=8)