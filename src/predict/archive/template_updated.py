import dgl
import torch
import torch.distributed
import pandas as pd
import pickle

from torch import nn
from torch.nn import functional as F
import os
import sys
import glob
import shutil
import datetime
import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import *
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed
from tqdm import tqdm
import pkg_resources
import warnings
import traceback


from dgl.nn.pytorch.conv import GATConv
from dgl.dataloading import GraphDataLoader
from scipy.spatial.distance import cdist


class GATLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.gat1 = GATConv(2, hidden_dim, num_heads=1)
        self.gat2 = GATConv(hidden_dim, hidden_dim, num_heads=1)
        self.lin = nn.Linear(hidden_dim, 3)

    def forward(self, g, h, unsplice, splice, alpha0, beta0, gamma0, dt):

        h = self.gat1(g[0], h)
        h = F.relu(h)
        h = self.gat2(g[1], h)
        h = F.relu(h)
        h = torch.sigmoid(self.lin(h))

        us = g[1].dstdata['feat'][:, 0]
        s = g[1].dstdata['feat'][:, 1]

        beta = h[:,:,:,0].squeeze()
        gamma = h[:,:,:,1].squeeze()
        alphas = h[:,:,:,2].squeeze()

        alphas = alphas * alpha0
        beta =  beta * beta0
        gamma = gamma * gamma0


        unsplice_predict = us + (alphas - beta*us)*dt
        splice_predict = s + (beta*us - gamma*s)*dt

        return h.mean(dim=3).squeeze()
    

def train(g, features, labels, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # features = g.ndata['feat'] 
    # labels = g.ndata["label"]["_N"].to("cuda")
    train_nid = torch.tensor(range(g.num_nodes())).type(torch.int64)
 
    for epoch in range(24):
        model.train()
        sampler = dgl.dataloading.MultiLayerNeighborSampler([15, 10])
        dataloader = dgl.dataloading.DataLoader(
            g, train_nid, sampler, use_ddp=False,
            batch_size=1024, shuffle=True, drop_last=False, num_workers=0)
 
        total_loss = 0
         
        for step, (input_nodes, output_nodes, blocks) in enumerate((dataloader)):
            batch_inputs = features[input_nodes]
            batch_labels = labels[output_nodes]

            unsplice, splice = batch_inputs[:, 0], batch_inputs[:, 1]
            umax, smax = max(unsplice), max(splice)
            alpha0 = np.float32(umax*2)
            beta0 = np.float32(1.0)
            gamma0 = np.float32(umax/smax*1)

            batch_pred = model(blocks, batch_inputs, unsplice, splice, alpha0, beta0, gamma0, 0.5)
            
            loss = F.mse_loss(batch_pred, batch_labels)
            total_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
 
 
        sampler = dgl.dataloading.NeighborSampler([-1,-1])
        dataloader = dgl.dataloading.DataLoader(g, train_nid, sampler,
                                                batch_size=1024,
                                                shuffle=False,
                                                drop_last=False,
                                                num_workers=0,
                                                )
 
 
        mse = evaluate(model, features, labels, dataloader)
        print("Epoch {:05d} | MSE {:.4f} | Loss {:.4f} ".format(epoch, mse, total_loss))
 
 
def evaluate(model, features, labels, dataloader):
    with torch.no_grad():
        model.eval()
        ys = []
        y_hats = []
        for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
            with torch.no_grad():
                batch_inputs = features[input_nodes]
                batch_labels = labels[output_nodes]

                unsplice, splice = batch_inputs[:, 0], batch_inputs[:, 1]
                umax, smax = max(unsplice), max(splice)
                alpha0 = np.float32(umax*2)
                beta0 = np.float32(1.0)
                gamma0 = np.float32(umax/smax*1)

                ys.append(batch_labels)
                y_hats.append(model(blocks, batch_inputs, unsplice, splice, alpha0, beta0, gamma0, 0.5))
        # return F.accuracy()
        return F.mse_loss(torch.cat(y_hats), torch.cat(ys))
    


def make_graph(unsplice, splice, embedding1, embedding2):
    num_nodes = len(unsplice)

    g = dgl.DGLGraph()
    g.add_nodes(num_nodes)

    node_pairs = np.column_stack((unsplice, splice))

    # Calculate Euclidean distances between all pairs of nodes
    distances = cdist(node_pairs, node_pairs)
    # print(distances.shape)

    # Iterate through each node and connect it to the closest nodes
    for i in range(num_nodes):
        closest_indices = np.argsort(distances[i])[1:31]

        g.add_edges(i, closest_indices)
        g.add_edges(closest_indices, i)
    
    feat = torch.tensor(node_pairs, dtype=torch.float32)
    labels = torch.tensor(np.ones(12329), dtype=torch.float32)

    g.ndata['feat'] = feat
    g.ndata['labels'] = labels
    return g, feat, labels


cell_type_u_s_path = '/Users/jenniferli/Downloads/CSCI 2952G/GastrulationErythroid_cell_type_u_s.csv'

g, f, labels = None, None, None
# Check if the processed data exists, if not, read and process the data
try:
    with open('processed_data.pkl', 'rb') as file:
        cell_type_u_s, g, f, labels = pickle.load(file)
except FileNotFoundError:
    cell_type_u_s = pd.read_csv(cell_type_u_s_path)
    filtered_df = cell_type_u_s[cell_type_u_s['gene_name'] == 'Myo1b']
    
    unsplice = filtered_df.iloc[:, 1].tolist()
    splice = filtered_df.iloc[:, 2].tolist()
    g, f, labels = make_graph(unsplice, splice)
    
    # Save the processed data
    with open('processed_data.pkl', 'wb') as file:
        pickle.dump((cell_type_u_s, g, f, labels), file)


model = GATLayer(100)
train(g, f, labels, model)





# torch.distributed.destroy_process_group()


# sampler = dgl.dataloading.MultiLayerNeighborSampler([15, 10, 5])
# train_nid = torch.tensor(range(g.num_nodes())).type(torch.int64)
# dataloader = dgl.dataloading.DataLoader(
#     g, train_nid, sampler, use_ddp=True,
#     batch_size=1024, shuffle=True, drop_last=False, num_workers=4)

# print(unsplice)
# print(splice)
# print(unsplice.shape)
# print(splice.shape)


# cell_type_u_s

# gene_list=['Myo1b']
# # Define the graph and features
# n_cells = 1000
# g = dgl.graph((torch.arange(n_cells - 1), torch.arange(1, n_cells)), num_nodes=n_cells)
# unspliced_mrna = torch.randn(n_cells, 1)
# spliced_mrna = torch.randn(n_cells, 1)
# features = torch.cat((unspliced_mrna, spliced_mrna), dim=1)

# # Cluster cells by Euclidean distance
# clusterer = KMeans(n_clusters=10)
# clusterer.fit(features.cpu())
# cell_clusters = clusterer.labels_

# # Define the model and optimizer
# model = SplicePredictor(128)
# optimizer = torch.optim.Adam(model.parameters())

# # Training loop
# for epoch in range(100):
#     # Forward pass
#     logits = model(g, features)

#     # Define loss function and calculate loss
#     loss = F.binary_cross_entropy(logits, target_labels)

#     # Backward pass and update parameters
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

# # Use the trained model to predict alpha, beta, and gamma for new data
# new_features = torch.randn(100, 2)
# new_logits = model(g, new_features)
# predicted_alpha, predicted_beta, predicted_gamma = torch.split(new_logits, 1, dim=1)