import os
import sys
import glob
import pandas as pd
import math
import matplotlib.pyplot as plt
from orig import velocityo


cell_type_u_s_path='/Users/jenniferli/Downloads/CSCI 2952G/GastrulationErythroid_cell_type_u_s.csv'
cell_type_u_s=pd.read_csv(cell_type_u_s_path)
cell_type_u_s

gene_list=['Myo1b']

loss_df, cellDancer_df=velocityo(cell_type_u_s,\
                                   gene_list=gene_list,\
                                   permutation_ratio=0.125)
cellDancer_df