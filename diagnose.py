from distutils.command import check
from lib2to3.pgen2 import driver
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.autograd import Variable
import wide_and_deep_model
import torch
import pickle
import time

def data_processing(data_path):
    # read the data 
    data_df = pd.read_excel(data_path, header=0)

    # transform the data
    data_df = data_df.drop(columns=["gene"])
    data_df = data_df.transpose()
    new_header = data_df.iloc[0]
    data_df = data_df.iloc[1:, :]
    data_df.columns = new_header 
    
    # get the genes
    # if the gene is not in the data, we replace it with the mean value
    with open("checkpoints/used_genes.pkl", "rb") as f:
        genes = pickle.load(f)
    
    mean_genes_df = pd.read_csv('checkpoints/mean_genes.csv', header=0)
    mean_genes_df = mean_genes_df.transpose()
    new_header = mean_genes_df.iloc[0]
    mean_genes_df = mean_genes_df.iloc[1:, :]
    mean_genes_df.columns = new_header 

    # Rename duplicated columns
    data_df.to_csv('Dump/temp.csv')
    data_df = pd.read_csv('Dump/temp.csv', header = 0)
    

    for gene in genes:
        if gene not in data_df:
            print(gene)
            data_df[gene] = mean_genes_df[gene]
    data_df = data_df[genes]
    return data_df

def detect(data_df):
    # Standardize the data
    data_df = data_df.values
    data_df = np.log2(data_df.astype('float64'))
    mean_genes_df = pd.read_csv('checkpoints/mean_genes_log2.csv', header=0)
    mean_genes_df = mean_genes_df.transpose()
    new_header = mean_genes_df.iloc[0]
    mean_genes_df = mean_genes_df.iloc[1:, :]
    mean_genes_df.columns = new_header 
    mean_genes = mean_genes_df.values

    std_genes_df = pd.read_csv('checkpoints/std_genes_log2.csv', header=0)
    std_genes_df = std_genes_df.transpose()
    new_header = std_genes_df.iloc[0]
    std_genes_df = std_genes_df.iloc[1:, :]
    std_genes_df.columns = new_header 
    std_genes = std_genes_df.values

    data_df = ((data_df - mean_genes)/std_genes).astype('float64')

    x_deep = torch.from_numpy(data_df[:,-25:])
    x_wide = torch.from_numpy(data_df)
    x_wide = Variable(x_wide).float()
    x_deep = Variable(x_deep).float()

    # Initialize model
    model = wide_and_deep_model.WideAndDeep(25, 105, [5], dropout=0.15, num_classes= 4)
    checkpoint = torch.load("checkpoints/wide_and_deep.ckpt")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Forward passing
    start = time.time()
    probability = model(x_wide, x_deep)
    end = time.time()
    _, prediction = torch.max(probability, dim= 1)
    probability = probability.detach().numpy()
    prediction = prediction.detach().numpy()
    inf_time = end - start
    return probability, prediction, inf_time

def end2end(data_path):
    data = data_processing(data_path)
    probability, prediction, inf_time = detect(data)
    return probability, prediction, inf_time
