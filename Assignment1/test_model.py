# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 01:19:31 2021

@author: HP
"""
import torch.nn as nn
import numpy as np
import torch
from torch.utils.data import  Dataset
import csv


############## DataLoading ########################
class LoadData(Dataset):
    def __init__(self):
           
        #Load csv file
        with open('OnlineNewsPopularity.csv', "r") as f:
            xy = list(csv.reader(f, delimiter=","))
        
        #Convert data to numpy array
        xy = np.array(xy)
        
        #Remove first 2 columns and first row
        xy = xy[1:,2:].astype(np.float64)
        
        self.x_data = xy[:,:-1]
        self.y_data = xy[:,[-1]]
        
        # Here the last column is the class label, the rest are the features
        self.x_data = torch.from_numpy(self.x_data)   # size [n_samples, n_features]
        self.y_data = torch.from_numpy(self.y_data) # size [n_samples, 1]
        

        #Number of samples and features
        self.num_samples, self.num_features = self.x_data.shape
        
    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
            return self.x_data[index], self.y_data[index]
        
    # we can call len(dataset) to return the size
    def __len__(self):
            return self.num_samples
        
    #get the number of fetures  
    def features(self):
        return self.num_features
###################################################     


#################### Model ########################
# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NeuralNet, self).__init__()
        
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(hidden_dim, 100)
        self.lin3 = nn.Linear(100, 50)
        self.lin4 = nn.Linear(50, output_dim )
        
    def forward(self, x):
        
        out = self.lin1(x)
        out = self.relu(out)
        
        out = self.lin2(out)
        out = self.relu(out)
        
        out = self.lin3(out)
        out = self.relu(out)
        
        out = self.lin4(out)
        out = self.relu(out)
        
        return out
###################################################

if __name__=="__main__":
    
    # create dataset
    dataset = LoadData()
    first_data = dataset[0]
    features, target = first_data
    print(features.shape, target.shape)
    
    #Hyperparameters
    input_size = dataset.features()
    hidden_size = 200
    output_size = 1
    
    FILE = "saved_model/model_7665.pth"
    
    loaded_model = NeuralNet(input_size, hidden_size, output_size)
    loss = nn.MSELoss(reduction='mean')
    
    loaded_model.load_state_dict(torch.load(FILE))
    loaded_model.eval()

    print(len(dataset.x_data[30000:]))
    
    with torch.no_grad():
        y_hat = loaded_model(dataset.x_data[30000:].float())
        y_real = dataset.y_data[30000:]
        l = loss(y_hat, y_real)
        print('Minimum Testing Loss :',torch.sqrt(l).item())
