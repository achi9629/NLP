# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 17:20:32 2021

@author: HP
"""
import torch.nn as nn
import numpy as np
import torch
from torch.utils.data import Dataset
import csv
from torch.utils.data import TensorDataset
import copy

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
# Fully connected neural network with hidden layers
class NeuralNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NeuralNet, self).__init__()
        
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(hidden_dim, 100)
        self.lin3 = nn.Linear(100, 50)
        self.lin4 = nn.Linear(50, output_dim )
        
        input1 = torch.empty(hidden_dim, input_dim)
        input2 = torch.empty(100, hidden_dim)
        input3 = torch.empty(50, 100)
        input4 = torch.empty(output_dim, 50)
        
        self.lin1.weight = torch.nn.Parameter(torch.nn.init.xavier_normal_(input1))
        self.lin2.weight = torch.nn.Parameter(torch.nn.init.xavier_normal_(input2))
        self.lin3.weight = torch.nn.Parameter(torch.nn.init.xavier_normal_(input3))
        self.lin4.weight = torch.nn.Parameter(torch.nn.init.xavier_normal_(input4))
        
    def forward(self, x):
        
        out = self.lin1(x)
        out = self.relu(out)
        
        out = self.lin2(out)
        out = self.relu(out)
        
        out = self.lin3(out)
        out = self.relu(out)
        
        out = self.lin4(out)
        
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
    batch_s = 64
    learn_rate = 0.0001
    epochs = 30

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset = TensorDataset(dataset.x_data[:30000], dataset.y_data[:30000]),
                                               batch_size=batch_s,
                                               shuffle=True)                                       
    
    
    # Model, loss and optimizer
    model = NeuralNet(input_size, hidden_size, output_size)
    
    loss = nn.MSELoss(reduction='mean')
    
    # Train the model
    optimizer = torch.optim.RMSprop(params=model.parameters(),lr = 0.0001, 
                                alpha=0.9, eps=1e-07, 
                                momentum=0.8)
    
    for epoch in range(epochs):
            
        train_loss = 0
        
        for i,(x,y) in enumerate(train_loader):
            
            # Forward pass
            y_hat = model(x.float())
            l = loss(y_hat.float(), y.float())
            train_loss += l*len(x)
            
            # Backward and optimize
            l.backward()       
            optimizer.step()
            optimizer.zero_grad()
            
        
        print (f'Epoch [{epoch+1}/{epochs}], Training_Loss: {torch.sqrt(train_loss/(len(train_loader)*batch_s))}')
           
                
    with torch.no_grad():
        y_hat = model(dataset.x_data[30000:].float())
        y_real = dataset.y_data[30000:]
        l = loss(y_hat, y_real)
        print(f'Testing_Loss :{torch.sqrt(l)}')
    
     #%%
    best_model_wts = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_model_wts)   
    
    # save only state dict
    FILE = "saved_model/model_7.pth"
    torch.save(model.state_dict(), FILE)
    
    loaded_model = NeuralNet(input_size, hidden_size, output_size)
    loaded_model.load_state_dict(torch.load(FILE))
    loaded_model.eval()
    
    with torch.no_grad():
            y_hat = loaded_model(dataset.x_data[30000:].float())
            y_real = dataset.y_data[30000:]
            l = loss(y_hat, y_real)
            print('Minimum Testing Loss :',torch.sqrt(l).item())


