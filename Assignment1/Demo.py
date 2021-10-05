# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 17:20:32 2021

@author: HP
"""
import time
import torch.nn as nn
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import math
import numpy as np
import csv
from torch.utils.data import TensorDataset
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
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
    batch_s = [64, 128, 256, 512, 2000, 5000, 10000, 15000]
    learn_rate = [0.0001]
    epochs = 100

    minimum_loss = 9999.
    epoch_no = 0
    learning = 0
    size = 0
    for batch in batch_s:
        for lr in learn_rate:
            
            # Data loader
            train_loader = torch.utils.data.DataLoader(dataset = TensorDataset(dataset.x_data[:30000], dataset.y_data[:30000]),
                                                       batch_size=batch,
                                                       shuffle=True)
            
            test_loader = torch.utils.data.DataLoader(dataset = TensorDataset(dataset.x_data[30000:], dataset.y_data[30000:]),
                                                       batch_size=batch,
                                                       shuffle = False)                                        
            
            
            #Example data
            examples = iter(train_loader)
            example_data, example_targets = examples.next()
            print(example_data.shape, example_targets.shape)
            print(len(example_targets))
            
            
            # Model, loss and optimizer
            model = NeuralNet(input_size, hidden_size, output_size)
            
            loss = nn.MSELoss(reduction='mean')
            optimizer = torch.optim.RMSprop(params=model.parameters(),lr = lr)
            

            # Train the model
            n_steps = len(train_loader)
            
            best_model_wts = copy.deepcopy(model.state_dict())

            for epoch in range(epochs):
                
                for phase in ['train', 'test']:
                    
                    if phase == 'train':
                        
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
                                
                        print (f'Epoch [{epoch+1}/{epochs}], Batch_size {batch}, Learning_Rate {lr}, Training_Loss: {torch.sqrt(train_loss/(len(train_loader)*batch))}')
                   
                    else:
                        
                        with torch.no_grad():
                            y_hat = model(dataset.x_data[30000:].float())
                            y_real = dataset.y_data[30000:]
                            l = loss(y_hat, y_real)
                            print(f'Epoch [{epoch+1}/{epochs}], Batch_size {batch}, Learning_Rate {lr}, Testing_Loss :{torch.sqrt(l)}')
                            
                            if torch.sqrt(l).item() < minimum_loss:
                                
                                minimum_loss = torch.sqrt(l).item()
                                best_model_wts = copy.deepcopy(model.state_dict())
                                epoch_no = epoch
                                learning = lr
                                size = batch
                    
            
    model.load_state_dict(best_model_wts)   
    with torch.no_grad():
        y_hat = model(dataset.x_data[30000:].float())
        y_real = dataset.y_data[30000:]
        l = loss(y_hat, y_real)
        print('Minimum Testing Loss :',torch.sqrt(l).item())
        print('Epoch Number:',epoch_no)
        print('Learning Rate: ', learning)
        print('Batch Size: ',size)
    
    #%%
    # save only state dict
    FILE = "saved_model/model_7661.pth"
    torch.save(model.state_dict(), FILE)
    
    loaded_model = NeuralNet(input_size, hidden_size, output_size)
    loaded_model.load_state_dict(torch.load(FILE))
    loaded_model.eval()
    
    with torch.no_grad():
            y_hat = loaded_model(dataset.x_data[30000:].float())
            y_real = dataset.y_data[30000:]
            l = loss(y_hat, y_real)
            print('Minimum Testing Loss :',torch.sqrt(l).item())
            print('Epoch Number:',epoch_no)


