# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 19:07:10 2021

@author: HP
"""

import torch.nn as nn
import torch
from sklearn.metrics import f1_score
import numpy as np

class ANN(nn.Module):
    
    def __init__(self, in_features, hid_features, out_features):
        
        super(ANN, self).__init__()
        
        self.lin1 = nn.Linear(in_features, hid_features)
        self.lin2 = nn.Linear(hid_features, hid_features)
        self.lin3 = nn.Linear(hid_features, hid_features)
        self.lin4 = nn.Linear(hid_features, out_features)
        self.act = nn.Sigmoid()
        
    def forward(self, X):
        out = self.lin1(X)
        out = self.act(out)
        
        out = self.lin2(out)
        out = self.act(out)
        
        out = self.lin3(out)
        out = self.act(out)
        
        out = self.lin4(out)
        
        return out
 
    
def sp_mx_sp_tensor(f):
    sparse_mx = f.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    matrix = torch.sparse.FloatTensor(indices, values, shape)  
    return matrix
 
def main(X_train_f, Y_train, X_val_f, Y_val, X_test_f, Y_test):
    
    in_features = X_train_f.shape[1]
    hid_features = 100
    out_features = 3
    epochs = 200
    lr = 0.001
    
    model = ANN(in_features, hid_features, out_features)
    loss = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)
    
    print(type(X_train_f))
    # X_train = sp_mx_sp_tensor(X_train_f)
    # X_val   = sp_mx_sp_tensor(X_val_f)
    # X_test  = sp_mx_sp_tensor(X_test_f)
    
    # Y_train = torch.tensor(Y_train)
    # Y_val = torch.tensor(Y_val)
    # Y_test = torch.tensor(Y_test)
    
    # for epoch in range(epochs):
        
    #     Y_hat = model(X_train)
        
    #     l = loss(Y_hat, Y_train)
        
    #     l.backward()       
    #     optimizer.step()
    #     optimizer.zero_grad()
        
    #     if (epoch+1)%10 ==0:
    #         train_score = f1_score(Y_hat, Y_train)
    #         Y_hat_test = model(X_val)
    #         val_score = f1_score(Y_hat_test, Y_val)
    #         l_val = loss(Y_hat_test, Y_val)
    #         print(f' epoch {epoch+1}/{epochs}, loss {l.item()} train score {train_score} val loss {l_val} val score {val_score}')
        
    # Y_hat_test = model(X_test)
    # l_test = loss(Y_test, Y_hat_test)
    # test_score = f1_score(Y_hat_test, Y_test)
    # print(f'test loss {l_test} test score {test_score}')
