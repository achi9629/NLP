# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 03:25:05 2021

@author: HP
"""

from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt

def visualize_tsne(X, Y):
    perplexities = [1, 5, 10, 15, 20, 25, 30, 35, 50, 100]
    for perplexity in perplexities:
        X_embedded = TSNE(n_components=2, perplexity = perplexity).fit_transform(X)  
        print('perplexity :' ,perplexity, X_embedded.shape)
        
        color = [ 'red' if y ==0 else 'green' if y==1 else 'blue' for y in Y]
        
        fig, ax = plt.subplots()
        plt.scatter(X_embedded[:,0], X_embedded[:,1], c = color, label=perplexity)
        ax.legend()
        plt.grid()
        plt.show()
    
#     fig, ax = plt.subplots()
# mpl.rcParams['figure.dpi'] = 500
# ax.plot(s,tau_1, 'blue',label='Tau1')
# plt.plot(s,tau_2,'orange',label = 'Tau2')
# plt.grid()
# ax.legend()
# plt.show()