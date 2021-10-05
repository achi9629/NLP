# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 16:30:23 2021

@author: HP
"""

# Load Glove vectors
import os
import numpy as np
import pickle
from tqdm import tqdm

def glove(word_index, embedding_dim):
    
    # glove_dir = ''
    # embeddings_index = {} # empty dictionary
    # f = open(os.path.join(glove_dir, 'glove.6B.300d.txt'), encoding="utf-8")

    # for line in tqdm(f):
    #     values = line.split()
    #     word = values[0]
    #     coefs = np.asarray(values[1:], dtype='float32')
    #     embeddings_index[word] = coefs
    # f.close()
    # print('Found %s word vectors.' % len(embeddings_index))
    # # print(embeddings_index['the'])

    # embedding_matrix_glove = np.random.rand(len(word_index)+1, embedding_dim)

    # for word, i in word_index.items():
    #     embedding_vector = embeddings_index.get(word)
    #     if embedding_vector is not None:
    #         embedding_matrix_glove[i] = embedding_vector
            
    # dbfile = open('embedding_matrix_glove.pickle', 'ab')
    # pickle.dump(embedding_matrix_glove, dbfile)                     
    # dbfile.close()
    
        
    dbfile = open('embedding_matrix_glove.pickle', 'rb')     
    embedding_matrix_glove_loaded = pickle.load(dbfile)
    dbfile.close()
    
    return embedding_matrix_glove_loaded