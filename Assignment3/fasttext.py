# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 21:19:05 2021

@author: HP
"""

import os
from tqdm import tqdm
import numpy as np
import pickle

def load_vectors(word_index, embedding_dim):

    # fname = ''
    # embeddings_index = {} # empty dictionary
    # f = open(os.path.join(fname, 'wiki-news-300d-1M.vec'), encoding="utf-8")
    # for line in tqdm(f):
    #     values = line.split()
    #     word = values[0]
    #     coefs = np.asarray(values[1:], dtype='float32')
    #     embeddings_index[word] = coefs
    # f.close()
    # print('Found %s word vectors.' % len(embeddings_index))
    
    # embedding_matrix_fast = np.random.rand(len(word_index)+1, embedding_dim)

    # for word, i in word_index.items():
    #     #if i < max_words:
    #     embedding_vector = embeddings_index.get(word)
    #     if embedding_vector is not None:

    #         embedding_matrix_fast[i] = embedding_vector
            
    # dbfile = open('embedding_matrix_fast.pickle', 'ab')
    # pickle.dump(embedding_matrix_fast, dbfile)                     
    # dbfile.close()
        
    dbfile = open('embedding_matrix_fast.pickle', 'rb')     
    embedding_matrix_fast_loaded = pickle.load(dbfile)
    dbfile.close()
    
    
    return embedding_matrix_fast_loaded