# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 22:07:47 2021

@author: HP
"""

from tqdm import tqdm
import numpy as np
import pickle

def word2vec(word_index, embedding_dim):

    # dbfile = open('wv.pickle', 'rb')     
    # wv = pickle.load(dbfile)
    # dbfile.close()
    
    # embedding_matrix_word2vec = np.random.rand(len(word_index)+1, embedding_dim)

    # for word, i in tqdm(word_index.items()):
    #     try:
    #         embedding_matrix_word2vec[i] = wv[word]
    #     except:
    #         pass
            
    # dbfile = open('embedding_matrix_word2vec.pickle', 'ab')
    # pickle.dump(embedding_matrix_word2vec, dbfile)                     
    # dbfile.close()
    
        
    dbfile = open('embedding_matrix_word2vec.pickle', 'rb')     
    embedding_matrix_word2vec_loaded = pickle.load(dbfile)
    dbfile.close()
    
    
    
    return embedding_matrix_word2vec_loaded
