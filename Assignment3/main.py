# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 16:58:39 2021

@author: HP
"""
from preprocess import data_extract
from glove import glove
from fasttext import load_vectors
from word2vec import word2vec
from model_glove import Glove
from model_word2vec import word
from model_fasttext import fasttext
from model_DME import dynamic_embedding
from sklearn.model_selection import train_test_split

#%%
if __name__=='__main__':
    train_padded, Y_train, test_padded, Y_test, word_index, maxlen = data_extract()
    X_train, X_val, Y_train, Y_val = train_test_split(train_padded, Y_train, 
                                                      test_size = 0.2, 
                                                      shuffle = True, 
                                                      random_state = 1234)
    
    
    #%%
    embedding_dim = 300
    embedding_matrix_glove = glove(word_index, embedding_dim) 
    model = Glove(word_index, embedding_dim, embedding_matrix_glove)

    #%%     
    embeddings_matrix_fast = load_vectors(word_index, embedding_dim) 
    model = fasttext(word_index, embedding_dim, embeddings_matrix_fast)
    
    #%%
    embedding_matrix_word2vec = word2vec(word_index, embedding_dim)    
    model = word(word_index, embedding_dim, embedding_matrix_word2vec)
    
    #%%
    model = dynamic_embedding(word_index, embedding_matrix_word2vec, embedding_matrix_glove, embeddings_matrix_fast, maxlen)
    # 

    #%%
    history = model.fit([X_train]*3, Y_train, 
                            batch_size=256, 
                            epochs=10, verbose=2, 
                            validation_data= ([X_val]*3, Y_val))
    
    #%%
    model.evaluate([X_val], Y_val)
    model.evaluate([test_padded], Y_test)
