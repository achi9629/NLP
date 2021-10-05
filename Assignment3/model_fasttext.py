# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 15:16:01 2021

@author: HP
"""

from keras.layers import Embedding, LSTM, Dense, Bidirectional, Input, Concatenate, Dropout
from keras.models import Model, Sequential
from Attention import Attention

def fasttext(word_index, embedding_dim, embeddings_matrix_fast):
    
    model = Sequential()
    model.add(Embedding(len(word_index)+1, embedding_dim, weights = [embeddings_matrix_fast], trainable=False))
    model.add(Bidirectional(LSTM(300,)))
    # model.add(Attention(10))
    model.add(Dense(20, activation="relu"))
    model.add(Dense(20, activation="relu"))
    model.add(Dense(2,activation='softmax'))
    # print(model.layers[0])
    # model.layers[0].set_weights([embeddings_matrix_fast])
    # model.layers[0].trainable = False

    print(model.summary())
    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
    
    return model