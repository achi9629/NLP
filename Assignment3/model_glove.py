# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 15:12:17 2021

@author: HP
"""
from keras.layers import Embedding, LSTM, Dense, Bidirectional, Input, Concatenate, Dropout
from keras.models import Model, Sequential
from Attention import Attention

def Glove(word_index, embedding_dim, embeddings_matrix_glove):
    
    model = Sequential()
    model.add(Embedding(len(word_index)+1, embedding_dim))
    model.add(Bidirectional(LSTM(300,)))
    # model.add(Attention(10))
    model.add(Dense(20, activation="relu"))
    model.add(Dense(20, activation="relu"))
    model.add(Dense(2,activation='softmax'))
    # encoder_inputs = Input(shape=(None,))
    # embedded_sequences = Embedding(vocab_size, embedding_dim)(encoder_inputs)
    # lstm = LSTM(300, return_sequences = True)(embedded_sequences)
    # # (lstm, forward_h, forward_c, backward_h, backward_c) = Bidirectional(LSTM(300, 
    # #                                                                           return_sequences=True, 
    # #                                                                           return_state=True), 
    # #                                                                           name="bi_lstm_1")(lstm)
    # # state_h = Concatenate()([forward_h, backward_h])
    # # state_c = Concatenate()([forward_c, backward_c])
    # # context_vector, attention_weights = Attention(10)(lstm, state_h)
    # dense1 = Dense(20, activation="relu")(lstm)
    # dropout = c(dense1)
    # output = Dense(2,activation='softmax')(dropout)

    # model = Model(inputs=encoder_inputs, outputs=output)
    print(model.layers[0])
    model.layers[0].set_weights([embeddings_matrix_glove])
    model.layers[0].trainable = False

    print(model.summary())
    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
    
    return model