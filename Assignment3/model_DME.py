# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 19:44:45 2021

@author: HP
"""
# from keras.utils import multiply
from keras.layers import LSTM, Dense, Activation, Bidirectional, Maximum, Input, Embedding, Reshape, Concatenate, Lambda, multiply
from keras.models import Model
import keras.backend as K


def concatenate_emb(word_index, list_emb, maxlen):
    inputs = []
    outputs = []
    
    input1 = Input(shape=(maxlen,))
    emb1 = Embedding(len(word_index)+1, 300, weights=[list_emb[0]], trainable=False)(input1)
    re1 = Reshape((-1,300,1))(emb1)
    inputs.append(input1)
    outputs.append(re1)
    
    input2 = Input(shape=(maxlen,))
    emb2 = Embedding(len(word_index)+1, 300, weights=[list_emb[1]], trainable=False)(input2)
    re2 = Reshape((-1,300,1))(emb2)
    inputs.append(input2)
    outputs.append(re2)
    
    input3 = Input(shape=(maxlen,))
    emb3 = Embedding(len(word_index)+1, 300, weights=[list_emb[2]], trainable=False)(input3)
    re3 = Reshape((-1,300,1))(emb3)
    inputs.append(input3)
    outputs.append(re3)
    
    concat = Concatenate(axis = 3)(outputs)
    return Model(inputs, concat)


def dynamic_me(maxlen):
    
    inp = Input(shape=(maxlen, 300, 3))
    re1 = Reshape((maxlen, 300*3))(inp)
    dense = LSTM(3, activation='softmax')(re1)
    re2 = Reshape((maxlen, 1, 3))(dense)
    # mul = multiply([inp, re2])
    mul = inp*re2
    output = Lambda(lambda l: K.sum(l, axis=3))(mul)
    
    return Model(inp, output)

def CDME(maxlen, ):

    inp = Input(shape=(maxlen, 300, 3))
    x = Reshape((maxlen, -1))(inp)
    x = Bidirectional(LSTM(3, return_sequences=True))(x)
    x = Lambda(lambda t: [t[:,:,:int(3)],  
                          t[:,:,int(3):]])(x)
    x = Maximum()(x)
    x = Activation('sigmoid')(x)
    x = Reshape((maxlen, 1, 3))(x)
    x = multiply([inp, x])
    print('x : ',x.shape)
    out = Lambda(lambda t: K.sum(t, axis=-1))(x)
    print('out',out.shape)
    return Model(inp, out)

def dynamic_embedding(word_index, 
                      embedding_matrix_word2vec, 
                      embedding_matrix_glove, 
                      embeddings_matrix_fast, maxlen):
    
    read_emb = concatenate_emb(word_index, [embedding_matrix_word2vec, 
                                            embeddings_matrix_fast, 
                                            embedding_matrix_glove], 
                                            maxlen=maxlen)
    
    dme = CDME(maxlen)
    x = dme(read_emb.output)
    x = LSTM(128, )(x)
    # # print(x[0].shape, x[1].shape, x[2].shape)
    print(x.shape)
    # # x = LSTM(64)(x)
    # # print(x[0].shape, x[1].shape, x[2].shape)
    # x = Dense(20, activation="relu")(x)
    # print(x.shape)
    # x = Dense(20, activation="relu")(x)
    # out = Dense(2, activation='softmax')(x)
    # print(out.shape)
    # model = Model(read_emb.input, out)
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # # model.summary()

    
    return model