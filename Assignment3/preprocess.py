# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from os import listdir
import re
from nltk.stem import WordNetLemmatizer
from keras.preprocessing.text import text_to_word_sequence
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
  


def decontracted(phrase):
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " ", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    # phrase = re.sub(r"\'", "", phrase)
    return phrase

def preprocess(sentence):
    
    words3 = text_to_word_sequence(sentence)
    
    words4 = []
    for c in words3:
        if c.isdigit():
            if len(c) <= 2:
                words4.append(c)
            else:
                continue
        else :
            words4.append(c)
            
    words4 = [s for s in words4 if s != ""]
    
    words5 = []
    for c in words4:
        if c.isalnum() and c.isalpha():
            words5.append(c)
            
    words5 = [s for s in words5 if s != ""]
    
    words6 = [] 
    for word in words5:
        if not word.isdigit() and len(word) > 2:
            words6.append(word) 
            
    
    words7 = [s for s in words6 if s != ""]
    
    words8 = [word.lower() for word in words7]
    
    return words8

def lemmatizer(sentences):
    
    lemmatizer = WordNetLemmatizer()
      
    lemmarized_sentences = []
    
    for sentence in sentences:
        words = []
        for word in sentence:
            words.append(lemmatizer.lemmatize(word))
        lemmarized_sentences.append(words)

    
    return lemmarized_sentences

def one_hot_encoding(Y):
    e = np.eye(2)
    Y = e[Y]
    return Y

def train_valid(train_padded, Y):
    
    index = np.arange(len(Y))
    for i  in range(5): index = np.random.permutation(index)
    Y_train = Y[index]
    train_padded = train_padded[index]
    X_train, X_val, Y_train, Y_val = train_test_split(train_padded, Y, 
                                                      test_size = 0.2, 
                                                      shuffle = True, 
                                                      random_state = 1234)
    
    return X_train, X_val, Y_train, Y_val

def data_extract():
    
    path = 'rt-polaritydata'
    folders = listdir(path)
    print('Classes :',folders)
    
    with open(path + '/' + folders[1], "r", encoding='utf-8', errors='ignore') as reader:
        sentences = [line for line in reader]
        
    Y = [1]*len(sentences)
    
    sentences_test = sentences[4500:]
    sentences_train = sentences[:4500]

    Y_train = Y[:4500]
    Y_test = Y[4500:]
    
    with open(path + '/' + folders[0], "r", encoding='utf-8', errors='ignore') as reader:
        sentences = [line for line in reader]
        
    Y = [0]*len(sentences)
    
    sentences_test = sentences_test + sentences[4500:]
    sentences_train = sentences_train + sentences[:4500]

    Y_train = Y_train + Y[:4500]
    Y_test = Y_test + Y[4500:]
    Y_train = one_hot_encoding(Y_train)
    Y_test = one_hot_encoding(Y_test)
    
    sentences_list_train = []
    sentences_list_test = []
    
    for sentence in sentences_train:
        sentences_list_train.append(decontracted(sentence))
        
    for sentence in sentences_test:
        sentences_list_test.append(decontracted(sentence))
             
    sentences_cleaned_train = []
    sentences_cleaned_test = []
    
    for sentence in sentences_list_train:
        sentences_cleaned_train.append(preprocess(sentence))
        
    for sentence in sentences_list_test:
        sentences_cleaned_test.append(preprocess(sentence))
    
    sentences_cleaned_test = lemmatizer(sentences_cleaned_test)
    sentences_cleaned_train = lemmatizer(sentences_cleaned_train)
    
    train_padded, test_padded, word_index, maxlen = tokenization(sentences_cleaned_train, sentences_cleaned_test)
    
    return train_padded, Y_train, test_padded, Y_test, word_index, maxlen
    
def tokenization(sentences_cleaned_train, sentences_cleaned_test):
    
    oov_token = '<UNK>'
    pad_type = 'post'
    trunc_type = 'post'

        
    tokenizer = Tokenizer(oov_token = oov_token)     
    tokenizer.fit_on_texts(sentences_cleaned_train)
    word_index = tokenizer.word_index

    train_sequences = tokenizer.texts_to_sequences(sentences_cleaned_train)
    # maxlen = max([len(x) for x in train_sequences]) 
    maxlen = 100
    train_padded = pad_sequences(train_sequences, padding = pad_type, 
                                  truncating = trunc_type, maxlen = maxlen)
        

        
    test_sequences = tokenizer.texts_to_sequences(sentences_cleaned_test)
    test_padded = pad_sequences(test_sequences, padding = pad_type, 
                                truncating = trunc_type, maxlen = maxlen)
    
    return train_padded, test_padded, word_index, maxlen
    

    
    
    
    
    
    









