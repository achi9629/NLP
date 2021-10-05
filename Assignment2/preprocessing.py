# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 20:16:05 2021

@author: HP
"""
from os import listdir
import string
from collections import Counter
import numpy as np

# Removing unnecessary letters
def remove_unnecessary(words1):
    
    #Removing \t 
    table = str.maketrans('', '', '\t')
    words1 = [word.translate(table) for word in words1]
    
    #Removing \n
    table = str.maketrans('', '', '\n')
    words2 = [word.translate(table) for word in words1]
    
    # Removing punctuations except '
    table = str.maketrans('', '', (string.punctuation).replace("'", ""))
    words3 = [word.translate(table) for word in words2]
    
    # #Removing empty words 
    words3 = [s for s in words3 if s != ""]
    
    #Removing inverted commas and first inverted comma
    words4 = []
    for word in words3:
        if (word[0] and word[len(word)-1] == "'"):
            word = word[1:len(word)-1]
        elif(word[0] == "'"):
            word = word[1:len(word)]
        else:
            word = word
        words4.append(word)
    
    # To lower case
    words7 = [word.lower() for word in words4]

    return words7

#Strip sentences to words
def tokenization(lines):
    sentence = []
    for line in lines:
        words = remove_unnecessary(line.split(' '))
        sentence.append(words)
    return sentence

#Remove empty lists
def remove_empty_lists(l):
    
    sent = []
    for i in l:
        if len(i) == 0:
            continue
        sent.append(' '.join(i))
        
    return sent

#Read every document
def read_documents(file_names):
    documents = []
    for file in file_names:    
        document = []
        f = open( file, 'r')
        x = f.readlines()
        documents.append(x)
        

    doc_words = []
    for document in documents:
        doc_words.append(' '.join(remove_empty_lists(tokenization(document))))
        
    return doc_words

#Start preprocessing
def preprocess(path):
    
    folders = listdir(path)
    print('Classes :',folders)
    
    files = []
    for folder in folders:
        files.append([path+'/'+folder+'/'+f for f in listdir(path+'/'+folder)])
        
    print('train_set :',sum([len(i) for i in files]))
    # print('sample_document :',files[0][0])
    
    Y = []
    for i in range(len(files)):
        for j in range(len(files[i])):
            Y.append(i)
            
    print('Each Class frequency :',Counter(Y))
    
    
    file_names = []
    for i in range(len(files)):
        for f in files[i]:
            file_names.append(f)
    
    documents = read_documents(file_names)      
    return documents, np.array(Y)

# documents, Y = preprocess('20news-bydate-train')