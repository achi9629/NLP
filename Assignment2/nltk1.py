# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 17:15:26 2021

@author: HP
"""

from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

def function(sentences):
    
    outputs = []
    outputs1 = []
    lemmatizer = WordNetLemmatizer()
    ps = PorterStemmer()

    # for sentence in sentences:
    #     outputs.append("".join([ps.stem(i) for i in sentence]))
    for output in sentences:
        outputs1.append("".join([lemmatizer.lemmatize(i) for i in output]))
        
    return outputs1