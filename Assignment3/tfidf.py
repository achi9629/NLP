# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 21:42:33 2021

@author: HP
"""

from sklearn.feature_extraction.text import TfidfVectorizer 

def vectorize():
    
    vectorizer = TfidfVectorizer(analyzer='word', 
                                 stop_words='english',
                                  # lowercase=(True))
                                    # max_features = 5000)
                                )
    return vectorizer