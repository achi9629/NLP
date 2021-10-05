# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 20:26:12 2021

@author: HP
"""

from preprocessing import preprocess
from tfidf import vectorize
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import  SVC
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import pickle
from visualization import visualize_tsne
from nltk1 import function

#%%

train_path = '20news-bydate-train'
test_path = '20news-bydate-test'
    
sentences, Y = preprocess(train_path)

# output_train = function(sentences)

sentences_test, Y_test = preprocess(test_path)

# output_test = function(sentences_test)

vectorizer = vectorize()

X = vectorizer.fit_transform(sentences)
X_test = vectorizer.transform(sentences_test)

#%%
# TSNE
visualize_tsne(X, Y)
visualize_tsne(X_test, Y_test)

#%%

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, 
                                                  test_size = .20, 
                                                  shuffle = (True),
                                                  random_state = 1234)

print(X_train.shape, X_val.shape, Y_train.shape, Y_val.shape)

modesl = ['KNN', 'SVC', 'DT', 'RF', 'Logistic', 'Naive']
classifiers = [ KNeighborsClassifier(100),
                SVC(C=1),
                DecisionTreeClassifier(max_depth=None),  
                RandomForestClassifier(n_estimators=5, max_depth=None),
                LogisticRegression(),
                MultinomialNB()
                ]
                
#%%
for i, model in enumerate(classifiers):
    
    obj = model
    obj.fit(X_train, Y_train)
    Y_hat = obj.predict(X_val)
    
    print(modesl[i],' - Validation score :',f1_score(Y_val, Y_hat, average='micro'))
    
#%%
#Best validation accuracy is on Naive Bayes
obj = MultinomialNB()
obj.fit(X_train, Y_train)
Y_hat = obj.predict(X_val)
print('Validation acc :',f1_score(Y_val, Y_hat, average='micro'))

#%%
#Testing
Y_hat_test = obj.predict(X_test)
print('Test score :',f1_score(Y_test, Y_hat_test, average='micro'))


#%%
filename = '9772.sav'
file_train = 'X_train.pkl'
file_val = 'X_val.pkl'
file_test = 'X_test.pkl'
test_path = '20news-bydate-test'

#%%
#Saving Files
# pickle.dump(obj, open(filename, 'wb'))
# pickle.dump(X_train, open(file_train, 'wb'))
# pickle.dump(X_val, open(file_val, 'wb'))
# pickle.dump(X_test, open(file_test, 'wb'))

#%%
#Loading Model
loaded_model = pickle.load(open(filename, 'rb'))
loaded_X_train = pickle.load(open(file_train, 'rb'))
loaded_X_val = pickle.load(open(file_val, 'rb'))
loaded_X_test = pickle.load(open(file_test, 'rb'))

sentences_test, Y_test = preprocess(test_path)

Y_hat_test1 = loaded_model.predict(loaded_X_test)
print('Test score :',f1_score(Y_test, Y_hat_test1, average='micro'))