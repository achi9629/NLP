from os import listdir
import re
from nltk.stem import WordNetLemmatizer
from keras.preprocessing.text import text_to_word_sequence
import numpy as np
from sklearn.model_selection import train_test_split

  


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
        lemmarized_sentences.append(' '.join(words))

    
    return lemmarized_sentences

def one_hot_encoding(Y):
    e = np.eye(2)
    Y = e[Y]
    return Y

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
    # Y_train = one_hot_encoding(Y_train)
    # Y_test = one_hot_encoding(Y_test)
    
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
    
    
    return sentences_cleaned_train, Y_train, sentences_cleaned_test, Y_test
    

#%%
from tfidf import vectorize
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import  SVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

#%%

sentences_cleaned_train, Y_train, sentences_cleaned_test, Y_test = data_extract()
vectorizer = vectorize()

X = vectorizer.fit_transform(sentences_cleaned_train)
X_test = vectorizer.transform(sentences_cleaned_test)

X_train, X_val, Y_train, Y_val = train_test_split(X, Y_train, 
                                                  test_size = .20, 
                                                  shuffle = (True),
                                                  random_state = 1234)

# print(X_train.shape, X_val.shape, Y_train.shape, Y_val.shape)

modesl = ['KNN', 'SVC', 'DT', 'RF', 'Logistic', 'Naive']
classifiers = [ KNeighborsClassifier(100),
                SVC(C=1),
                DecisionTreeClassifier(max_depth=None),  
                RandomForestClassifier(n_estimators=5, max_depth=None),
                LogisticRegression(),
                MultinomialNB()
                ]

for i, model in enumerate(classifiers):
    
    obj = model
    obj.fit(X_train, Y_train)
    Y_hat = obj.predict(X_val)
    
    print(modesl[i],' - Validation score :',accuracy_score(Y_val, Y_hat))
    
Y_hat_test = obj.predict(X_test)
print('Test score :',accuracy_score(Y_test, Y_hat_test))



