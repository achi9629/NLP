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
from sklearn.neural_network import MLPClassifier
import pickle
from nltk1 import function
from visualization import visualize_tsne, visualize_SVD

#%%
train_path = '20news-bydate-train'
test_path = '20news-bydate-test'
    
sentences, Y = preprocess(train_path)
sentences_test, Y_test = preprocess(test_path)

vectorizer = vectorize()

output = function(sentences)
X = vectorizer.fit_transform(sentences)

#%%
visualize_tsne(X, Y)

#%%
filename = '9772.sav'
file_train = 'X_train.pkl'
file_val = 'X_val.pkl'
file_test = 'X_test.pkl'
test_path = '20news-bydate-test'

#%%
loaded_model = pickle.load(open(filename, 'rb'))
loaded_X_train = pickle.load(open(file_train, 'rb'))
loaded_X_val = pickle.load(open(file_val, 'rb'))
loaded_X_test = pickle.load(open(file_test, 'rb'))

sentences_test, Y_test = preprocess(test_path)

Y_hat_test1 = loaded_model.predict(loaded_X_test)
print('Test score :',f1_score(Y_test, Y_hat_test1, average='micro'))

#%%
