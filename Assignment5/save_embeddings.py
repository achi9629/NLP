# save embeddings
#%%
from preprocess1 import preprocess 
from gensim.models.keyedvectors import KeyedVectors
import pickle
import numpy as np

#%%
# paths
train_path = "IMDB Dataset.csv"

#%%
# import data
X_full,Y_full = preprocess(train_path)
X = X_full[:25000]
y = Y_full[:25000]

#%%
dbfile = open('wv.pickle', 'rb')     
wv = pickle.load(dbfile)
dbfile.close()
# model = KeyedVectors.load_word2vec_format("wv.pickle", binary=True)

#%%
word2vec_dict = {}
notp = []
count=0
for article in X:
    for word in article:
        # print(word)
        try:
            wordvec = wv[word]
        except KeyError:
            count+=1
            # print(word)
            notp.append(word)
            continue
            # wordvec = OOV
            
        word2vec_dict[word] = wordvec
# print(set(notp))
print(count)
#987216
#%%
# store word2vec_dict as pickle for later

with open('word2vec_dict.pickle', 'wb') as handle:
    pickle.dump(word2vec_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('word2vec_dict.pickle', 'rb') as handle:
#     word2vec_dict = pickle.load(handle)



#%%
fasttext_o_dict = {}
with open("wiki-news-300d-1M.vec", errors='ignore') as reader:      
    for line in reader:
        line = line.strip().split(" ")
        fasttext_o_dict[line[0]] = np.asarray(line[1:], dtype=np.float32)
#         # print(line)


#%%
fasttext_dict = {}
notp = []
count=0
for article in X:
    for word in article:
        # print(word)
        try:
            wordvec = fasttext_o_dict[word]
        except KeyError:
            count+=1
            # print(word)
            notp.append(word)
            continue
            # wordvec = OOV
            
        fasttext_dict[word] = wordvec
# print(set(notp))
print(count)
#61041
#%%
# store fasttext_dict as pickle for later

with open('fasttext_dict.pickle', 'wb') as handle:
    pickle.dump(fasttext_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('fasttext_dict.pickle', 'rb') as handle:
#     fasttext_dict = pickle.load(handle)
    
#%%
glove_o_dict = {}
with open("glove.6B.300d.txt", errors='ignore') as reader:      
    for line in reader:
        line = line.strip().split(" ")
        glove_o_dict[line[0]] = np.asarray(line[1:], dtype=np.float32)
        # print(line)
#%%
glove_dict = {}
notp = []
total_words = 0
count=0
for article in X:
    for word in article:
        total_words+=1
        # print(word)
        try:
            wordvec = glove_o_dict[word]
        except KeyError:
            count+=1
            # print(word)
            notp.append(word)
            continue
            # wordvec = OOV
            
        glove_dict[word] = wordvec
# print(set(notp))
print(count)
print(total_words)
#20493
#5863142
#%%
# store glove_dict as pickle for later

with open('glove_dict.pickle', 'wb') as handle:
    pickle.dump(glove_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('glove_dict.pickle', 'rb') as handle:
#     glove_dict = pickle.load(handle)
    