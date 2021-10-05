#%%
import re
from keras.preprocessing.text import text_to_word_sequence
from nltk.stem import WordNetLemmatizer
# import contractions


#%%
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
#%%
categories = ['positive', 'negative']
def preprocess(path):
    lemmatizer = WordNetLemmatizer()
    X = []
    y = []
    with open(path, errors='ignore') as reader:
        
        first_line = True
        for para in reader:
            if(first_line):
                first_line=False
                continue
            para = para.replace('<br />','')
            words = text_to_word_sequence(para,filters='!£$"#%&()*+,-/:;<=>?@[\\]^_`{|}~\t\n')
            # print(words)
            
            new_words = []
            
            for word in words:

                if word=='':
                    continue
                if word == '.':
                    new_words.append('.')
                    continue
    
                
                # removing all alphanumerics
                if re.search(r'\d', word)!=None:
                    # print(word)
                    letters = list(word)
                    if(letters[-1]=="."):
                        new_words.append('.')
                
                # separating words with full stops
                else :
                    if re.search('[a-z]+', word)!=None:
                    
                        num_of_dots = len(re.findall('\.', word))
                        if num_of_dots == 0:
                            
                            if word.isalpha():
                                new_words.append(word)
                                
                        if num_of_dots == 1:
                            
                            if(word[-1]=="."): # fullstop char
                                new_words.append(word[:-1])
                                new_words.append('.')
                            else: # inbetween
                                sep_words = word.split(".")
                                if sep_words[0]!='':
                                    new_words.append(sep_words[0])
                                if sep_words[1]!='':
                                    new_words.append(sep_words[1])
    
                        if num_of_dots > 1:
                            if(word[-1]=="."): # fullstop char there
                                sep_words = word[:-1].split(".")
                                for i in sep_words:
                                    if i!='':
                                        new_words.append(i)
                                new_words.append('.')
                            else:
                                sep_words = word.split(".")
                                for i in sep_words:
                                    if i!='':
                                        new_words.append(i)
                    
            
            category = new_words[-1]

            y.append(category)
            new_words =new_words[:-1]
            
            # ending para with fullstop
            if new_words[-1] !='.':
                new_words.append('.')
            
            new_words_1 = []
            
            for x in new_words:
                if x=='':
                    continue
                x = decontracted(x).split()
        
                if len(x)==1:                    
                    new_words_1.append(lemmatizer.lemmatize(x[0].replace("'","")))
                else:
                    for x1 in x:
                        new_words_1.append(lemmatizer.lemmatize(x1))
                    
            X.append(new_words_1)
            # print(new_words_1)
    
    return X,y

#%%
# train_path = "IMDB Dataset.csv"
# x,y = preprocess(train_path)

#%%


