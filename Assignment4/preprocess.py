#%%
import re
from keras.preprocessing.text import text_to_word_sequence

#%%
def preprocess(path):
    
    X = []
    y = []
    with open(path) as reader:
        
        line = [True]
        for sentence in reader:
            if(line):
                line=False
                continue
            
            sentence1 = text_to_word_sequence(sentence,filters='!"#%&()*+,-/:;<=>?@[\\]^_`{|}~\t\n')
            # print(words)
            
            new_sentence = []
            
            for word in sentence1:

                if word=='':
                    continue
                if word == '.':
                    new_sentence.append('.')
                    continue
                
                               
                if re.search('\$', word)!=None:
                    new_sentence.append('$')
                    letters = list(word)
                    if(letters[-1]=="."):
                        new_sentence.append('.')
                        
                elif re.search('£', word)!=None:
                    new_sentence.append('£')
                    letters = list(word)
                    if(letters[-1]=="."):
                        new_sentence.append('.')
                        
                
                elif re.search(r'\d', word)!=None:
                    # print(word)
                    letters = list(word)
                    if(letters[-1]=="."):
                        new_sentence.append('.')
                
                else :
                    if re.search('[a-z]+', word)!=None:
                    
                        num_of_dots = len(re.findall('\.', word))
                        if num_of_dots == 0:
                            
                            if word.isalpha():
                                new_sentence.append(word)
                                
                        if num_of_dots == 1:
                            
                            if(word[-1]=="."): 
                                new_sentence.append(word[:-1])
                                new_sentence.append('.')
                            else: 
                                sep_words = word.split(".")
                                if sep_words[0]!='':
                                    new_sentence.append(sep_words[0])
                                if sep_words[1]!='':
                                    new_sentence.append(sep_words[1])
    
                        if num_of_dots > 1:
                            if(word[-1]=="."): 
                                sep_words = word[:-1].split(".")
                                for i in sep_words:
                                    if i!='':
                                        new_sentence.append(i)
                                new_sentence.append('.')
                            else:
                                sep_words = word.split(".")
                                for i in sep_words:
                                    if i!='':
                                        new_sentence.append(i)
                    
        
            category = new_sentence[-1]

            y.append(category)
            new_words =new_sentence[:-1]
            

            if new_sentence[-1] !='.':
                new_sentence.append('.')

            X.append(new_sentence)
    
    return X,y
