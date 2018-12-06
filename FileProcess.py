
# coding: utf-8

# In[2]:



import pandas as pd
import numpy as np
import tensorflow as tf
import time

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import gensim


# In[ ]:


window_size=2;
stop_words_list=set(stopwords.words('english'))
def read_data(file_name):
    with open(file_name, 'r') as f:
        word_vocab = set()  # not using list to avoid duplicate entry
        word_count = {};
        word_index={};

        word2vector = {}
        count=0;
        for line in f:
            line_ = line.strip()  # Remove white space
            words_Vec = line_.split()
            word_vocab.add(words_Vec[0])
            word_index[words_Vec[0]]=count;
            count=count+1;
            word2vector[words_Vec[0]] = np.array(words_Vec[1:], dtype=float)
    print("Total Words in DataSet:", len(word_vocab))
    return word_vocab, word2vector, word_index

vocab, w2v,word_index = read_data("glove_6B_200d.txt");


full_data=pd.read_csv('./data.csv');
print("shape of data:",np.shape(full_data))




catch_df=full_data[full_data['Id'].str.startswith(('c'))]
sent_df=full_data[full_data['Id'].str.startswith(('s'))]


#utiil
def get_word_count(string):
    tokens = string.split()
    n_tokens = len(tokens)
    return n_tokens  

def ReLU(x):
    return abs(x) * (x > 0)


def get_features(inputs):
    out=np.zeros((len(inputs),200));
    for i in range(len(inputs)):
        try:
            c = w2v[inputs[i]]
        except KeyError:
            c = np.zeros((1,200));
        out[i]=c;
    
    return np.transpose(out);
 

def lookup_indexes(sentences):
    
    sentence_indexes=[];
    for word in sentences:
        if word in word_index:
            sentence_indexes.append(word_index[word]);
        else:
            #print("unknown word",word)
            sentence_indexes.append(word_index['unk']);
        
    return sentence_indexes;


# In[ ]:


def get_features_row(file_id,Id,text,catch_words):
    text=text.lower();
    
    word_count=get_word_count(text);
    words=text.split()
    input_filter=np.random.rand(300,200*(2*window_size+1));
    sentence_vector=np.zeros((300,1));
    count=0;
    rows_list=[]
    if word_count>2*window_size+1:
        for i in range(window_size,word_count-window_size):
            
            inputs=words[i-window_size:i+window_size+1];
            inputs=lookup_indexes(inputs)
            is_catchword = 1 if words[i] in catch_words else 0
            
            rows_list.append({"file_id":file_id, "Id":Id, "words":inputs, "is_catchword":is_catchword})
    return rows_list
   


# In[ ]:


def get_catch_words(catch_df):
    catch_words=[]
    for index, row in catch_df.iterrows():
        
        words=word_tokenize(row['text']);
        temp=[word for word in words if not word in stop_words_list]
        catch_words=catch_words+temp;
    return catch_words;


# In[ ]:


def get_dataframe(file_name):
    file_data=full_data[full_data.file_id==file_name];
    catch_df=file_data[file_data['Id'].str.startswith(('c'))]
    catch_words=get_catch_words(catch_df);
    
    sent_df=file_data[file_data['Id'].str.startswith(('s'))]
    sent_df = sent_df[:-1]
    file_dataframe=[];
    count=0;
    for index, row in sent_df.iterrows():
        count=count+1;
        #print(get_features_row_dummy(row['file_id'],row['Id'],row['text']))
        temp=get_features_row(row['file_id'],row['Id'],row['text'],catch_words)
        if len(temp)>0:
            file_dataframe=file_dataframe+temp;

    return pd.DataFrame(file_dataframe);


# In[ ]:


print(get_dataframe('08_1056.xml'))

