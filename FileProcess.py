# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import tensorflow as tf
import time
import sys
import tensorflow_hub as hub
import pandas as pd
import numpy as np
import os
from os.path import isfile,join
import xml.etree.cElementTree as ET
import pickle

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import gensim
import enchant
dirpath = "corpus/fulltext/"
embedding_filename = "word2vec.model" 
#embedding_filename = "w2v_cbow_all_size50_window10" 
# In[ ]:


start = time.time()
window_size = 2;
embedding_size = 50
stop_words_list = set(stopwords.words('english'))


dictionary, word_embeds = pickle.load(open(embedding_filename, 'rb'))



f=open('filtered_words.txt', 'r')
words = f.readlines()
word_index = {}
simplied_embedding_array = np.zeros((len(words), embedding_size))
foundEmbed = 0
for i in range(len(simplied_embedding_array)):
    index = -1
    w = words[i].strip()
    if w in dictionary:
        index = dictionary[w]
    elif w.lower() in dictionary:
        index = dictionary[w.lower()]
    if index >= 0:
        foundEmbed += 1
        simplied_embedding_array[i] = word_embeds[index]
    else:
        simplied_embedding_array[i] = np.random.rand(embedding_size) * 0.02 - 0.01
    word_index[w] = i
    
print ("Found embeddings: ", foundEmbed, "/", len(words))
print ('word_index len = ', len(word_index))

embedding_array = simplied_embedding_array


# full_data = pd.read_csv('./data.csv');
# print("shape of data:", np.shape(full_data))
# catch_df = full_data[full_data['Id'].str.startswith(('c'))]
# sent_df = full_data[full_data['Id'].str.startswith(('s'))]


# utiil
def get_word_count(string):
    tokens = string.split()
    n_tokens = len(tokens)
    return n_tokens


def ReLU(x):
    return abs(x) * (x > 0)


def get_features(inputs):
    out = np.zeros((len(inputs), 200));
    for i in range(len(inputs)):
        try:
            c = w2v[inputs[i]]
        except KeyError:
            c = np.zeros((1, 200));
        out[i] = c;

    return np.transpose(out);


def lookup_indexes(sentences):
    sentence_indexes = [];
    d = enchant.Dict('en_US')
    for word in sentences:
        if d.check(word):
            if word in word_index:
                sentence_indexes.append(word_index[word]);
            else:
                sentence_indexes.append(word_index['unk']);
        else:
            sentence_indexes.append(word_index['unk']);

    return sentence_indexes;


def get_sentences(file):
    #with open(dirpath + file, 'r',encoding="utf-8", errors='replace') as f:
    with open(dirpath + file, 'r') as f:
        data=str(f.read());
        data=data.lower()
        data = data.replace("\"id=", "id=\"");
        data=data.replace("\n","")
        data=data.replace('".*?=.*?"', "",)
        data=data.replace("&","");
        xml = ET.fromstring(str(data))
        name=None;
        rows_list=[];
        catchphrases=[];
        sentences=[];
        for child in xml:
            if child.tag=="catchphrases":
                for catchphrase in child:
                    id=catchphrase.attrib.get("id")
                    #print(catchphrase.text)
                    catchphrases.append({"file_id":file,"Name":name,"Id":id,"text":catchphrase.text})
                    #catchphrases+=tokenizer.tokenize(catchphrase.text)
            if child.tag=="sentences":
                for sentence in child:
                    id = sentence.attrib.get("id")
                    sentences.append(sentence.text)
                    #sentences+=tokenizer.tokenize(sentence.text)
    
    return sentences





# In[ ]:


def get_features_row(sent_df,file_id, Id, text, catch_words):
    text = text.lower();

    word_count = get_word_count(text);
    words = text.split()
    input_filter = np.random.rand(300, 200 * (2 * window_size + 1));
    sentence_vector = np.zeros((300, 1));
    count = 0;
    rows_list = []
    if word_count > 2 * window_size + 1:
        for i in range(window_size, word_count - window_size):
            #sentence = get_sentence_hash(sent_df,Id, file_id)
            inputs = words[i - window_size:i + window_size + 1];
            inputs = lookup_indexes(inputs)
            is_catchword = 1 if words[i] in catch_words else 0

            rows_list.append(
                {"file_id": file_id, "Id": Id, "words": inputs, "is_catchword": is_catchword})
    return rows_list


# In[ ]:


def get_catch_words(catch_df):
    catch_words = []
    catch_phrases = []
    for index, row in catch_df.iterrows():
        catch_phrases.append(row['text']);
        words = word_tokenize(row['text']);
        temp = [word for word in words if not word in stop_words_list]
        catch_words = catch_words + temp;
    return catch_words, catch_phrases;


'''
def get_file_sentences(file_name):
    temp_data=full_data[(full_data.file_id==file_name) & full_data['Id'].str.startswith(('s'))]
    return list(temp_data.text.values)[:-1]
'''


def get_dataframe(sent_df,catch_df):
    
    catch_words, catch_phrases = get_catch_words(catch_df);
    sent_df = sent_df[:-1]
    file_dataframe = [];
    count = 0;
    for index, row in sent_df.iterrows():
        count = count + 1;
        # print(get_features_row_dummy(row['file_id'],row['Id'],row['text']))
        temp = get_features_row(sent_df,row['file_id'], row['Id'], row['text'], catch_words)
        if len(temp) > 0:
            file_dataframe = file_dataframe + temp;

    return pd.DataFrame(file_dataframe), catch_phrases;


print("time-taken for FP: ",time.time()-start)