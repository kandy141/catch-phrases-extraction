
import gensim
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from os import listdir
from os.path import isfile, join
import pandas as pd





def get_data(filename,full_data):
    file_data=full_data[full_data.file_id==filename];
    final_str = "";

    for txt in file_data['text'][:-1]:
        final_str = final_str + txt;
    return final_str;


#This function does all cleaning of data using two objects above
def nlp_clean(data,tokenizer,stopword_set):
   new_data = []
   for d in data:
      new_str = d.lower()
      dlist = tokenizer.tokenize(new_str)
      dlist = list(set(dlist).difference(stopword_set))
      new_data.append(dlist)
   return new_data




class LabeledLineSentence(object):
    def __init__(self, doc_list, labels_list):
        self.labels_list = labels_list
        self.doc_list = doc_list
    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
              yield gensim.models.doc2vec.LabeledSentence(doc,[self.labels_list[idx]])





#
# d2v_model = gensim.models.doc2vec.Doc2Vec.load('doc2vec.model')
# #start testing
# #printing the vector of document at index 1 in docLabels
# docvec = d2v_model.docvecs[1]
# print(docvec)


def main():
    print("calling main method");

    tokenizer = RegexpTokenizer(r'\w+')
    stopword_set = set(stopwords.words('english'))

    docLabels = []
    docLabels = [f for f in listdir("corpus/fulltext") if f.endswith('.xml')]
    # create a list data that stores the content of all text files in order of their names in docLabels

    complete_data = pd.read_csv('./data.csv');

    print("fetched data");
    data = []
    for doc in docLabels:
        data.append(get_data(doc,complete_data));

    data = nlp_clean(data,tokenizer,stopword_set);
    # iterator returned over all documents
    it = LabeledLineSentence(data, docLabels)

    token_count = sum([len(sentence) for sentence in data])

    model = gensim.models.Doc2Vec(size=200, min_count=0, alpha=0.025, min_alpha=0.025)
    model.build_vocab(it)
    # training of model
    for epoch in range(100):
        print('iteration ' + str(epoch + 1))
        model.train(it, total_examples=token_count, epochs=1);
        model.alpha -= 0.002;
        model.min_alpha = model.alpha

    # saving the created model
    model.save('./doc2vec_models/doc2vec.model')
    print("model saved");

main()









