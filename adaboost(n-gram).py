# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 20:29:50 2022

@author: diane
"""
import pandas as pd
import numpy as np

# n-gram
from collections import Counter, defaultdict
from typing import List # somehow python 3.9 requires

from nltk.tokenize.toktok import ToktokTokenizer
# from sklearn.metrics import precision_recall_fscore_support


# keras lstm model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# DT
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import *


train = pd.read_csv('valid_dataset/train.csv')#.head(100)
test = pd.read_csv('valid_dataset/test.csv')#.head(50)
#valid = pd.read_csv('valid_dataset/valid.csv').head(50)
col_name = [i for i in train[0:1]][1:33]

class Ngram:
    def __init__(self, config, n=2):
        self.tokenizer = ToktokTokenizer()
        self.n = n
        self.model = None
        self.config = config
        self.counts = None # the counts for corpus


    def tokenize(self, sentence):
        return self.tokenizer.tokenize(sentence)

    #turning words that appeared less than 10/1 times as [UNK], num=10 or 2
    def turnToUnk(self, corpus, num):
        corpus_list = [j for i in corpus for j in i]
        counts = Counter(corpus_list)
        pdCounts = pd.DataFrame(counts.values(),index = list(counts))
        lessWords = list(pdCounts[pdCounts[0]<num].index)
        pd_corpus = pd.DataFrame(corpus_list)
        pd_corpus[pd_corpus[0].isin(lessWords)] = '[UNK]'
        pd_index = (pd_corpus[pd_corpus[0].str.contains('[CLS]',na=False)])
        length = len(pd_index.index)
        corpus = []
        for i,f in enumerate(pd_index.index):
            if(i == length-1):
                break
            ff = pd_index.index[i+1]
            corpus.append(list(pd_corpus[f:ff][0]))
        return corpus

    def get_ngram(self, corpus_tokenize: List[List[str]]):
        # begin your code (Part 1)
        
        bigram_tokens = [(x, i[j + 1]) for i in corpus_tokenize 
            for j, x in enumerate(i) if j < len(i) - 1]
        bigrams_freq = Counter(bigram_tokens)

        prob = defaultdict(lambda: defaultdict(lambda: 0))
        counts = defaultdict(lambda: defaultdict(lambda :0))
        for i in bigrams_freq:
            counts[i[0]][i[1]] = bigrams_freq[i]
        for i1 in counts:
            total_count = float(sum(counts[i1].values()))
            for i2 in counts[i1]:
                prob[i1][i2] = counts[i1][i2]/ total_count

        return prob, counts
    
    def train(self, df):
        '''
        Train n-gram model.
        '''
        corpus = [['[CLS]'] + self.tokenize(document) for document in df['text']] + [['[CLS]'] + self.tokenize(document) for document in df['title']] 
        # [CLS] represents start of sequence
        
        self.model, self.counts = self.get_ngram(corpus)
                
biGram = Ngram(2)
biGram.train(train)

feature_num = 800
feature_counter = []
for i in biGram.counts:
    for r in biGram.counts[i]:
        feature_counter.append((i+" "+r,biGram.counts[i][r]))

df = pd.DataFrame(feature_counter)
df = df.sort_values(by=[1],ascending=False)
feature = list(df.iloc[0:feature_num,0])


x_train = []
x_test = []
x_valid = []


for i,f in enumerate(train['text']):
    corpus = [['[CLS]'] + biGram.tokenize(f)] + [['[CLS]'] + biGram.tokenize(train['title'][i])]

    bigram_tokens = [(x+" "+i[j + 1]) for i in corpus 
                for j, x in enumerate(i) if j < len(i) - 1]
    count = [0]* feature_num
    for f in bigram_tokens:
        if(f in feature):
            count[feature.index(f)]+=1
    x_train.append(count)



for i,f in enumerate(test['text']):
    corpus = [['[CLS]'] + biGram.tokenize(f)] + [['[CLS]'] + biGram.tokenize(test['title'][i])]
    
    bigram_tokens = [(x+" "+i[j + 1]) for i in corpus 
                for j, x in enumerate(i) if j < len(i) - 1]
    count = [0]*feature_num
    for f in bigram_tokens:
        if(f in feature):
            count[feature.index(f)]+=1
    x_test.append(count)

    
x_train = np.array(x_train)
x_test = np.array(x_test)

y_train = train['label']
y_test = test['label']

def print_score(clf, x, y):
    pred = clf.predict(x)
    print(classification_report(y, pred, zero_division=1))
    print('accuracy', accuracy_score(y, pred))
    print('loss: ', mean_squared_error(y, pred))
    print('precision:', precision_score(y, pred))
    print('recall:' , recall_score(y, pred))
    print('f1-score:', f1_score(y, pred))
    print('MCC: ', matthews_corrcoef(y, pred))
    print('\n')
    

#%% decision tree with news only
pure_clf = AdaBoostClassifier()
pure_clf.fit(x_train, y_train)
print('the training data score of news only is:')
print_score(pure_clf, x_train, y_train)
print('the testing data score of news only is: ')
print_score(pure_clf, x_test, y_test)


#%% decision tree with writing pattern
writing_train = train[col_name[2]]
writing_test = test[col_name[2]]
writing_train = np.c_[writing_train, train[col_name[3]]]
writing_test = np.c_[writing_test, test[col_name[3]]]
for i in range(16,26):
    writing_train = np.c_[writing_train, train[col_name[i]]]
    writing_test = np.c_[writing_test, test[col_name[i]]]

x_train_w = np.c_[x_train, writing_train]
x_test_w = np.c_[x_test, writing_test]


clf_w = AdaBoostClassifier()
clf_w.fit(x_train_w, y_train)
print('the training data score of news + writing is:')
print_score(clf_w, x_train_w, y_train)
print('the testing data score of news + writing is:')
print_score(clf_w, x_test_w, y_test)

#%% decition tree with Quantity pattern
quantity_train = train[col_name[4]]
quantity_test = test[col_name[4]]
for i in range(5,16):
    quantity_train = np.c_[quantity_train, train[col_name[i]]]
    quantity_test = np.c_[quantity_test, test[col_name[i]]]
    
for i in range(26, 32):
    quantity_train = np.c_[quantity_train, train[col_name[i]]]
    quantity_test = np.c_[quantity_test, test[col_name[i]]]

x_train_q = np.c_[x_train, quantity_train]
x_test_q = np.c_[x_test, quantity_test]

clf_q = AdaBoostClassifier()
clf_q.fit(x_train_q, y_train)
print('the training data score of news + quantity is:')
print_score(clf_q, x_train_q, y_train)
print('the testing data score of news + quantity is:')
print_score(clf_q, x_test_q, y_test)

#%% decision tree total data

x_train = np.c_[x_train, writing_train]
x_train = np.c_[x_train, quantity_train]
x_test = np.c_[x_test, writing_test]
x_test = np.c_[x_test, quantity_test]

clf = AdaBoostClassifier()
clf.fit(x_train, y_train)
print('the training data score of news in total is:')
print_score(clf, x_train, y_train)
print('the testing data score of news in total is: ')
print_score(clf, x_test, y_test)