# -*- coding: utf-8 -*-
"""
Created on Sat May 14 22:02:55 2022

@author: user
"""
# basic libraries
import pandas as pd
import numpy as np

# n-gram
from collections import Counter, defaultdict
from typing import List # somehow python 3.9 requires

# preprocessing
from nltk.tokenize.toktok import ToktokTokenizer

# keras lstm model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Adaboost
from sklearn.ensemble import AdaBoostClassifier
# GaussianNB
from sklearn.naive_bayes import GaussianNB
# DecisionTree
from sklearn.tree import DecisionTreeClassifier

# evaluation
from sklearn.metrics import *

from tqdm import tqdm
# Voting Classifier
from sklearn.ensemble import VotingClassifier

import joblib

#%% load data
train = pd.read_csv('C:/D/Study/Intro to AI/Final Project/dataset (with valid)/train.csv')#.head(100)
test = pd.read_csv('C:/D/Study/Intro to AI/Final Project/dataset (with valid)/test.csv')#.head(50)
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
   


#%% score function

def print_score(clf, x, y):
    pred = clf.predict(x)
    print(classification_report(y, pred, zero_division=1))
    print('accuracy:    ', accuracy_score(y, pred))
    print('loss:        ', mean_squared_error(y, pred))
    print('precision:   ', precision_score(y, pred))
    print('recall:      ' , recall_score(y, pred))
    print('f1-score:    ', f1_score(y, pred))
    print('MCC:         ', matthews_corrcoef(y, pred))
    print('\n')



TRAINED_VOTING_MODEL = False

#%% decision tree with news only
'''
    *************************************************
    *   Classification with News Only               *
    *************************************************
'''
print('***** Data Set: News Only *****')

print('Voting Classifier:')
pure_clf_nb = GaussianNB()
pure_clf_DT = DecisionTreeClassifier()
vclf_pure = VotingClassifier(estimators=[('NB_pure', pure_clf_nb), ('DT_pure', pure_clf_DT)], voting='soft', verbose=True)
vclf_pure.fit(x_train, y_train)
print('the training data score of news only is:')
print_score(vclf_pure, x_train, y_train)
print('the testing data score of news only is: ')
print_score(vclf_pure, x_test, y_test)

print('*******************************')

# Save models
if not TRAINED_VOTING_MODEL:
    joblib.dump(vclf_pure, 'ngram_Voting_Pure.model')



#%% decision tree with writing pattern
'''
    *************************************************
    *   Classification with Writing Pattern         *
    *************************************************
'''
print('***** Data Set: /w Writing Pattern *****')

writing_train = train[col_name[2]]
writing_test = test[col_name[2]]
writing_train = np.c_[writing_train, train[col_name[3]]]
writing_test = np.c_[writing_test, test[col_name[3]]]
for i in range(16,26):
    writing_train = np.c_[writing_train, train[col_name[i]]]
    writing_test = np.c_[writing_test, test[col_name[i]]]

x_train_w = np.c_[x_train, writing_train]
x_test_w = np.c_[x_test, writing_test]



print('Voting Classifier:')
clf_w_nb = GaussianNB()
clf_w_DT = DecisionTreeClassifier()
vclf_w = VotingClassifier(estimators=[('NB_WP', clf_w_nb), ('DT_WP', clf_w_DT)], voting='soft', verbose=True)
vclf_w.fit(x_train_w, y_train)
print('the training data score of news + writing is:')
print_score(vclf_w, x_train_w, y_train)
print('the testing data score of news + writing is:')
print_score(vclf_w, x_test_w, y_test)

print('****************************************')

# Save models
if not TRAINED_VOTING_MODEL:
    joblib.dump(vclf_w, 'ngram_Voting_WritingPattern.model')



#%% decition tree with Quantity pattern
'''
    *************************************************
    *   Classification with Quantity Pattern        *
    *************************************************
'''
print('***** Data Set: /w Quantity Pattern *****')

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


print('Voting Classifier:')
clf_q_nb = GaussianNB()
clf_q_DT = DecisionTreeClassifier()
vclf_q = VotingClassifier(estimators=[('NB_QP', clf_q_nb), ('DT_QP', clf_q_DT)], voting='soft', verbose=True)
vclf_q.fit(x_train_q, y_train)
print('the training data score of news + quantity is:')
print_score(vclf_q, x_train_q, y_train)
print('the testing data score of news + quantity is:')
print_score(vclf_q, x_test_q, y_test)

print('*****************************************')

# Save models
if not TRAINED_VOTING_MODEL:
    joblib.dump(vclf_q, 'ngram_Voting_QuantityPattern.model')



#%% decision tree total data
'''
    *************************************************
    *   Classification with Total Data              *
    *************************************************
'''
print('***** Data Set: Total Data *****')

x_train = np.c_[x_train, writing_train]
x_train = np.c_[x_train, quantity_train]
x_test = np.c_[x_test, writing_test]
x_test = np.c_[x_test, quantity_test]


print('Voting Classifier:')
clf_nb = GaussianNB()
clf_DT = DecisionTreeClassifier()
vclf_total = VotingClassifier(estimators=[('NB_total', clf_nb), ('DT_total', clf_DT)], voting='soft', verbose=True)
vclf_total.fit(x_train, y_train)
print('the training data score of news in total is:')
print_score(vclf_total, x_train, y_train)
print('the testing data score of news in total is: ')
print_score(vclf_total, x_test, y_test)

print('********************************')

# Save models
if not TRAINED_VOTING_MODEL:
    joblib.dump(vclf_total, 'ngram_Voting_Total.model')


