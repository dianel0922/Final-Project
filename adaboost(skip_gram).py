# -*- coding: utf-8 -*-
"""
Created on Sat May 14 22:02:55 2022

@author: user
"""
# basic libraries
import pandas as pd
import numpy as np

# preprocessing

from nltk.tokenize.toktok import ToktokTokenizer
# generate word embedding
import gensim

# keras lstm model

from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Activation, Dense, Dropout, Input, Embedding, SpatialDropout1D
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# Adaboost
from sklearn.ensemble import AdaBoostClassifier

# evaluation
from sklearn.metrics import *

#%% load data
train = pd.read_csv('valid_dataset/train.csv')#.head(100)
test = pd.read_csv('valid_dataset/test.csv')#.head(50)
#valid = pd.read_csv('valid_dataset/valid.csv').head(50)
general = pd.read_csv('valid_dataset/general.csv', encoding='unicode_escape')#.head(50)

col_name = [i for i in train[0:1]][1:33]
#print(col_name)
   
#%% train skip-gram model
tokenizer = ToktokTokenizer()
corpus = [tokenizer.tokenize(document) for document in train['title'] ] + [ tokenizer.tokenize(document1) for document1 in train['text']]
model = gensim.models.Word2Vec(corpus, min_count = 3, window = 5, sg = 1) # sg -> skip gram = true
model.train(corpus, total_examples=len(corpus), epochs = 5)

vector = model.wv

words = model.wv.index_to_key
wvs = model.wv[words]
dic = {}
for i,f in enumerate(words):
    dic[f] = wvs[i]

#%% fit train data    
x_train = []

for i,f in enumerate(train['text']):
    one = np.zeros((500, 1))
    corpus = [['[CLS]'] + tokenizer.tokenize(f)] + [['[CLS]'] + tokenizer.tokenize(train['title'][i])]
        
    if len(corpus[0]) < 500:
        for i in range(500-len(corpus[0])):
            corpus[0].append('0')

    one=[]
    for ii,ff in enumerate(corpus[0][0:500]):
        try:
            one.append(dic[ff].mean())
        except:
            one.append(0)
            
    # print(one)
    x_train.append(one)

#%% fit test data
x_test = []
for i,f in enumerate(test['text']):
    corpus = [['[CLS]'] + tokenizer.tokenize(f)] + [['[CLS]'] + tokenizer.tokenize(test['title'][i])]
        
    if len(corpus[0]) < 500:
        for i in range(500-len(corpus[0])):
            corpus[0].append('0')

    one=[]
    for ii,ff in enumerate(corpus[0][0:500]):
        try:
            one.append(dic[ff].mean())
        except:
            one.append(0)
            
    # print(one)
    x_test.append(one)

#%% some preprocess
x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = train['label']
y_test = test['label']

#%% score function

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

#%% fit general data
x_general = []
for i,f in enumerate(general['content']):
    corpus = [['[CLS]'] + tokenizer.tokenize(f)] + [['[CLS]'] + tokenizer.tokenize(general['title'][i])]
        
    if len(corpus[0]) < 500:
        for i in range(500-len(corpus[0])):
            corpus[0].append('0')

    one=[]
    for ii,ff in enumerate(corpus[0][0:500]):
        try:
            one.append(dic[ff].mean())
        except:
            one.append(0)
            
    # print(one)
    x_general.append(one)

#%% some preprocess
x_general = np.array(x_general)
y_general = general['label']

#%%
x_general_w = np.c_[x_general, np.zeros((len(x_general), 12))]
x_general_q = np.c_[x_general, np.zeros((len(x_general), 18))]
x_general_t = np.c_[x_general, np.zeros((len(x_general), 30))]

print('the general data score of news only is: ')
print_score(pure_clf, x_general, y_general)
print('the general data score of news + writing is: ')
print_score(clf_w, x_general_w, y_general)
print('the general data score of news + quality is: ')
print_score(clf_q, x_general_q, y_general)
print('the general data score of news in total is: ')
print_score(clf  , x_general_t, y_general)