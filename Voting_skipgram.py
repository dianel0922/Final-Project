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
writing_patterns = ['title_special_chars', 'text_special_chars',
       'title_dets', 'text_dets', 'title_capitals', 'text_capitals',
       'title_short_sents', 'text_short_sents', 'title_long_sents',
       'text_long_sents']

Quantity = ['title_words', 'text_words',
       'title_sents', 'text_sents', 'title_words_per_sent',
       'text_words_per_sent', 'title_verbs', 'text_verbs', 'title_adjs',
       'text_adjs', 'title_advs', 'text_advs', 'title_rate_adjs_and_advs',
       'text_rate_adjs_and_advs','title_fog', 'text_fog', 'title_smog', 'text_smog',
       'title_ari', 'text_ari']

Both = ['title_words', 'text_words',
       'title_sents', 'text_sents', 'title_words_per_sent',
       'text_words_per_sent', 'title_verbs', 'text_verbs', 'title_adjs',
       'text_adjs', 'title_advs', 'text_advs', 'title_rate_adjs_and_advs',
       'text_rate_adjs_and_advs', 'title_special_chars', 'text_special_chars',
       'title_dets', 'text_dets', 'title_capitals', 'text_capitals',
       'title_short_sents', 'text_short_sents', 'title_long_sents',
       'text_long_sents', 'title_fog', 'text_fog', 'title_smog', 'text_smog',
       'title_ari', 'text_ari']


config = Both

train = pd.read_csv('C:/D/Study/Intro to AI/Final Project/dataset (with valid)/train.csv')#.head(100)
test = pd.read_csv('C:/D/Study/Intro to AI/Final Project/dataset (with valid)/test.csv')#.head(50)
#valid = pd.read_csv('valid_dataset/valid.csv').head(50)

col_name = [i for i in train[0:1]][1:33]
#print(col_name)
   
#%% train skip-gram model
'''
TRAINED_WORD2VEC_MODEL = True

tokenizer = ToktokTokenizer()
corpus = [tokenizer.tokenize(document) for document in train['title'] ] + [ tokenizer.tokenize(document1) for document1 in train['text']]
model = gensim.models.word2vec.Word2Vec(corpus, min_count = 3, window = 5, sg = 1) # sg -> skip gram = true
if TRAINED_WORD2VEC_MODEL:
    model = gensim.models.word2vec.Word2Vec.load('Word2Vec.wvmodel')
else:
    print('----- Word2Vec Start Training ------')
    model.train(tqdm(corpus), total_examples=len(corpus), epochs = 5)
    print('----- Word2Vec Finish Training -----')

    # Save Word2Vec model file
    model.save('Word2Vec.wvmodel')

vector = model.wv

tokenizer = Tokenizer(num_words = 6000)
tokenizer.fit_on_texts(train['title'])
sequences_title = tokenizer.texts_to_sequences(train['title'])

tokenizer = Tokenizer(num_words = 6000)
tokenizer.fit_on_texts(train['text'])
sequences_text = tokenizer.texts_to_sequences(train['text'])

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data_title = pad_sequences(sequences_title, maxlen= 10000)
data_text = pad_sequences(sequences_text, maxlen = 10000)
data = np.c_[data_title, data_text]

labels = train['label']
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

x_train = data
y_train = labels

#print(f'x_train: {x_train}, size = {len(x_train)}')
#print(f'y_train: {y_train}, size = {len(y_train)}')

#%% fit test data

sequences_title = tokenizer.texts_to_sequences(test['title'])       
sequences_test = tokenizer.texts_to_sequences(test['text'])

data_title = pad_sequences(sequences_title, maxlen= 10000 )
data_text = pad_sequences(sequences_test, maxlen= 10000)
y_test = test['label']
x_test = np.c_[data_title, data_text]
'''
#print(x_test)
#print(y_test)

#%% train skip-gram model
tokenizer = ToktokTokenizer()
corpus = [tokenizer.tokenize(document) for document in train['title']]
model = gensim.models.Word2Vec(corpus, min_count = 5, window = 5, sg = 1) # sg -> skip gram = true
model.train(tqdm(corpus), total_examples=len(corpus), epochs = 5)

vector = model.wv
words = model.wv.index_to_key
wvs = model.wv[words]
dic = {}
for i,f in enumerate(words):
    dic[f] = wvs[i]
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
x_train_pure = np.array(x_train)
x_test_pure = np.array(x_test)
# 後面有feature的
x_train_w = np.concatenate([x_train_pure, train[writing_patterns]], axis=1)
x_test_w = np.concatenate([x_test_pure, test[writing_patterns]], axis=1)
x_train_q = np.concatenate([x_train_pure, train[Quantity]], axis=1)
x_test_q = np.concatenate([x_test_pure, test[Quantity]], axis=1)
x_train_all = np.concatenate([x_train_pure, train[Both]], axis=1)
x_test_all = np.concatenate([x_test_pure, test[Both]], axis=1)
y_train = train['label']
y_test = test['label']
#%% meaning unknown code 
'''
EMBEDDING_DIM = 100
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    try:
      embedding_vector = vector[word]
    except:
      embedding_vector = None
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
    
#embedding_matrix = np.transpose(embedding_matrix)

#print(type(embedding_matrix))
#print(embedding_matrix.shape)
'''
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



'''
TRAINED_ADABOOST_MODEL = True
TRAINED_DT_MODEL = True
'''
TRAINED_VOTING_MODEL = False

#%% decision tree with news only
'''
    *************************************************
    *   Classification with News Only               *
    *************************************************
'''
print('***** Data Set: News Only *****')

'''
print('Adaboost:')
pure_clf_ada = AdaBoostClassifier()
if TRAINED_ADABOOST_MODEL:
    pure_clf_ada = joblib.load('AdaBoost_Pure.model')
else:
    pure_clf_ada.fit(x_train, y_train)
print('the training data score of news only is:')
print_score(pure_clf_ada, x_train, y_train)
print('the testing data score of news only is: ')
print_score(pure_clf_ada, x_test, y_test)

print('Decision Tree:')
pure_clf_DT = DecisionTreeClassifier()
if TRAINED_DT_MODEL:
    pure_clf_DT = joblib.load('DecisionTree_Pure.model')
else:
    pure_clf_DT.fit(x_train, y_train)
print('the training data score of news only is:')
print_score(pure_clf_DT, x_train, y_train)
print('the testing data score of news only is: ')
print_score(pure_clf_DT, x_test, y_test)
'''

print('Voting Classifier:')
pure_clf_nb = GaussianNB()
pure_clf_DT = DecisionTreeClassifier()
vclf_pure = VotingClassifier(estimators=[('NB_pure', pure_clf_nb), ('DT_pure', pure_clf_DT)], voting='soft', verbose=True)
vclf_pure.fit(x_train_pure, y_train)
print('the training data score of news only is:')
print_score(vclf_pure, x_train_pure, y_train)
print('the testing data score of news only is: ')
print_score(vclf_pure, x_test_pure, y_test)

print('*******************************')

# Save models
'''
if not TRAINED_ADABOOST_MODEL:
    joblib.dump(pure_clf_ada, 'AdaBoost_Pure.model')
if not TRAINED_DT_MODEL:
    joblib.dump(pure_clf_DT, 'DecisionTree_Pure.model')
'''
if not TRAINED_VOTING_MODEL:
    joblib.dump(vclf_pure, 'Voting_Pure.model')



#%% decision tree with writing pattern
'''
    *************************************************
    *   Classification with Writing Pattern         *
    *************************************************
'''
print('***** Data Set: /w Writing Pattern *****')

'''
writing_train = train[col_name[2]]
writing_test = test[col_name[2]]
writing_train = np.c_[writing_train, train[col_name[3]]]
writing_test = np.c_[writing_test, test[col_name[3]]]
for i in range(16,26):
    writing_train = np.c_[writing_train, train[col_name[i]]]
    writing_test = np.c_[writing_test, test[col_name[i]]]

x_train_w = np.c_[x_train, writing_train]
x_test_w = np.c_[x_test, writing_test]
'''


'''
print('Adaboost:')
clf_w_ada = AdaBoostClassifier()
if TRAINED_ADABOOST_MODEL:
    clf_w_ada = joblib.load('AdaBoost_WritingPattern.model')
else:
    clf_w_ada.fit(x_train_w, y_train)
print('the training data score of news + writing is:')
print_score(clf_w_ada, x_train_w, y_train)
print('the testing data score of news + writing is:')
print_score(clf_w_ada, x_test_w, y_test)

print('Decision Tree:')
clf_w_DT = DecisionTreeClassifier()
if TRAINED_DT_MODEL:
    clf_w_DT = joblib.load('DecisionTree_WritingPattern.model')
else:
    clf_w_DT.fit(x_train_w, y_train)
print('the training data score of news + writing is:')
print_score(clf_w_DT, x_train_w, y_train)
print('the testing data score of news + writing is:')
print_score(clf_w_DT, x_test_w, y_test)
'''

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
'''
if not TRAINED_ADABOOST_MODEL:
    joblib.dump(clf_w_ada, 'AdaBoost_WritingPattern.model')
if not TRAINED_DT_MODEL:
    joblib.dump(clf_w_DT, 'DecisionTree_WritingPattern.model')
'''
if not TRAINED_VOTING_MODEL:
    joblib.dump(vclf_w, 'Voting_WritingPattern.model')



#%% decition tree with Quantity pattern
'''
    *************************************************
    *   Classification with Quantity Pattern        *
    *************************************************
'''
print('***** Data Set: /w Quantity Pattern *****')

'''
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
'''

'''
print('Adaboost:')
clf_q_ada = AdaBoostClassifier()
if TRAINED_ADABOOST_MODEL:
    clf_q_ada = joblib.load('AdaBoost_QuantityPattern.model')
else:
    clf_q_ada.fit(x_train_q, y_train)
print('the training data score of news + quantity is:')
print_score(clf_q_ada, x_train_q, y_train)
print('the testing data score of news + quantity is:')
print_score(clf_q_ada, x_test_q, y_test)

print('Decision Tree:')
clf_q_DT = DecisionTreeClassifier()
if TRAINED_DT_MODEL:
    clf_q_DT = joblib.load('DecisionTree_QuantityPattern.model')
else:
    clf_q_DT.fit(x_train_q, y_train)
print('the training data score of news + quantity is:')
print_score(clf_q_DT, x_train_q, y_train)
print('the testing data score of news + quantity is:')
print_score(clf_q_DT, x_test_q, y_test)
'''

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
'''
if not TRAINED_ADABOOST_MODEL:
    joblib.dump(clf_q_ada, 'AdaBoost_QuantityPattern.model')
if not TRAINED_DT_MODEL:
    joblib.dump(clf_q_DT, 'DecisionTree_QuantityPattern.model')
'''
if not TRAINED_VOTING_MODEL:
    joblib.dump(vclf_q, 'Voting_QuantityPattern.model')



#%% decision tree total data
'''
    *************************************************
    *   Classification with Total Data              *
    *************************************************
'''
print('***** Data Set: Total Data *****')
'''
x_train = np.c_[x_train, writing_train]
x_train = np.c_[x_train, quantity_train]
x_test = np.c_[x_test, writing_test]
x_test = np.c_[x_test, quantity_test]
'''
'''
print('Adaboost:')
clf_ada = AdaBoostClassifier()
if TRAINED_ADABOOST_MODEL:
    clf_ada = joblib.load('AdaBoost_Total.model')
else:
    clf_ada.fit(x_train, y_train)
print('the training data score of news in total is:')
print_score(clf_ada, x_train, y_train)
print('the testing data score of news in total is: ')
print_score(clf_ada, x_test, y_test)

print('Decision Tree:')
clf_DT = DecisionTreeClassifier()
if TRAINED_DT_MODEL:
    clf_DT = joblib.load('DecisionTree_Total.model')
else:
    clf_DT.fit(x_train, y_train)
print('the training data score of news in total is:')
print_score(clf_DT, x_train, y_train)
print('the testing data score of news in total is: ')
print_score(clf_DT, x_test, y_test)
'''

print('Voting Classifier:')
clf_nb = GaussianNB()
clf_DT = DecisionTreeClassifier()
vclf_total = VotingClassifier(estimators=[('NB_total', clf_nb), ('DT_total', clf_DT)], voting='soft', verbose=True)
vclf_total.fit(x_train_all, y_train)
print('the training data score of news in total is:')
print_score(vclf_total, x_train_all, y_train)
print('the testing data score of news in total is: ')
print_score(vclf_total, x_test_all, y_test)

print('********************************')

# Save models
'''
if not TRAINED_ADABOOST_MODEL:
    joblib.dump(clf_ada, 'AdaBoost_Total.model')
if not TRAINED_DT_MODEL:
    joblib.dump(clf_DT, 'DecisionTree_Total.model')
'''
if not TRAINED_VOTING_MODEL:
    joblib.dump(vclf_total, 'Voting_Total.model')


