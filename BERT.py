'''
    Reference: https://towardsdatascience.com/text-classification-with-bert-in-pytorch-887965e5820f
'''
import numpy as np
import torch
from torch import nn
import numpy as np
from transformers import BertTokenizer
from transformers import BertModel
import pandas as pd

from torch.optim import Adam
from tqdm import tqdm

import pytz
from datetime import datetime
from math import sqrt

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
dataPath = 'dataset/train.csv'

df = pd.read_csv(dataPath, index_col=0)
#new_df = df_train.sort_values(["text_length"], ascending=False)
#print(new_df[["text", "text_length"]].head(2))
np.random.seed(377)
df_train, df_val, df_train2, df_val2 = np.split(df.sample(frac=1, random_state=42), [int(.4*len(df)), int(.5*len(df)), int(.9*len(df))])
#print(len(df_train), len(df_val))


df_test = pd.read_csv("C:\D\Study\Intro to AI\Final Project\exp2_dataset\general.csv", index_col=0)
# If dataset is modified by Excel, use the instruction below instead of the one above.
#df_test = pd.read_csv("C:\D\Study\Intro to AI\Final Project\exp2_dataset\general.csv", index_col=0, encoding='mbcs')
print(df_test)

#labels = {'0':0, '1':1}

class Dataset(torch.utils.data.Dataset):
    def __init__(self, df):

        self.labels = [label for label in df['label']]
        #print('label type: %s' % (type(self.labels[0])))
        #self.texts = [tokenizer(text, 
        #                       padding='max_length', max_length = 512, truncation=True,
        #                        return_tensors="pt") for text in df['text']]

        # experiment 2
        self.texts = [tokenizer(str(text), 
                               padding='max_length', max_length = 512, truncation=True,
                                return_tensors="pt") for text in df['content']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y

class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 2)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer

def train(model, train_data, val_data, learning_rate, epochs):

    train, val = Dataset(train_data), Dataset(val_data)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=1, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=1)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr= learning_rate)

    if use_cuda:

            model = model.cuda()
            criterion = criterion.cuda()

    for epoch_num in range(epochs):

            total_acc_train = 0
            total_loss_train = 0

            for train_input, train_label in tqdm(train_dataloader):

                train_label = train_label.type(torch.LongTensor)
                train_label = train_label.to(device)
                mask = train_input['attention_mask'].to(device)
                input_id = train_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)
                
                batch_loss = criterion(output, train_label)
                total_loss_train += batch_loss.item()
                
                acc = (output.argmax(dim=1) == train_label).sum().item()
                total_acc_train += acc

                model.zero_grad()
                batch_loss.backward()
                optimizer.step()
            
            total_acc_val = 0
            total_loss_val = 0

            with torch.no_grad():

                for val_input, val_label in val_dataloader:

                    val_label = val_label.type(torch.LongTensor)
                    val_label = val_label.to(device)
                    mask = val_input['attention_mask'].to(device)
                    input_id = val_input['input_ids'].squeeze(1).to(device)

                    output = model(input_id, mask)

                    batch_loss = criterion(output, val_label)
                    total_loss_val += batch_loss.item()
                    
                    acc = (output.argmax(dim=1) == val_label).sum().item()
                    total_acc_val += acc
            
            print(
                f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
                | Train Accuracy: {total_acc_train / len(train_data): .3f} \
                | Val Loss: {total_loss_val / len(val_data): .3f} \
                | Val Accuracy: {total_acc_val / len(val_data): .3f}')

def evaluate(model, test_data):

    test = Dataset(test_data)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()

    if use_cuda:

        model = model.cuda()
        criterion = criterion.cuda()

    # True Positive, False Positive, True Negative, False Negative
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    total_loss_val = 0
    with torch.no_grad():

        for test_input, test_label in tqdm(test_dataloader):

            test_label = test_label.type(torch.LongTensor)
            test_label = test_label.to(device)
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)

            tp += ((output.argmax(dim=1) == test_label) & (output.argmax(dim=1) == torch.ones(2, device=device, dtype=torch.int32))).sum().item()
            fp += ((output.argmax(dim=1) != test_label) & (output.argmax(dim=1) == torch.ones(2, device=device, dtype=torch.int32))).sum().item()
            tn += ((output.argmax(dim=1) == test_label) & (output.argmax(dim=1) == torch.zeros(2, device=device, dtype=torch.int32))).sum().item()
            fn += ((output.argmax(dim=1) != test_label) & (output.argmax(dim=1) == torch.zeros(2, device=device, dtype=torch.int32))).sum().item()

            batch_loss = criterion(output, test_label)
            total_loss_val += batch_loss.item()
            #print('output.argmax(dim=1) : ', output.argmax(dim=1))
            #print('test_label           : ', test_label)
    total_acc_test = tp + tn
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * precision * recall / (precision + recall)
    mcc = (tp * tn - fp * fn) / sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    print('******** Test ********')
    print('tp = %d | fp = %d | tn = %d | fn = %d' % (tp, fp, tn , fn))
    print(f'Accuracy    : {total_acc_test / len(test_data): .3f}')
    print(f'Loss        : {total_loss_val / len(test_data): .3f}')
    print(f'Precision   : {precision: .3f}')
    print(f'Recall      : {recall: .3f}')
    print(f'F1 Score    : {f1_score: .3f}')
    print(f'MCC         : {mcc: .3f}')
    print('**********************')

EPOCHS = 5
model = BertClassifier()
LR = 1e-6

USING_SAVED_MODEL = False
savedModelName = 'Bert_model_2022-06-04 12-28-44.551944+08-00'

if USING_SAVED_MODEL:
    model = torch.load(savedModelName)
else:
    train(model, df_train, df_val, LR, EPOCHS)

    savePath = 'Bert_model_' + str(datetime.now(pytz.timezone('Asia/Taipei'))) + '.pt'
    savePath = savePath.replace(':', '-')
    torch.save(model, savePath)

evaluate(model, df_test)