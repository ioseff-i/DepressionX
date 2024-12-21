import torch
import torch.nn as nn
import numpy as np
# Assuming you have your text data in tensors:
from torch.utils.data import DataLoader
import pandas as pd
from scipy import stats
from tqdm import tqdm, trange
import pickle
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer
from sentence_transformers import SentenceTransformer
from torch_geometric.data import Data, DataLoader
import fasttext
import fasttext.util
import torch_geometric
from torch_geometric.nn import GCNConv, global_mean_pool, SAGEConv, TransformerConv,GINConv, global_mean_pool
import torch.optim as optim
from torch_geometric.nn import GCNConv,SAGEConv, GATConv,TransformerConv, GPSConv
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from sklearn.model_selection import StratifiedKFold
from operator import itemgetter
from torch_geometric.data import Data




import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import torch
from transformers import BertModel, BertTokenizer
from torch_geometric.data import Data
from torch import nn
import torch.nn.functional as F
import torch_geometric
import torch.nn.functional as F  # Add this line
from tqdm import tqdm
from torch_geometric.nn import GCNConv, global_mean_pool, SAGEConv, TransformerConv
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv,SAGEConv, GATConv
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from sklearn.model_selection import StratifiedKFold
from operator import itemgetter
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
from torch_geometric.nn import GINConv, global_mean_pool
import math
from itertools import combinations
from collections import OrderedDict
import pandas as pd
import numpy as np
import re
import string
import pickle
import optuna
import networkx as nx
import argparse
import copy
import json
import time
from sentence_transformers import SentenceTransformer
from datetime import datetime
import matplotlib.pyplot as plt
import os
import seaborn as sn
from tqdm import tqdm, trange
from scipy.special import softmax
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.model_selection import StratifiedKFold
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import  LabelEncoder
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn import metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import Adam
# from torch.utils.data import DataLoader, Dataset
from collections import defaultdict
from gensim.models import Word2Vec
from transformers import AutoTokenizer, AutoModel
from transformers import (AdamW, get_cosine_schedule_with_warmup,
                          get_cosine_with_hard_restarts_schedule_with_warmup)
from torch.nn import Linear, Dropout, ReLU, Tanh, Sigmoid, LogSoftmax, NLLLoss
import torch_geometric as torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv,TransformerConv,SuperGATConv,SAGEConv, GPSConv,GINConv
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer
import ast
import warnings
from transformers import AutoModel

import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer


def graded_eval_metrics(y_true,y_pred):
    false_positive = 0
    false_negative = 0
    true_positive = 0
    diff = 0
    
    # Circle through y_true for comparisons to y_pred
    for i in range(0,len(y_true)):
        
        # Ordinal Error Stats
        if abs(y_true[i]-y_pred[i]) > 1:
            diff += 1
        
        # TP, FP & FN Stats
        if y_pred[i] > y_true[i]:
            false_positive += 1
        elif y_pred[i] < y_true[i]:
            false_negative += 1
        elif y_pred[i] == y_true[i]:
            true_positive += 1
        else:
            pass
        
    # Divided each Metric by the Size of the Testing Set
    # If I didn't divide TP as well as FP & FN, I had odd results
    false_positive = false_positive/len(y_true)
    false_negative = false_negative/len(y_true)
    true_positive = true_positive/len(y_true)
    
    ordinal_error = diff/len(y_true)
        
    # Metric Calculations
    # Added in Except as sometimes during Training had ZeroDivison Error
    precision = true_positive/(true_positive+false_positive + 0.001)
    recall = true_positive/(true_positive+false_negative + 0.001)
    try:
        f1_score = 2 * ((precision*recall)/(precision+recall + 0.001))
    except ZeroDivisionError:
        f1_score = 0
    
    # eval_df = pd.DataFrame({'Metric':['Precision','Recall','F1 Score','Ordinal Error'],
    #                        'Value':[precision,recall,f1_score,ordinal_error]})
    
    # return eval_df, precision, recall, f1_score, ordinal_error
    return f1_score


def clean_text(text):
    # Lowercase conversion
    text = text.lower()

    text = re.sub(r'[.!?;]', '.', text)

    # Remove other punctuation 
    # text = re.sub(r'[^\w\s]', '', text)
    text = text.replace(',','') 
    text = text.replace('-','')
    text = text.replace('_','')
    text = text.replace('/','')
    text = text.replace('#','')

    punc = '''!()-[]{};:'"\,<>/?@#$%^&*_~'''
    for i in punc:
        text = text.replace(i,'')


    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Tokenize text
    tokens = word_tokenize(text)
    
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    text = ' '.join(tokens)
    return text

warnings.filterwarnings('ignore')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()


sent_emb_model = SentenceTransformer('all-distilroberta-v1',device=device)


ft = fasttext.load_model('wiki.en.bin')


def get_ft_embeddings(word):
    return torch.tensor(ft[word.lower()]).unsqueeze(0).unsqueeze(0).to(device)





def prepare_document(document, max_sent_len = 32, max_word_len = 64):
    document = clean_text(document.lower())

    if len(document.split('.')) < max_sent_len:
        document = document + ".none"*(max_sent_len - len(document.split('.')))
    elif len(document.split('.')) > max_sent_len:
        document = '.'.join(document.split('.')[:max_sent_len])

    docs = []
    for i in document.split('.'):
        # if len(i) ==0:
        #     continue
        if len(i.split()) <= max_word_len:
            doc = i = i + ' none'*(max_word_len-len(i.split()))
            
        elif len(i.split()) > max_word_len:
            doc = " ".join(i.split()[:max_word_len])

        
        if len(doc) !=0:
            docs.append(doc)
        
    
    return ".".join(docs)



def get_post_embeddings(doc):
    return torch.tensor(sent_emb_model.encode(doc))

def get_sent_embeddings(doc):
    return torch.tensor(sent_emb_model.encode(doc.split('.')))


def get_word_embeddings(doc):
    sent_list = []
    for j in doc.split('.'):
        sent_list.append(torch.cat([get_ft_embeddings(k) for k in j.split()],dim=1))
    return torch.cat(sent_list,dim=0)


def kg_graph_to_data(graph):
    node_features = [graph.nodes[node]['features'] for node in graph.nodes]
    x = torch.tensor(node_features, dtype=torch.float32)
    
    edge_index = torch.tensor(list(graph.edges), dtype=torch.int64).t().contiguous()
    
    return Data(x=x, edge_index=edge_index)
    

df_path = 'data_path/data.csv''
kg_path ='data_path/data_kg_graphs.pickle'



df = pd.read_csv(df_path)



label_map = {'minimum':0,
                 'mild':1,
                 'moderate':2,
                 'severe':3}

df['label'] = df['label'].replace(label_map)


df.index = range(len(df))



word_lens = []
for x in df.text:
    for y in x.split('.'):
        word_lens.append(len(y.split()))

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, vectors,labels, texts):
        self.vectors = vectors
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.vectors[idx], self.labels[idx], self.texts[idx]


class ContrastiveLoss(nn.Module):
    def __init__(self, temp=0.5):
        super(ContrastiveLoss, self).__init__()
        self.temp = temp

    def forward(self, batch, preds, labels):
        ce_loss = F.cross_entropy(preds, labels)

        # Calculate cosine similarity
        batch_norm = F.normalize(batch, p=2, dim=1)
        similarity_matrix = torch.matmul(batch_norm, batch_norm.T)

        # Exclude diagonal values and apply temperature
        similarity_matrix /= self.temp

        # Compute log probabilities with softmax
        log_probabilities = F.log_softmax(similarity_matrix, dim=1)

        # Compute contrastive loss
        masked_log_prob = log_probabilities[~torch.eye(log_probabilities.shape[0], dtype=bool)].reshape(log_probabilities.shape[0], -1)
        contrastive_loss = -torch.mean(torch.sum(masked_log_prob, dim=1))

        total_loss = ce_loss + contrastive_loss
        return total_loss
class KBNet(nn.Module):
    def __init__(self, num_features, hidden_size, gat_heads, dropout):
        super(KBNet, self).__init__()
        self.heads = gat_heads
        self.dropout = dropout

        self.conv1 = GINConv(nn.Sequential(nn.Linear(num_features, hidden_size), nn.Sequential()), hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)


        self.conv2 = GINConv(nn.Sequential(nn.Linear(hidden_size, hidden_size ), nn.Sequential()), hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        
        self.conv3 = GATConv(in_channels = hidden_size, 
                             out_channels = int(hidden_size / 2),
                             heads = self.heads,
                             concat = False)
        self.bn3 = nn.BatchNorm1d(int(hidden_size / 2))

        self.dropout = nn.Dropout(dropout)

    def forward(self, data):

        x, edge_index = data.x,data.edge_index

  

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.dropout(x)

        return x


from sklearn.metrics.pairwise import cosine_similarity


import torch.nn.functional as F

class DepressionX(nn.Module):
    def __init__(self, embedding_dim=300, hidden_size=128, heads=8, gat_heads=8, dropout=0.25, num_class=4):
        super().__init__()
        self.embed_dim = embedding_dim
        self.hidden_size = hidden_size
        self.heads = heads
        self.gat_heads = gat_heads
        self.num_class = num_class

        self.word_mh = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.heads, batch_first=True)
        self.word_linear = nn.Linear(self.embed_dim, self.hidden_size)

        self.sent_mh = nn.MultiheadAttention(embed_dim=768, num_heads=self.heads, batch_first=True)
        self.sent_linear = nn.Linear(768, self.hidden_size)

        self.post_linear = nn.Linear(768, self.hidden_size)
        self.final_text_linear = nn.Linear(self.hidden_size * 2, self.hidden_size)

        self.dropout = nn.Dropout(p=dropout)
        self.activation = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(int(self.hidden_size * 2))  # Batch normalization

        self.kb_net = KBNet(num_features=768, hidden_size=2 * self.hidden_size, gat_heads=self.gat_heads,
                            dropout=dropout).to(device)
        self.kb_bn = nn.BatchNorm1d(int(self.hidden_size))  # Batch normalization

        self.final_linear = nn.Linear(2 * self.hidden_size, self.num_class)

    def forward(self, data, graph):
        word_embeddings = data[0]
        sent_embeddings = data[1]
        post_embeddings = data[2]


        word_enc_vects = []
        word_att_vects = []

        for x in word_embeddings:
            result = self.word_mh(x, x, x)
            word_enc_vects.append(result[0].unsqueeze(0))
            word_att_vects.append(result[1].mean(dim=1).unsqueeze(0))

        word_encoding = torch.cat(word_enc_vects, dim=0)
        word_attention = torch.cat(word_att_vects, dim=0)

        word_encoding = (word_encoding * word_attention.unsqueeze(-1)).mean(dim=2)


        
        sent_encoding, sent_attention = self.sent_mh(sent_embeddings, sent_embeddings, sent_embeddings)
        sent_attention = sent_attention.mean(dim=1)
        sent_encopding = sent_encoding * sent_attention.unsqueeze(-1)

        word_sent_comb = torch.cat([
            F.relu(self.word_linear(word_encoding)),  # ReLU activation
            F.relu(self.sent_linear(sent_encoding))  # ReLU activation
        ], dim=1).mean(dim=1)

        word_sent_comb = self.dropout(word_sent_comb)  # Dropout

        
        post_encodings = self.post_linear(post_embeddings)

        post_encodings = self.dropout(post_encodings)  # Dropout

        final_text_encoding = torch.cat([
            word_sent_comb,
            post_encodings
        ], dim=1)

        # final_text_encoding = self.batch_norm(final_text_encoding)  # Batch normalization

        kb_ = self.kb_net(graph.to(device))
        kb_ = self.kb_bn(kb_)
        kb_, _ = torch.max(kb_, dim=0)
        kb_ = kb_.unsqueeze(0).repeat(final_text_encoding.shape[0], 1)

        y = F.relu(self.final_text_linear(final_text_encoding))

        x = torch.cat((y, kb_), dim=1)  # ReLU activation

        x = self.dropout(x)  # Dropout

        output = self.final_linear(x)
        return output, x, word_attention, sent_attention,y,self.kb_net





df['text'] = df['text'].apply(prepare_document)

max_seq_len = np.max([len(x.split('.')) for x in list(df.text)])



params = {'lr': 0.01999606606081115, 'epochs': 100, 'heads': 4, 'gat_heads': 4, 'dropout': 0.4, 'temp': 0.3, 'batch_size': 8, 'scale': 3.0, 'hidden_size': 128, 'max_len': 64,'num_class':4}


labels = nn.functional.softmax(
    torch.tensor(
        -1*params['scale']* np.abs(df['label'].values.reshape(-1,1) - np.tile([0,1,2,3], (len(df), 1))),
        dtype = torch.float32)
    )
file = open(kg_path,'rb')
kb = pickle.load(file)
kg_data = kg_graph_to_data(kb)

texts = list(df.text)

def preparator(docs):
    return [(
        get_word_embeddings(doc).to(device),
        get_sent_embeddings(doc).to(device),
        get_post_embeddings(doc).to(device)
    ) for doc in docs]

embeddings = preparator(texts)

f1_scores = []
precisions = []
recalls = []
accuracies = []
graded_f1 = []

best_one = 0

train_data, test_data,train_labels,test_labels,train_text,test_text = train_test_split(indiv_data,labels,texts,test_size = 0.2)
train_data, val_data,train_labels,val_labels,train_text,val_text = train_test_split(train_data,train_labels,train_text,test_size = 0.2)

train_loader = DataLoader(TextDataset(train_data, train_labels.to(device),train_text), batch_size=params['batch_size'], shuffle=True)
val_loader = DataLoader(TextDataset(val_data, val_labels.to(device),val_text), batch_size=params['batch_size'], shuffle=False)
test_loader = DataLoader(TextDataset(test_data, test_labels.to(device), test_text), batch_size=params['batch_size'], shuffle=False)

for i in trange(10):

    train_data, test_data,train_labels,test_labels,train_text,test_text = train_test_split(embeddings,labels, texts,test_size = 0.2)
    train_data, val_data,train_labels,val_labels,train_text,val_text = train_test_split(train_data,train_labels,train_text, test_size = 0.2)

    train_loader = DataLoader(TextDataset(train_data, train_labels.to(device),train_text), batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(TextDataset(val_data, val_labels.to(device),val_text), batch_size=params['batch_size'], shuffle=False)
    test_loader = DataLoader(TextDataset(test_data, test_labels.to(device),test_text), batch_size=params['batch_size'], shuffle=False)




    model = DepressionX(hidden_size=params['hidden_size'],heads = params['heads'], gat_heads = params['gat_heads'],dropout = params['dropout'],num_class = params['num_class']).to(device)


    early_stop_counter = 0
    early_stop_limit = 15
                    

    criterion = ContrastiveLoss(temp=params['temp'])

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = np.inf

    optimizer = Adam(model.parameters(),lr=params['lr'])

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=10, num_training_steps=params['epochs'])




    for epoch in trange(params['epochs']):

        model.train()
        for data in train_loader:
            optimizer.zero_grad()
            
            output,rep,_,_,_,_ = model(data[0], kg_data)
                    
            loss = criterion(rep, output, data[1])

            torch.cuda.empty_cache()
                    
            loss.backward()
            optimizer.step()
                    
                        
        model.eval()
        for val_data in val_loader:
            val_output,val_rep,_,_,_,_ = model(val_data[0],kg_data)
                    
            val_loss = criterion(val_rep,val_output, val_data[1])

            torch.cuda.empty_cache()
                    
            if scheduler is not None:
                scheduler.step()
            if val_loss >= best_loss:
                early_stop_counter += 1

            if scheduler is not None:
                scheduler.step()
                            
                            
            if val_loss >= best_loss:
                early_stop_counter += 1
            else:
                best_model_wts = copy.deepcopy(model.state_dict())
                early_stop_counter = 0
                best_loss = val_loss

            if early_stop_counter == early_stop_limit:
                break
        torch.cuda.empty_cache()


    model.load_state_dict(best_model_wts)

    preds = []
    test_labels = []
    for test_data in test_loader:
        test_out,_,_,_,_,kg_test= model(test_data[0],kg_data)

        torch.cuda.empty_cache()

        for i in F.log_softmax(test_out, dim=1).max(1)[1].cpu().numpy():
            preds.append(i)
        for j in test_data[1].max(1)[1].cpu().numpy():
            test_labels.append(j)
                                            

    f1_scor = f1_score(
        preds,test_labels, average='weighted'
    )

    g_f1 = graded_eval_metrics(
        preds,test_labels
    )

    prec = precision_score(
        preds,test_labels, average='weighted'
    )

    rec = recall_score(
        preds,test_labels,average='weighted'
    )

    acc = accuracy_score(
        preds,test_labels
    )

    f1_scores.append(f1_scor)
    graded_f1.append(g_f1)
    precisions.append(prec)
    recalls.append(rec)
    accuracies.append(acc)


    if f1_scor >= best_one:
        best_one = f1_scor
        torch.save(model, 'model.pth')

        torch.save(optimizer.state_dict(), 'optimizer.pth')

print(f'F1 Median: {np.median(f1_scores)}, Graded F1 Mean {np.mean(graded_f1)}, Precision Mean {np.median(precisions)}, Recall Mean:{np.median(recalls)}, Accuracy Mean: {np.median(accuracies)}')
print()
print(f'F1 Std: {np.std(f1_scores,ddof = 1)},Graded F1 Std:{np.std(graded_f1,ddof = 1)}, Precision Std {np.std(precisions,ddof = 1)}, Recall Std:{np.std(recalls,ddof = 1)}, Accuracy Std: {np.std(accuracies,ddof = 1)}')











#Graph Explainability part:

class DepExp(nn.Module):
  def __init__(self, hidden_size):
    super(DepExp, self).__init__()
    self.inp_dim = hidden_size
    self.lin = torch.nn.Linear(2*self.inp_dim, 1)
    self.softmax = torch.nn.Softmax(dim=0)
  def forward(self, data):
    EDGE_VEC= torch.tensor(np.array([torch.cat([data.x[x[0]],data.x[x[1]]]).cpu().numpy() for x in data.edge_index.view(-1,2)]))
    new_edges = EDGE_VEC*self.lin(EDGE_VEC)
    new_edges = self.softmax(new_edges).sum(axis=1)
    top_indices = torch.topk(new_edges, k=20,largest = True).indices
    return data.edge_index.T[top_indices].T


exp_model = DepExp(768)
criterion = nn.SmoothL1Loss()
optimizer = optim.Adam(exp_model.parameters(), lr=0.01)

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    nd = Data(x=kg_data.x, edge_index=exp_model(kg_data))
    predictions = kg_test(nd.to(device))
    y = kg_test(kg_data)

    # Calculate the loss
    loss = criterion(predictions, y)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    Print the loss every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')



#Knowledge Graph Generator

# Knowledge Base generation using Wikipedia Data
# from https://huggingface.co/Babelscape/rebel-large
# needed to load the REBEL model

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import math
import torch
import wikipedia
from newspaper import Article, ArticleException
from GoogleNews import GoogleNews
from pyvis.network import Network
import IPython
import pandas as pd
import pickle
import re

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large").to(device)

def extract_relations_from_model_output(text):
    relations = []
    relation, subject, relation, object_ = '', '', '', ''
    text = text.strip()
    current = 'x'
    text_replaced = text.replace("<s>", "").replace("<pad>", "").replace("</s>", "")
    for token in text_replaced.split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                relations.append({
                    'head': subject.strip(),
                    'type': relation.strip(),
                    'tail': object_.strip()
                })
                relation = ''
            subject = ''
        elif token == "<subj>":
            current = 's'
            if relation != '':
                relations.append({
                    'head': subject.strip(),
                    'type': relation.strip(),
                    'tail': object_.strip()
                })
            object_ = ''
        elif token == "<obj>":
            current = 'o'
            relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    if subject != '' and relation != '' and object_ != '':
        relations.append({
            'head': subject.strip(),
            'type': relation.strip(),
            'tail': object_.strip()
        })
    return relations

# extract relations for each span and put them together in a knowledge base
def from_text_to_kb(texts, span_length=128, verbose=False):
    batch_size = 256
    dps = []
    num_of_res = len(texts.split())//batch_size

    for i in range(num_of_res):
        text = " ".join(texts.split()[i*batch_size:(i+1)*batch_size])

        inputs = tokenizer([text], return_tensors="pt")

        # compute span boundaries
        num_tokens = len(inputs["input_ids"][0])
        if verbose:
            print(f"Input has {num_tokens} tokens")
        num_spans = math.ceil(num_tokens / span_length)
        if verbose:
            print(f"Input has {num_spans} spans")
        overlap = math.ceil((num_spans * span_length - num_tokens) /
                            max(num_spans - 1, 1))
        spans_boundaries = []
        start = 0
        for i in range(num_spans):
            spans_boundaries.append([start + span_length * i,
                                    start + span_length * (i + 1)])
            start -= overlap
        if verbose:
            print(f"Span boundaries are {spans_boundaries}")

        # transform input with spans
        tensor_ids = [inputs["input_ids"][0][boundary[0]:boundary[1]]
                    for boundary in spans_boundaries]
        tensor_masks = [inputs["attention_mask"][0][boundary[0]:boundary[1]]
                        for boundary in spans_boundaries]
        inputs = {
            "input_ids": torch.stack(tensor_ids).to(device),
            "attention_mask": torch.stack(tensor_masks).to(device)
        }

        # generate relations
        num_return_sequences = 3
        gen_kwargs = {
            "max_length": 256,
            "length_penalty": 0,
            "num_beams": 3,
            "num_return_sequences": num_return_sequences
        }
        generated_tokens = model.generate(
            **inputs,
            **gen_kwargs,
        )

        # decode relations
        dps.append(tokenizer.batch_decode(generated_tokens,
                                            skip_special_tokens=False))
    text = " ".join(texts.split()[(i+1)*batch_size:])
    inputs = tokenizer([text], return_tensors="pt")
    num_tokens = len(inputs["input_ids"][0])
    if verbose:
        print(f"Input has {num_tokens} tokens")
    num_spans = math.ceil(num_tokens / span_length)
    if verbose:
        print(f"Input has {num_spans} spans")
    overlap = math.ceil((num_spans * span_length - num_tokens) /
                        max(num_spans - 1, 1))
    spans_boundaries = []
    start = 0
    for i in range(num_spans):
        spans_boundaries.append([start + span_length * i,
                                start + span_length * (i + 1)])
        start -= overlap
    if verbose:
        print(f"Span boundaries are {spans_boundaries}")

        # transform input with spans
    tensor_ids = [inputs["input_ids"][0][boundary[0]:boundary[1]]
                for boundary in spans_boundaries]
    tensor_masks = [inputs["attention_mask"][0][boundary[0]:boundary[1]]
                    for boundary in spans_boundaries]
    inputs = {
        "input_ids": torch.stack(tensor_ids).to(device),
        "attention_mask": torch.stack(tensor_masks).to(device)
    }

        # generate relations
    num_return_sequences = 3
    gen_kwargs = {
        "max_length": 256,
        "length_penalty": 0,
        "num_beams": 3,
        "num_return_sequences": num_return_sequences
    }
    generated_tokens = model.generate(
        **inputs,
        **gen_kwargs,
    )

        # decode relations
    dps.append(tokenizer.batch_decode(generated_tokens,
                                        skip_special_tokens=False))


    # create kb
    kb = KB()

    for decoded_preds in dps:
        i = 0
        for sentence_pred in decoded_preds:
            current_span_index = i // num_return_sequences
            relations = extract_relations_from_model_output(sentence_pred)
            for relation in relations:
                relation["meta"] = {
                    "spans": [spans_boundaries[current_span_index]]
                }
                kb.add_relation(relation)
            i += 1

    return kb

class KB():
    def __init__(self):
        self.entities = {}
        self.relations = []
    def are_relations_equal(self, r1, r2):
        return all(r1[attr] == r2[attr] for attr in ["head", "type", "tail"])

    def exists_relation(self, r1):
        return any(self.are_relations_equal(r1, r2) for r2 in self.relations)

    def add_relation(self, r):
        if not self.exists_relation(r):
            self.relations.append(r)

    def print(self):
        print("Relations:")
        for r in self.relations:
            print(f"  {r}")

    def merge_relations(self, r1):
        r2 = [r for r in self.relations
              if self.are_relations_equal(r1, r)][0]
        spans_to_add = [span for span in r1["meta"]["spans"]
                        if span not in r2["meta"]["spans"]]
        r2["meta"]["spans"] += spans_to_add

    def get_wikipedia_data(self, candidate_entity):
        try:
            page = wikipedia.page(candidate_entity, auto_suggest=False)
            entity_data = {
                "title": page.title,
                "url": page.url,
                "summary": page.summary
            }
            return entity_data
        except:
            return None

    def add_entity(self, e):
        self.entities[e["title"]] = {k:v for k,v in e.items() if k != "title"}

    def add_relation(self, r):
        # check on wikipedia
        candidate_entities = [r["head"], r["tail"]]
        entities = [self.get_wikipedia_data(ent) for ent in candidate_entities]

        # if one entity does not exist, stop
        if any(ent is None for ent in entities):
            return

        # manage new entities
        for e in entities:
            self.add_entity(e)

        # rename relation entities with their wikipedia titles
        r["head"] = entities[0]["title"]
        r["tail"] = entities[1]["title"]

        # manage new relation
        if not self.exists_relation(r):
            self.relations.append(r)
        else:
            self.merge_relations(r)

    def print(self):
        print("Entities:")
        for e in self.entities.items():
            print(f"  {e}")
        print("Relations:")
        for r in self.relations:
            print(f"  {r}")

def clean_string(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)

    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)
    string = emoji_pattern.sub(r'', string)

    return string.strip().lower()


df['text'] = df['text'].apply(clean_string)
text = " ".join(list(df.text))

kb = from_text_to_kb(text)
kb.print()

with open('data_path/kg_graphs.pickle', 'wb') as handle:
        pickle.dump(kb, handle)





#Knowledge Graph Embedder







import pickle
import networkx as nx
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import math
import torch
from ex_knowledge_base_generator import KB
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import pickle
from transformers import BertModel, BertTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
warnings.filterwarnings("ignore")
from sentence_transformers import SentenceTransformer
sent_emb_model = SentenceTransformer('all-distilroberta-v1',device=device)


from gensim.models import Word2Vec
wv_model = Word2Vec.load("text8_300d.model")


def get_wv_embeddings(sentences):
    vectors = []
    for i in sentences:
        try:
            vectors.append(wv_model.wv[i.lower()])
        except:
            vectors.append(np.zeros(300))
    return vectors


file1 = open('data_path/kg_graphs.pickle', 'rb')
kb1 = pickle.load(file1)

G = nx.DiGraph()


symptom_words = [
    "Sadness",
    "Pessimism",
    "Past failure",
    "Loss of pleasure",
    "Guilty feelings",
    "Punishment feelings",
    "Self-dislike",
    "Self-criticalness",
    "Suicidal thoughts",
    "Crying",
    "Agitation",
    "Loss of interest",
    "Indecisiveness",
    "Worthlessness",
    "Loss of energy",
    "Changes in sleep patterns",
    "Irritability",
    "Changes in appetite",
    "Concentration difficulty",
    "Tiredness or fatigue",
    "Loss of interest in sex"
]

for i in tqdm(kb1.entities):
    txt = kb1.entities[i]['summary']
    try:
        val = cosine_similarity([wv_model.wv[i.lower()]], get_wv_embeddings(symptom_words)).max()
        if  val >= 0.5:
            feats = torch.tensor(sent_emb_model.encode(txt),dtype=torch.float32).detach().cpu()
            G.add_node(i,entity = i,text = txt, features = feats.cpu().numpy() )
        else:
            continue
    except:
        continue

for i in tqdm(kb1.relations):
    if (i['head'] != i['tail']) and (i['head'] in G.nodes) and (i['tail'] in G.nodes):
        try:
            G.add_edge(i['head'],i['tail'])
        except:
            continue


nodes_to_remove = [node for node in G.nodes if G.degree(node) == 0]
G.remove_nodes_from(nodes_to_remove)


nodes  = {value: index for index, value in enumerate(set(G.nodes))}
G = nx.relabel_nodes(G, nodes)


with open('data_path/data_kg_graphs.pickle', 'wb') as handle:
    pickle.dump(G, handle)
