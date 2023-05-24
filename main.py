import json
import math
import warnings
from ast import literal_eval
from typing import Optional, Union
import copy
import pickle

import numpy as np
import pandas as pd

import random
import matplotlib.pyplot as plt
import seaborn as sns

import os 
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence

import torchvision

from PIL import Image

from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, classification_report
from sklearn.preprocessing import MultiLabelBinarizer

import albumentations as A
from albumentations.pytorch import ToTensorV2

from easydict import EasyDict as edict

import wandb

from dataset import MemeExpDataset, collate_fn
from modules.mmbt_model import MultimodalBertEncoder
from modules.context_transformer import ContextualTransformerEncoder
from modules.gated_fusion import GatedFusion, GMF

os.environ['CUDA_VISIBLE_DEVICES'] = "1"

api_key = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
os.environ['WANDB_API_KEY'] = api_key

torch.cuda.empty_cache()

# Configuration
exp_name = "12.2.ca_gated_synlstm"
IMG_SIZE = 224
IMAGE_DIR = './data/images'

# Albumentation transforms

train_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE), 
    A.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

test_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE), 
    A.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

class MALSTM(nn.Module):
    def __init__(self, input_sz, hidden_sz, g_sz):
        super(DiscLSTM, self).__init__()
        self.input_sz = input_sz
        self.hidden_sz = hidden_sz
        self.g_sz = g_sz
        self.all1 = nn.Linear((self.hidden_sz * 1 + self.input_sz  * 1),  self.hidden_sz)
        self.all2 = nn.Linear((self.hidden_sz * 1 + self.input_sz  +self.g_sz), self.hidden_sz)
        self.all3 = nn.Linear((self.hidden_sz * 1 + self.input_sz  +self.g_sz), self.hidden_sz)
        self.all4 = nn.Linear((self.hidden_sz * 1 + self.input_sz  * 1), self.hidden_sz)

        self.all11 = nn.Linear((self.hidden_sz * 1 + self.g_sz),  self.hidden_sz)
        self.all44 = nn.Linear((self.hidden_sz * 1 + self.g_sz), self.hidden_sz)

        self.init_weights()
        self.drop = nn.Dropout(0.5)
    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_sz)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def node_forward(self, xt, ht, Ct_x, mt, Ct_m):

        # # # new standard lstm
        hx_concat = torch.cat((ht, xt), dim=1)
        hm_concat = torch.cat((ht, mt), dim=1)
        hxm_concat = torch.cat((ht, xt, mt), dim=1)


        i = self.all1(hx_concat)
        o = self.all2(hxm_concat)
        f = self.all3(hxm_concat)
        u = self.all4(hx_concat)
        ii = self.all11(hm_concat)
        uu = self.all44(hm_concat)

        i, f, o, u = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o), torch.tanh(u)
        ii,uu = torch.sigmoid(ii), torch.tanh(uu)
        Ct_x = i * u + ii * uu + f * Ct_x
        ht = o * torch.tanh(Ct_x) 

        return ht, Ct_x, Ct_m 

    def forward(self, x, m, init_stat=None):
        batch_sz, seq_sz, _ = x.size()
        hidden_seq = []
        cell_seq = []
        if init_stat is None:
            ht = torch.zeros((batch_sz, self.hidden_sz)).to(x.device)
            Ct_x = torch.zeros((batch_sz, self.hidden_sz)).to(x.device)
            Ct_m = torch.zeros((batch_sz, self.hidden_sz)).to(x.device)
        else:
            ht, Ct = init_stat
        for t in range(seq_sz):  # iterate over the time steps
            xt = x[:, t, :]
            mt = m[:, t, :]
            ht, Ct_x, Ct_m= self.node_forward(xt, ht, Ct_x, mt, Ct_m)
            hidden_seq.append(ht)
            cell_seq.append(Ct_x)
            if t == 0:
                mht = ht
                mct = Ct_x
            else:
                mht = torch.max(torch.stack(hidden_seq), dim=0)[0]
                mct = torch.max(torch.stack(cell_seq), dim=0)[0]
        hidden_seq = torch.stack(hidden_seq).permute(1, 0, 2) ##batch_size x max_len x hidden
        return hidden_seq

class MemeExpModel(nn.Module):
    
    def __init__(self):
        
        super().__init__()
        
        self.text_encoder = AutoModel.from_pretrained('bert-base-uncased')
        
        # Arguments for MMBT Classifier
        args = {
            'img_hidden_sz': 2048,
            'hidden_sz': 768,
            'img_embed_pool_type': 'avg',
            'num_image_embeds': 1,
            'cls_token_id': 101,
            'sep_token_id': 102,
            'dropout': 0.1,
            'bert_model': 'bert-base-uncased'
        }

        self.gated_fusion = GatedFusion(768)
        self.gated_fusion_2 = GatedFusion(768)

        self.kg_proj = nn.Sequential(
            nn.Linear(1024, 768),
            nn.LeakyReLU(),
            nn.Dropout(0.1)
        )
            
        args = edict(args)
        
        self.meme_encoder = MultimodalBertEncoder(args)    
        
        dropout_prob = 0.1
        
        self.dropout = nn.Dropout(dropout_prob)
        
        hidden_dim = 768
        
        dim_model = 768
        dim_context =  768
        dropout_rate = 0.1
        self.transformer_encoder = ContextualTransformerEncoder(d_model=768, d_ff=768, n_heads=8, n_layers=1)
        
        hidden_dim = 768
        emb_dim = 768
        graph_dim = 768
        
        self.lstm = MALSTM(emb_dim, hidden_dim, graph_dim)

        
        hidden_dim = 768 + 768
        
        self.clf = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim//2, 1)
        )
        
    def forward(self, kg_embs, image, input_ids, attention_mask, token_type_ids,
               ctx_input_ids, ctx_attention_mask,
               num_sents):
        
        meme_ftrs = self.meme_encoder(input_ids, attention_mask, token_type_ids, image)
        
        
        kg_embs = self.kg_proj(kg_embs)
        
        meme_ftrs = self.gated_fusion(meme_ftrs, kg_embs)
        
        fused_embeddings = []
        
        for idx, num_sent in enumerate(num_sents):
                   
            _input_ids = ctx_input_ids[idx, :num_sent]
            _attention_mask = ctx_attention_mask[idx, :num_sent]
          
            output = self.text_encoder(
                _input_ids, _attention_mask
            )
            
            ctx_ftrs = output.pooler_output.unsqueeze(0)
        
        
            meme_embeddings = meme_ftrs[idx].unsqueeze(0).repeat((num_sent, 1))
            
            meme_ctx = meme_embeddings.unsqueeze(0)

            ctx_ftrs = self.transformer_encoder(ctx_ftrs, meme_ctx)        
            
            last_hidden_state = self.dropout(ctx_ftrs)
            ctx_ftrs = self.lstm(last_hidden_state, meme_ctx).squeeze(0)
            
            fused_embedding = torch.cat((meme_embeddings, ctx_ftrs), dim=1)
            
            fused_embeddings.append(fused_embedding)

        logits_list = []
        for fused_embedding in fused_embeddings:
            logits = self.clf(fused_embedding)
            logits_list.append(logits)

        return logits_list


    
    
def exact_match(y_true, y_pred):
    
    acc_list = []
    
    for y, y_hat in zip(y_true, y_pred):
        
        if y == y_hat:
            acc_list.append(1.0)
        else:
            acc_list.append(0.0)
   
    return sum(acc_list)/len(acc_list)
    
def hamming_score(y_true, y_pred):
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/\
                    float( len(set_true.union(set_pred)) )
            
           
            
        acc_list.append(tmp_a)
        
    if np.sum(acc_list) == 0:
        return 0
    
    return np.mean(acc_list)

def compute_macro_f1(y_true_list, y_pred_list):
   
    y_true_list = [y_true 
                   if type(y_true) == list else [y_true]
                   for y_true in y_true_list]
    y_pred_list = [y_pred
                   if type(y_pred) == list else [y_pred]
                   for y_pred in y_pred_list]

    score_list = [f1_score(y_true, y_pred, average='macro') 
                  for y_true, y_pred in 
                  zip(y_true_list, y_pred_list)]
    
    return np.mean(score_list)

def compute_macro_precision(y_true_list, y_pred_list):
   
    y_true_list = [y_true 
                   if type(y_true) == list else [y_true]
                   for y_true in y_true_list]
    y_pred_list = [y_pred
                   if type(y_pred) == list else [y_pred]
                   for y_pred in y_pred_list]
    
    score_list = [precision_score(y_true, y_pred, average='macro') 
                  for y_true, y_pred in 
                  zip(y_true_list, y_pred_list)]
    
    return np.mean(score_list)

def compute_macro_recall(y_true_list, y_pred_list):
 
    y_true_list = [y_true 
                   if type(y_true) == list else [y_true]
                   for y_true in y_true_list]
    y_pred_list = [y_pred
                   if type(y_pred) == list else [y_pred]
                   for y_pred in y_pred_list]
    
    score_list = [recall_score(y_true, y_pred, average='macro') 
                  for y_true, y_pred in 
                  zip(y_true_list, y_pred_list)]
    
    return np.mean(score_list)

    
if __name__ == "__main__":
    
    print("-"*30)
    print("Starting training script")
    print("-"*30)
    
    train_dataset_path = './data/sample.csv'
    val_dataset_path = './data/sample.csv'
    test_dataset_path = './data/sample.csv'
    
    train_ftr_dir = './data/kg_ftrs'
    test_ftr_dir = './data/kg_ftrs'
    
    train_dataset = MemeExpDataset(train_dataset_path, tokenizer, train_transform, IMAGE_DIR, train_ftr_dir)
    val_dataset = MemeExpDataset(val_dataset_path, tokenizer, train_transform, IMAGE_DIR, test_ftr_dir)
    test_dataset = MemeExpDataset(test_dataset_path, tokenizer, train_transform, IMAGE_DIR, test_ftr_dir)

    TRAIN_BATCH_SIZE = 16
    TEST_BATCH_SIZE = 16
    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, 
                              drop_last=False, pin_memory=False, 
                              num_workers=0, collate_fn=collate_fn)
    
    val_loader = DataLoader(val_dataset, batch_size=TEST_BATCH_SIZE, 
                              drop_last=False, pin_memory=False, 
                              num_workers=0, collate_fn=collate_fn)
    
    test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, 
                              drop_last=False, pin_memory=False, 
                              num_workers=0, collate_fn=collate_fn)
    
    model = MemeExpModel()

    _ = model.cuda()
    
    params = list(model.parameters())
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(params, lr=1e-4)
    
    epochs = 30
        
    save_dir = './model_ckpt/bert'
    
    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)
        
    train_step = 0
    val_step = 0
    test_step = 0
    
    
#     warnings.filterwarnings("ignore")
    for epoch in range(epochs):
        
        total_target = {
            'train': [],
            'test': [],
            'val': [],
        }
        
        total_preds = {
            'train': [],
            'test': [],
            'val': []
        }
        
        losses = {
            'train': [],
            'test': [],
            'val': [],
        }
        
        
        model.train()
        
        for i, batch in enumerate(tqdm(train_loader)):
            
            

            # Collect inputs
            kg_embs = batch['kg_embs'].cuda()
            image = batch['image'].cuda()
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            token_type_ids = batch['token_type_ids'].cuda()
            ctx_input_ids = batch['ctx_input_ids'].cuda()
            ctx_attention_mask = batch['ctx_attention_mask'].cuda()
            targets = batch['label']
            num_sents = batch['num_sents']
            
            optimizer.zero_grad()
            
            logits_list = model(kg_embs, image, input_ids, attention_mask, token_type_ids, 
                          ctx_input_ids, ctx_attention_mask, num_sents)
            
            
            loss = torch.tensor(0).float().cuda()
            
            for idx, target in enumerate(targets):
                _target = torch.tensor(target).unsqueeze(1).float().cuda()
                logits = logits_list[idx]

                _loss = criterion(logits, _target)

                loss += _loss
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
            preds = [torch.round(torch.sigmoid(logits.detach().squeeze())).cpu().int().tolist() for logits in logits_list]
            
            total_target['train'].extend(targets)
            total_preds['train'].extend(preds)
            
            losses['train'].append(loss.detach().cpu())
            
#             wandb.log({"epoch": epoch, "train_loss": loss}, step=i)
            wandb.log({"epoch": epoch, "train_loss": loss})
            train_step += 1
            
            torch.cuda.empty_cache()

            
        with torch.no_grad():
    
            model.eval()
            for j, batch in enumerate(tqdm(val_loader)):

                
                # Collect inputs
                kg_embs = batch['kg_embs'].cuda()
                image = batch['image'].cuda()
                input_ids = batch['input_ids'].cuda()
                attention_mask = batch['attention_mask'].cuda()
                token_type_ids = batch['token_type_ids'].cuda()
                ctx_input_ids = batch['ctx_input_ids'].cuda()
                ctx_attention_mask = batch['ctx_attention_mask'].cuda()
                targets = batch['label']
                num_sents = batch['num_sents']

                logits_list = model(kg_embs, image, input_ids, attention_mask, token_type_ids,
                              ctx_input_ids, ctx_attention_mask, num_sents)

                loss = torch.tensor(0).float().cuda()
                for idx, target in enumerate(targets):
                    _target = torch.tensor(target).unsqueeze(1).float().cuda()
                    logits = logits_list[idx]

                    _loss = criterion(logits, _target)

                    loss += _loss
                    
                losses['val'].append(loss.detach().cpu())
                
                wandb.log({"val_loss": loss})
                val_step += 1

                preds = [torch.round(torch.sigmoid(logits.detach().squeeze())).cpu().int().tolist() for logits in logits_list]

                total_target['val'].extend(targets)
                total_preds['val'].extend(preds)
                
                torch.cuda.empty_cache()

                
            for j, batch in enumerate(tqdm(test_loader)):

                # Collect inputs
                kg_embs = batch['kg_embs'].cuda()
                image = batch['image'].cuda()
                input_ids = batch['input_ids'].cuda()
                token_type_ids = batch['token_type_ids'].cuda()
                attention_mask = batch['attention_mask'].cuda()
                ctx_input_ids = batch['ctx_input_ids'].cuda()
                ctx_attention_mask = batch['ctx_attention_mask'].cuda()
                targets = batch['label']
                num_sents = batch['num_sents']

                logits_list = model(kg_embs, image, input_ids, attention_mask, token_type_ids, 
                              ctx_input_ids, ctx_attention_mask, num_sents)
                
                loss = torch.tensor(0).float().cuda()
                for idx, target in enumerate(targets):
                    _target = torch.tensor(target).unsqueeze(1).float().cuda()
                    logits = logits_list[idx]

                    _loss = criterion(logits, _target)

                    loss += _loss
                    
                losses['test'].append(loss.detach().cpu())
                
                wandb.log({"test_loss": loss})
                test_step += 1
                    
                preds = [torch.round(torch.sigmoid(logits.detach().squeeze())).cpu().int().tolist() for logits in logits_list]

                total_target['test'].extend(targets)
                total_preds['test'].extend(preds)
                
                torch.cuda.empty_cache()
                



                
        epoch_logs = []
        for phase in ['train', 'val', 'test']:
            
            phase_loss = torch.stack(losses[phase]).mean()
            
            if phase == 'test':
                
                pred_log_dir = os.path.join('pred_logs', exp_name)
                
                if os.path.exists(pred_log_dir) is False:
                    os.makedirs(pred_log_dir)
                
                pred_save_path = os.path.join(pred_log_dir, f"{epoch}_pred.pkl")
                
                pred_data = {
                    'preds': total_preds[phase],
                    'target': total_target[phase]
                }
                
                with open(pred_save_path, 'wb') as f:
                    pickle.dump(pred_data, f)
            
            epoch_logs.append("-"*40)
            epoch_logs.append(f"{phase.capitalize()} Results Epoch [{epoch+1}/{epochs}], Loss: {phase_loss.item():.4f}")
            epoch_logs.append(f"Hamming score: {hamming_score(total_target[phase], total_preds[phase]):.4f}")
            epoch_logs.append(f"F1 score: {compute_macro_f1(total_target[phase], total_preds[phase]):.4f}")
            epoch_logs.append(f"Exact Match score: {exact_match(total_target[phase], total_preds[phase]):.4f}")
            epoch_logs.append(f"Recall score: {compute_macro_recall(total_target[phase], total_preds[phase]):.4f}")
            epoch_logs.append(f"Precision score: {compute_macro_precision(total_target[phase], total_preds[phase]):.4f}")
            
        for log_line in epoch_logs:
            print(log_line)
           
        epoch_logs = [f"{log_line}\n" for log_line in epoch_logs]
        
        
#         12.1.ca_gated_synlstm
        with open(f"./result_logs/{exp_name}.txt", 'a') as f:
            f.writelines(epoch_logs)

