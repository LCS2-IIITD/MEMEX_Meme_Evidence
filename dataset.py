import os
from ast import literal_eval

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence

from PIL import Image

class MemeExpDataset(Dataset):
    
    def __init__(self, dataset_path, tokenizer, transform, image_dir, ftr_dir):
        
        self.dataset_path = dataset_path
        self.tokenizer = tokenizer
        self.transform = transform
        self.image_dir = image_dir
        self.ftr_dir = ftr_dir
        
        self.data = pd.read_csv(dataset_path)

        self.embeddings = []
        for idx in range(len(self.data)):
            ftr_path = os.path.join(self.ftr_dir, f"{idx}.pt")
            ftr = torch.load(ftr_path)
            self.embeddings.append(ftr)
        
    def __len__(self):
        return len(self.data)
#         return 10
    
    def get_encoded_text(self, text: str):
        
        
        encoded_inputs = self.tokenizer(
            text,
            padding="max_length",
            max_length=80,
            truncation=True,
            return_tensors='pt'
            )
        
        input_ids = encoded_inputs['input_ids'].squeeze()
        attention_mask = encoded_inputs['attention_mask'].squeeze()
        token_type_ids = encoded_inputs['token_type_ids'].squeeze()
        
        return input_ids, attention_mask, token_type_ids
    
    def __getitem__(self, idx):
        
        ocr_text = ' '.join(self.data.iloc[idx]['ocr_text'].split('\n'))
        
        ocr_text = ocr_text if isinstance(ocr_text, str) is True else ""
        
        texts = literal_eval(self.data.iloc[idx]['sentences'])
        
        texts = [text if isinstance(text, str) is True else "" for text in texts]
        
        kg_embs = self.embeddings[idx]
        
        
        # Image data
        image_name = self.data.iloc[idx]['image']
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        
        if self.transform is not None:
            image = self.transform(image=image)['image']
        
        # Encoded ocr text
        input_ids, attention_mask, token_type_ids = self.get_encoded_text(ocr_text)
        
        # Encoded context texts
        encoded_texts = [self.get_encoded_text(text) for text in texts]
        context_input_ids = [input_ids for input_ids, attention_mask, _ in encoded_texts]
        context_attention_mask = [attention_mask for input_ids, attention_mask, _ in encoded_texts]         
        
        # Output label
        label = literal_eval(self.data.iloc[idx]['labels'])
        
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'context_input_ids': context_input_ids,
            'context_attention_mask': context_attention_mask,
            'label': label,
            'image': image,
            'num_sents': len(encoded_texts),
            'kg_embs': kg_embs,
        }
    
def collate_fn(batch):
    
    input_size = list(batch[0]['input_ids'].shape)
    dummy_ids = torch.zeros(input_size).int()
    
    max_num_sents = max([item['num_sents'] for item in batch])
    
    
    # Pad the context inputs
    ctx_input_ids_list = []
    ctx_attention_mask_list = []
    for item in batch: 
        _pad_ids = [dummy_ids] * (max_num_sents - item['num_sents'])
        _input_ids = torch.stack(item['context_input_ids'] + _pad_ids)
        _attention_mask = torch.stack(item['context_attention_mask'] + _pad_ids)
        
        ctx_input_ids_list.append(_input_ids)
        ctx_attention_mask_list.append(_attention_mask)
    
    txt_input_ids = torch.stack([item['input_ids'] for item in batch])
    txt_attention_mask = torch.stack([item['attention_mask'] for item in batch])
    token_type_ids = torch.stack([item['token_type_ids'] for item in batch])
    ctx_input_ids = torch.stack(ctx_input_ids_list)
    ctx_attention_mask = torch.stack(ctx_attention_mask_list)
    
    num_sents = [item['num_sents'] for item in batch]
    image = torch.stack([item['image'] for item in batch])
    
    kg_embs = torch.stack([item['kg_embs'] for item in batch])
    
    labels = [item['label'] for item in batch]
    
    return {
        'input_ids': txt_input_ids, 
        'attention_mask': txt_attention_mask,
        'token_type_ids': token_type_ids, 
        'ctx_input_ids': ctx_input_ids,
        'ctx_attention_mask': ctx_attention_mask,
        'num_sents': num_sents,
        'image': image,
        'label': labels,
        'kg_embs': kg_embs
    }