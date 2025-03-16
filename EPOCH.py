#!/usr/bin/env python
#_*_coding:utf-8_*_

import os
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from datasets import Dataset #Dataset class of hungging face
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import time
import numpy as np

input = sys.argv[1]#sequence file, 'Name' column for sequence id, 'VH' column for heavy chain Fv sequence,  'VL' column for light chain Fv sequence 

def create_dataset(tokenizer,df):
    seqs_df = df.copy()
    seqs_df['ab_seq1_len'] = seqs_df["antibody_seq1"].fillna('').str.len()
    seqs_df['ab_seq2_len'] = seqs_df["antibody_seq2"].fillna('').str.len()
    seqs_df['ag_len'] = seqs_df["antigen_seq"].fillna('').str.len()

    seqs_df['complex_seq'] = (seqs_df["antibody_seq1"].fillna('') 
                              + seqs_df["antibody_seq2"].fillna('')
                              + seqs_df["antigen_seq"].fillna('')
                              )
    seqs_df['complex_len'] = seqs_df["complex_seq"].str.len()

    seqs_df["complex_seq"] = seqs_df["complex_seq"].str.replace('[OBUZ]', 'X', regex=True) 
    seqs_df['complex_seq'] = seqs_df.apply(lambda row: " ".join(row["complex_seq"]), axis=1)
    
    tokenized = tokenizer(list(seqs_df["complex_seq"]), max_length=1024, padding=False, truncation=True)
    dataset = Dataset.from_dict(tokenized)
    dataset = dataset.with_format("torch")
    return dataset, seqs_df

def model_test(model, test_set, test_df, device, threshold = 0.1191):
    predictions = []
    model.to(device)
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(len(test_set['input_ids']))):
            input_ids = test_set['input_ids'][i].unsqueeze(0).to(device)  
            attention_mask = test_set['attention_mask'][i].unsqueeze(0).to(device)  
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            prob = output.logits.softmax(dim=-1)[:,:,1][0][1:-1].tolist() # [1:-1] delete the prediction results of CLS and SEP

            prob = np.array(prob)
            pred_label = (prob >= threshold).astype(int)

            ab1_len = test_df['ab_seq1_len'].iloc[i]
            ab2_len = test_df['ab_seq2_len'].iloc[i]
            
            prob_rounded = prob.round(4)
            #predict label of ab_seq1
            prob_ab1 = prob_rounded[0:ab1_len]
            pred_label_ab1 = pred_label[0:ab1_len]

            ##predict label of ab_seq2 if exist
            if ab2_len == 0:
                prob_ab2 = []
                pred_label_ab2 = []
            else:
                prob_ab2 = prob_rounded[ab1_len:ab1_len + ab2_len]
                pred_label_ab2 = pred_label[ab1_len:ab1_len + ab2_len]

            #predict label of antigen  
            prob_ag = prob_rounded[ab1_len + ab2_len:]
            pred_label_ag = pred_label[ab1_len + ab2_len:]

            predictions.append({
                'id': test_df['id'].iloc[i],
                'ab_seq1_pred_prob': list(prob_ab1),
                'ab_seq1_pred_label': list(pred_label_ab1),
                'ab_seq2_pred_prob': list(prob_ab2),
                'ab_seq2_pred_label': list(pred_label_ab2),
                'ag_pred_prob':list(prob_ag),
                'ag_pred_label':list(pred_label_ag)})
            
    return pd.DataFrame(predictions)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained('./EPOCH')
trained_model = AutoModelForTokenClassification.from_pretrained('./EPOCH', num_labels=2)

test = pd.read_csv(input,index_col=False,sep='\t')
test_data,test_df = create_dataset(tokenizer, test)
predictions = model_test(trained_model,test_data, test_df,device,threshold = 0.1191)
pred_results = pd.merge(test, predictions,on='id', how='left')
pred_results.to_csv('./results/pred_results_'+time.strftime('%Y%m%d%H%M%S', time.localtime())+'.txt',sep='\t',index=False)