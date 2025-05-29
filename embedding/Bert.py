import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import load_dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments, BertModel

import os
import shutil
import sys

import time

from tqdm.contrib import tzip

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from preprocess.dataLoad import Paper, papers_load, papers_transform
from sentence_transformers import SentenceTransformer

data_path = '../public/arxiv_classification_dataset'
model_path = "../public/scibert_scivocab_cased"

datafiles = {
    "train": f"{data_path}/train*.parquet",
    "test": f"{data_path}/test*.parquet",
    "valid": f"{data_path}/validation*.parquet"
    }

dataset = load_dataset('parquet', data_files=datafiles)

class Bert_Embedding(nn.Module):
    def __init__(self):
        super(Bert_Embedding, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = SentenceTransformer(model_path)
    
    def encode(self, texts, batch_size=8, show_progress_bar=True):
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar
        )
    
    def embed_and_save(self, papers, save_path):
        st = time.time()
        print("开始加载数据")
        texts = [i.fields['AB'] for i in papers]
        titles = [i.fields['TI'] for i in papers]
        print("成功加载数据")
        print("开始编码")
        
        embeddings = self.model.encode(texts, batch_size=16, show_progress_bar=True)
        print("成功编码")
        
        print("开始保存数据")
        np.save(os.path.join(save_path, "paper_embeddings.npy"), embeddings)
        np.save(os.path.join(save_path, "paper_titles.npy"), titles)
        print("成功保存数据")
        et = time.time()
        print(f"耗时: {et - st:.4f} 秒")
        return embeddings, titles
    
    
if __name__ == '__main__':
    model = Bert_Embedding()
    
    data_path = path = '../public/NSP'
    papers_str = papers_load(path)
    papers = papers_transform(papers_str)
    print("数据加载完成")
    
    save_path = "../public"
    embeddings, titles = model.embed_and_save(papers, save_path)
    print(embeddings[:10])
    
