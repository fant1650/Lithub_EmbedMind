import os
import shutil
import re
from datetime import datetime
import logging

import numpy as np
import pandas as pd

from tqdm import tqdm

def papers_yield(path):
    dic = {}
    priority = {'N_':0, 'P1':1, 'P2':2, 'P3':3, 'S_':4}
    files = os.listdir(path)
    files = sorted(files, key=lambda s:(priority[s[:2]], int(re.search(r'\d+-', s).group()[:-1])))
    for i, file in enumerate(files):
        data_path = os.path.join(path, file)
        
        with open(data_path, 'r', encoding='utf-8-sig') as f:
            data500 = [p[:-1] for p in''.join(f.readlines()[:-1]).split('ER\n\n')[:-1]]
        for paper_str in data500:
            yield paper_str.strip()

def papers_load(path):
    dic = {}
    papers_str = []
    priority = {'N_':0, 'P1':1, 'P2':2, 'P3':3, 'S_':4}
    files = os.listdir(path)
    files = sorted(files, key=lambda s:(priority[s[:2]], int(re.search(r'\d+-', s).group()[:-1])))
    for i, file in tqdm(enumerate(files)):
        data_path = os.path.join(path, file)
        
        with open(data_path, 'r', encoding='utf-8-sig') as f:
            data500 = [p[:-1] for p in''.join(f.readlines()[:-1]).split('ER\n\n')[:-1]]
        papers_str += data500
    return papers_str

class Paper:
    def __init__(self, paper_str: str):
        self.fields = {
            'FN': "Clarivate Analytics Web of Science",'VR': "",'PT': "",'AU': [],'AF': [],'TI': "",  
            'SO': "",'LA': "English",'DT': "",'DE': [],'ID': [],'AB': "",'C1': [],'C3': [],'RP': [],  
            'EM': "",'RI': [],'OI': [],'FU': [],'FX': [],'CR': [],'NR': None,'TC': None,'Z9': None,  
            'U1': None,'U2': None,'PU': "",'PI': "",'PA': "",'SN': "",'EI': "",'J9': "",'JI': "",  
            'PD': "",'PY': None,'VL': "",'IS': "",'BP': "",'EP': "",'PG': None,'DI': "",'WC': "",  
            'WE': [],'SC': [],'GA': "",'UT': "",'PM': "",'OA': "",'DA': "",'ER': ""
            }
        self.valid_fields = []
        if paper_str:
            self._parse_paper_str(paper_str)
    def _parse_paper_str(self, paper_str: str): #Parse a paper information string into structured data
        lines = [line.strip() for line in paper_str.split('\n') if line.strip()]
        current_field = None #Track current field and its multi-line value
        current_value = []
        for line in lines:
            if len(line) >= 2 and line[:2].isupper() and line[:2] in self.fields: #Check if line starts with a field
                if current_field is not None: #Save previous field's value if exists
                    self._set_field(current_field, '\n'.join(current_value))
                    current_value = []
                
                field = line[:2] #Start new field
                value = line[2:].strip()
                current_field = field
                current_value.append(value)
                self.valid_fields.append(field)
            else:
                if current_field is not None:
                    current_value.append(line)
                    
        if current_field is not None:
            self._set_field(current_field, '\n'.join(current_value))
    def _set_field(self, field: str, value: str):
        try:
            if field in ['AU','AF','C1','C3','RP','ID','DE','CR','WE','SC','RI','OI','FU','FX']:
                values = [v.strip() for v in value.split('\n') if v.strip()]
                if field == 'CR':
                    self.fields[field].extend([{'text': v} for v in values])
                else:
                    self.fields[field].extend(values)
            else:
                self.fields[field] = value
        except ValueError:
            logging.warning(f"字段{field}有误: {value}")
            self.fields[field] = value    
            

def papers_transform(papers_str):
    res = []
    for i in papers_str:
        paper = Paper(i)
        res.append(paper)
    return res
    
class DataLoader_sklearn:
    def __init__(self, papers):
        data = []
        for i, p in enumerate(papers):
            vector = [p.fields['TI'], p.fields['SO'], p.fields['DT'], p.fields['ID'], 
                      p.fields['AB'], ','.join(p.fields['CR']), p.fields['SC']]
            data.append(vector)
        self.data = pd.DataFrame(data)
        self.count = len(self.data)



class DataLoader_torch:
    def __init__(self, papers):
        pass


if __name__ == '__main__':
    path = r"E:\data\NSP"
    # papers_iter = papers_yield(path)
    papers_str = papers_load(path)
    papers = papers_transform(papers_str)
    