# coding: utf-8
import numpy as np
import torch
from transformers import AutoTokenizer
# from transformers import RobertaTokenizer

# model_name = 'bert-base-chinese' #Bert中文预训练模型
model_name = 'nghuyong/ernie-1.0'
# model_name = 'nghuyong/ernie-2.0-en'
# model_name = 'roberta-base'

class DataSetLoader(torch.utils.data.Dataset):

    def __init__(self, max_len=512-2, Index=None, RawData=None):
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./bert/")  # 加载分词器
        # self.tokenizer = RobertaTokenizer.from_pretrained(model_name, cache_dir="./bert/")
        self.Index = Index
        # if DataPath != None:
        #     self.raw_data = open(DataPath, 'r').readlines()  # 按照行读取所有的数据行
        # self.tmp = self.raw_data[0].strip("\n").split("\t")
        # for k,v in enumerate(self.tmp):
        #     self.Index[v]=k
        self.raw_data = RawData

    def __len__(self):
        return len(self.raw_data)
        
    def __getitem__(self, item):
        line = self.raw_data[item]
        row = line.strip().split("\t")
        # print(self.Index)
        # print(row)
        # exit()
        labels = int(row[self.Index['label']])
        if len(row) < 2:
            raise RuntimeError("Data is illegal: " + line)
        # labels = torch.FloatTensor(int(row[0]))
        # sen_code = self.tokenizer(row[1])
        # exit()
        # tokens = self.tokenizer.convert_tokens_to_ids(sen_code)
        # if self.max_len:
        #     if len(sen_code) < self.max_len:
        #         tokens += ([0] * (self.max_len - len(sen_code)))
        #     else:
        #         tokens = tokens[:self.max_len]
        # tokens = torch.LongTensor(tokens)
        # return {"data":labels,"target":tokens}  # 返回

        # max_length:truncation/padding
        elif len(row) == 2:
            text = row[self.Index['text_a']]
            sen_code = self.tokenizer(text, return_tensors='pt', padding='max_length',truncation=True, max_length=self.max_len)
            tokens = torch.LongTensor(sen_code['input_ids'][0])
            token_type_ids = torch.LongTensor(sen_code['token_type_ids'][0])
            attention_mask = torch.LongTensor(sen_code['attention_mask'][0])
        elif len(row) == 3:
            text = row[self.Index['text_a']] + '[SEP]' + row[self.Index['text_b']]
            sen_code = self.tokenizer(text, return_tensors='pt', padding='max_length',truncation=True, max_length=self.max_len)
            tokens = torch.LongTensor(sen_code['input_ids'][0])
            token_type_ids = torch.LongTensor(sen_code['token_type_ids'][0])
            attention_mask = torch.LongTensor(sen_code['attention_mask'][0])
        else:
            text = row[self.Index['text_a']] + '[SEP]' + row[self.Index['text_b']]
            sen_code = self.tokenizer(text, return_tensors='pt', padding='max_length',truncation=True, max_length=self.max_len)
            tokens = torch.LongTensor(sen_code['input_ids'][0])
            token_type_ids = torch.LongTensor(sen_code['token_type_ids'][0])
            attention_mask = torch.LongTensor(sen_code['attention_mask'][0])

        dataset={"text": text, "tokens": tokens, "token_type_ids": token_type_ids,"attention_mask": attention_mask, "labels": labels}
        return dataset  # 返回
