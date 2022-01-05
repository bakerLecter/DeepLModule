import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import *
import logging
import numpy as np
 
# model_name = 'bert-base-chinese'  # 指定需下载的预训练模型参数
model_name = 'nghuyong/ernie-1.0'
# model_name = 'nghuyong/ernie-2.0-en'
# model_name = 'roberta-base'

class CAGBert(nn.Module):
    def __init__(self, args):
        super(CAGBert, self).__init__()
        logging.info("Load Model:" + str(model_name))
        self.model = AutoModel.from_pretrained(model_name, cache_dir="./bert/")
        self.dropout = nn.Dropout(0.1)
        # self.Layer_0 = nn.Linear(768, 768)
        self.Layer_1 = nn.Linear(768, int(args['label_size']))


    def forward(self, data, attention_mask=None):
        outputs = self.model(data, attention_mask=attention_mask)

        #MLP layer
        # output = torch.tanh(self.Layer_0(outputs[1]))

        # logits = self.Layer_1(output)

        logits = self.Layer_1(outputs[1])

        return logits
