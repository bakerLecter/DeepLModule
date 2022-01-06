# coding: utf-8
import os
import pkuseg
import numpy as np
import logging
# import tqdm

FLOPs = 0

class KnowledgeEnhance(object):
    def __init__(self, KGPath = None, MaxEntitiesSelect = 2):
        logging.info("Loading KG Injector".center(60, "="))
        self.DataPath = KGPath
        self.MaxEntitiesSelect = MaxEntitiesSelect
        self.kg = self.LoadKG()
        self.seg = pkuseg.pkuseg()

    def LoadKG(self):
        kg = {}
        with open(self.DataPath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    triple = line.strip("\n").split("\t") 
                except:
                    logging.WARNING("Bad KG Data Line:", line)

                obj = triple[0]
                value = ''.join(triple[1:])

                if obj in kg.keys():
                    kg[obj].add(value)
                else:
                    kg[obj] = set([value])
        logging.info("Load Knowledge Graph Done, KG size: %s",len(kg))
        return kg

    def KnowledgeInjector(self, Setence, Label):
        global FLOPs

        KGEnhance = []
        token = self.seg.cut(Setence)
        #Version 1.0 KGI
        #截取后续的文字
        SentenceGenerator = ''
        Temp = ''
        for word in token:
            SentenceGenerator += word
            # 1 Add
            FLOPs += 1
            ent = list(self.kg.get(word,[]))[:self.MaxEntitiesSelect]
            if len(ent) != 0:
                for k in ent:
                    Temp = [SentenceGenerator, k,'\t',str(Label)]
                    # 3 Add
                    FLOPs += 3
                    KGEnhance.append("".join(Temp))

        #Version 1.1 KGI
        #保留整段文本，仅作关键词修改
        
        KGEnhance.append(Setence + '\t' + str(Label))
        #3 Add
        FLOPs += 3
        return KGEnhance

    def DataKnowledgeInjector(self, RawData, IndexLen):
        global FLOPs
        Processed = []
        for words in RawData:
            #分离句子与标签
            word = words.strip("\n").split("\t")
            sentence = word[:-1]
            label = word[-1]
            #若是QA
            if len(word) > 2:
                tmp = ''
                for key in sentence:
                    tmp = tmp + '[OnO]' + key
                    # 2 Add
                    FLOPs += 2
                    DataTmp =  self.KnowledgeInjector(tmp,label)
                for k in DataTmp:
                    t = k.replace('[OnO]','\t').lstrip('\t')
                    count = t.split("\t")
                    if len(count) < IndexLen:
                        count.insert(1,',,')
                    t = '\t'.join(count)
                    Processed.append(t)
            #若是分类
            elif len(word) == 2:
                Processed.extend(self.KnowledgeInjector(sentence[0],label))
        return Processed,FLOPs
