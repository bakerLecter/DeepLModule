# coding: utf-8
from . import DataSetLoader
from EnhanceModule import KnowledgeEnhance
import logging
import time
#用于读取数据并进行知识注入
class DataProcess(object):
    def __init__(self, max_len=512, DataPath=None, CAG=None):
        self.max_len = max_len
        self.DataPath = DataPath
        self.CAG = CAG
        self.IndexLen = 0
        self.DataIndex,self.RawData = self.LoadRawDataFromFiles()
        if self.CAG != None:
            logging.info("Inject Knowledge to Dataset, origin Dataset size: %s",len(self.RawData))
            start = time.time()
            self.ProcessedData, TotalFLOPs = self.CAG.DataKnowledgeInjector(self.RawData, self.IndexLen)
            end = time.time()
            logging.info("Inject Knowledge cost: " + str(end - start))
            logging.info("Inject Knowledge FLOPs: " + str(TotalFLOPs))
            self.DSL = DataSetLoader.DataSetLoader(max_len = self.max_len, Index= self.DataIndex, RawData=self.ProcessedData)
        else:
            self.DSL = DataSetLoader.DataSetLoader(max_len = self.max_len, Index= self.DataIndex, RawData=self.RawData)

    def LoadRawDataFromFiles(self):
        if self.DataPath != None:
            raw_data = open(self.DataPath, 'r').readlines()  # 按照行读取所有的数据行
        Index={}
        tmp = raw_data[0].strip("\n").split("\t")
        for k,v in enumerate(tmp):
            Index[v]=k
        # print(Index)
        raw_data=raw_data[1:]
        self.IndexLen = len(Index)
        return Index, raw_data

    def DataSetPrepare(self):
        return self.DSL