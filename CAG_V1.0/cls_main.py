from utils import Prepare_DataSet
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
import time
import logging
import torch.utils.data as data
import os
import collections
import tqdm
import numpy as np
from importlib import import_module
from models.GenBert import GenBert
# from transformers import AdamW
from utils import optimization
# import random
from sklearn.metrics import precision_score, recall_score
from kg import Kg_Inject as KGI
import time

# 设置随机数 使实现可复现
seed = 7
#seed = random.randint(1,9999)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
# 指定根日志记录器级别
logging.basicConfig(level=logging.INFO,format="%(asctime)s - %(levelname)s - %(message)s")  # 使用指定字符串格式
args={}

ResultSet=[]

def evaluates_pr(labels, predictions):
    label=[]
    pred=[]
    for lb, pd in zip(labels, predictions):
        label.append(lb.cpu().numpy().tolist())
        pred.append(pd.cpu().numpy().tolist())
    if args['label_size'] > 2:
        precise = precision_score(label, pred, average='weighted')
        recall = recall_score(label, pred, average='weighted')
    else:
        precise = precision_score(label, pred)
        recall = recall_score(label, pred)

    return precise, recall


def evaluates(master_gpu_id, model, test_data, batch_size = 1,use_cuda=False, num_workers=4):
    model.eval()  # 测试模式
    test_data_loader = data.DataLoader(dataset=test_data,
                                       pin_memory=use_cuda,
                                       batch_size=batch_size,
                                       num_workers=num_workers,
                                       shuffle=False)
    total_loss = 0.0
    correct_sum = 0
    process_num = 0
    infos = []  # 存储预测的index 和 label
    num_batch = test_data_loader.__len__()

    for step, batch in enumerate(tqdm.tqdm(test_data_loader, unit="batch", ncols=100, desc="Evaluating process: ")):

        tokens = batch["tokens"].cuda(master_gpu_id) if use_cuda else batch["tokens"]  # 获取token
        token_type_ids = batch["token_type_ids"].cuda(master_gpu_id) if use_cuda else batch[
            "token_type_ids"]  # 获取token_type_ids
        attention_mask = batch["attention_mask"].cuda(master_gpu_id) if use_cuda else batch[
            "attention_mask"]  # 获取attention_mask

        labels = batch['labels'].cuda(master_gpu_id) if use_cuda else batch['labels']

        with torch.no_grad():
            logits = model(tokens, attention_mask)
            loss = F.cross_entropy(logits,labels)

        loss = loss.mean()
        loss_val = loss.item()  # 取值
        total_loss += loss_val  # 统计总的损失值

        # 返回batch个样本的最大值的[key,index]
        _, top_index = logits.topk(1)  # 取最大值 即预测概率最大的那一类

        # 统计batch个label相等的个数并求和
        correct_sum += (top_index.view(-1) == labels).sum().item()  # 进行预测是否正确的判断
        process_num += labels.shape[0]  # 统计总的样本数

        # 存储预测和真实标签
        for label, prediction in zip(labels, top_index.view(-1)):
            infos.append((prediction, label))
            
    acc = correct_sum / process_num
    logging.info('eval total avg loss:%s', format(total_loss / num_batch, "0.4f"))  # 验证结束 打印结果
    logging.info("Correct Prediction: " + str(correct_sum))
    logging.info("Accuracy Rate: " + format(acc, "0.4f"))
    # 计算精确率和召回率
    labels = [info[0] for info in infos]
    predictions = [info[1] for info in infos]
    precision, recall = evaluates_pr(labels, predictions)
    F1 = 2 * precision * recall / (precision + recall)
    logging.info("precision: " + str(precision))
    logging.info("recall: " + str(recall))
    logging.info('F1-Score: ' + str(F1))
    ResultSet.append([acc,precision,recall,F1])


def train_epoch(master_gpu_id, model, optimizer, data_loader, gradient_accumulation_steps, use_cuda):
    model.train()  # 训练模式
    data_loader.dataset.is_training = True
    total_loss = 0.0  # 总共的损失
    correct_sum = 0  # 正确预测总和
    process_sum = 0  # 处理数据的总和
    num_batch = data_loader.__len__()
    num_sample = data_loader.dataset.__len__()

    p_bar = tqdm.tqdm(data_loader, unit="batch", ncols=100)  # 进度条封装
    p_bar.set_description('train step loss')

    for step, batch in enumerate(p_bar):
        # model.zero_grad()  # 梯度归零
        
        tokens = batch["tokens"].cuda(master_gpu_id) if use_cuda else batch["tokens"]  # 获取token

        token_type_ids = batch["token_type_ids"].cuda(master_gpu_id) if use_cuda else batch["token_type_ids"]  # 获取token_type_ids

        attention_mask = batch["attention_mask"].cuda(master_gpu_id) if use_cuda else batch["attention_mask"]  # 获取attention_mask

        labels = batch['labels'].cuda(master_gpu_id) if use_cuda else batch['labels']

        logits = model(tokens, attention_mask)
        # print(tokens,logits)
        loss = F.cross_entropy(logits,labels)
        # 取平均

        loss = loss.mean()
        if gradient_accumulation_steps > 1:  # 梯度累积次数 其实意思就是去多次的平均值
            # 如果gradient_accumulation_steps == 1的话 代表不累计梯度 即不与前面的mini-batch相关
            loss /= gradient_accumulation_steps

        loss.backward()

        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()  # 梯度下降
            model.zero_grad()  # 梯度归零

        loss_val = loss.item()
        total_loss += loss_val

        # 统计
        _, top_index = logits.topk(1)
        correct_sum += (top_index.view(-1) == labels).sum().item()
        process_sum += labels.shape[0]

        p_bar.set_description('train step loss ' + format(loss_val, "0.4f"))
        
    acc_tmp=format(correct_sum / process_sum, "0.4f")
    logging.info("Total Training Samples:%s ", num_sample)
    logging.info('train total avg loss:%s', total_loss / num_batch)
    logging.info("Correct Prediction: " + str(correct_sum))
    logging.info("Accuracy Rate: " + acc_tmp)
    

    return total_loss / num_batch  # 返回平均每一个batch的损失值


def trains(master_gpu_id, model, epochs, optimizer, train_data, dev_set, test_data,batch_size, gradient_accumulation_steps=1, use_cuda=False, num_workers=4):
    logging.info("Start Training".center(60, "="))
    train_data_loader = data.DataLoader(dataset=train_data,
                                        pin_memory=use_cuda,
                                        batch_size=batch_size,
                                        num_workers=num_workers,
                                        shuffle=True)  # 是否打乱
    for epoch in range(1, epochs + 1):

        if epoch == 1:
            logging.info("Training Epoch: " + str(epoch))
        else:
            logging.info("=".center(59, "="))
            logging.info("Training Epoch: " + str(epoch))

        tmp_start = time.time()   
        avg_loss = train_epoch(master_gpu_id, model, optimizer, train_data_loader,gradient_accumulation_steps, use_cuda)
        tmp_end = time.time()
        logging.info("This epoch train cost time:"+str(tmp_end-tmp_start))
        logging.info("Average Loss: " + format(avg_loss, "0.4f"))

        logging.info("Evaluating Model in Dev set".center(60, "="))
        evaluates(master_gpu_id, model, dev_set, batch_size, use_cuda, num_workers)

        logging.info("Evaluating Model in Test set".center(60, "="))
        evaluates(master_gpu_id, model, test_data, batch_size, use_cuda, num_workers)



def LabelsCount(filepath):
    labels_set = set()
    Index = {}
    with open(filepath, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            try:
                line = line.strip().split("\t")
                if line_id == 0:
                    for k,v in enumerate(line):
                        Index[v]=k
                    continue
                label = int(line[Index['label']])
                labels_set.add(label)
            except:
                pass
    label_size = len(labels_set)
    return label_size

def main():
    train_path = "./datasets/lcqmc/train.tsv"
    dev_path = "./datasets/lcqmc/dev.tsv"
    test_path = "./datasets/lcqmc/test.tsv"
    kg_path = "./datasets/CnDbpedia.spo"

    args['label_size'] = LabelsCount(train_path)

    # Injector = KGI.KnowledgeInject(KGPath = kg_path, MaxEntitiesSelect = 2)

    #初始化模型
    logging.info("=".center(59, "="))
    model = GenBert(args)
    logging.info("Initialize Model Done".center(60, "="))
    max_len = 256

  
    # 初始化数据集 训练集和测试集
    logging.info("Load Training Dataset:%s", train_path)
    train_data = Prepare_DataSet.Prepare_DataSet(max_len=max_len, DataPath=train_path, KGI=None).DataSetPrepare()
    logging.info("Load Training Dataset Done, Total training line %s", train_data.__len__())

    dev_data = Prepare_DataSet.Prepare_DataSet(max_len=max_len, DataPath=dev_path, KGI=None).DataSetPrepare()
    logging.info("Load Dev Dataset Done, Total test line: %s", dev_data.__len__())

    test_data = Prepare_DataSet.Prepare_DataSet(max_len=max_len, DataPath=test_path, KGI=None).DataSetPrepare()
    logging.info("Load Test Dataset Done, Total test line: %s", test_data.__len__())
    

    # 初始化优化器
    # 超参数设置
    training_data_len = train_data.__len__()
    epochs = 5
    batch_size = 64
    gradient_accumulation_steps = 1 
    #5e-5, 3e-5, 2e-5
    init_lr = 5e-5
    warm_up_proportion = 0.1

    # optimizer = AdamW()
    optimizer = optimization.init_bert_adam_optimizer(model, training_data_len, epochs, batch_size, gradient_accumulation_steps, init_lr, warm_up_proportion)

    # 设置GPU
    gpu_ids = torch.cuda.device_count()
    use_cuda = torch.cuda.is_available()
    if gpu_ids == 1 and use_cuda:  # 一个gpu
        master_gpu_id = 0  # 主master
        model = model.cuda(0) if use_cuda else model
    else:  # 不使用gpu
        master_gpu_id = None
        

    # 开始训练
    trains(master_gpu_id=master_gpu_id, model=model, epochs=epochs,
        optimizer=optimizer, train_data=train_data, dev_set=dev_data, test_data=test_data,
        batch_size=batch_size, gradient_accumulation_steps=gradient_accumulation_steps,
        use_cuda=use_cuda, num_workers=1)

if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    logging.info("Total Cost time:"+str(end - start))
    logging.info("Seed:"+str(seed))