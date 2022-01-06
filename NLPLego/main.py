import argparse
import torch
import numpy as np
from Utils import LabelCount
from Model import CAGBert
from EnhanceModule import KnowledgeEnhance
from Utils import DataProcess
from Utils import Optimization
from Module import TrainModule
from Module import EvalModule

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model",
                        default=None,
                        type=str,
                        required=True,
                        help="pre-trained model name")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--seed",
                        default=7,
                        type=int,
                        required=True,
                        help="Seed for initial.")

    ## Other parameters
    parser.add_argument("--kg_dir",
                        default=None,
                        type=str,
                        help="If you want to use CAG, you have to list the location of the KG.")
    parser.add_argument("--max_seqlen",
                        default=256,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--batch_size",
                        default=32,
                        type=int,
                        help="Total batch size.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--epochs",
                        default=5,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--childTuningF',
                        default=True,
                        help="To Enable the Module ChildTuningF Algorithm.")

    args = parser.parse_args()

    #初始化随机变量，使其稳定可复现
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    #统计测试集的label数量
    modelArgs = {}
    modelArgs["label_size"] = LabelCount.LabelsCount(args.data_dir)

    #初始化模型
    model = CAGBert(modelArgs)

    #初始化知识注入器
    if args.kg_dir != None:
        CAGInjector = KnowledgeEnhance.KnowledgeEnhance(DataPath = args.kg_dir, MaxEntitiesSelect = 2)

    #初始化数据集
    TrainSet = DataProcess.DataProcess(max_len = args.max_seqlen, DataPath = args.data_dir + "/train.tsv", CAG = None).DataSetPrepare()
    DevSet = DataProcess.DataProcess(max_len = args.max_seqlen, DataPath = args.data_dir + "/dev.tsv", CAG = None).DataSetPrepare()
    TestSet = DataProcess.DataProcess(max_len = args.max_seqlen, DataPath = args.data_dir + "/test.tsv", CAG = None).DataSetPrepare()

    #优化器
    optimizer = Optimization.init_bert_adam_optimizer(model,TrainSet.__len__(),args.epochs,args.batch_size,args.gradient_accumulation_steps,args.learning_rate, args.warmup_proportion, args.childTuningF)

    #GPU配置
    gpu_ids = torch.cuda.device_count()
    use_cuda = torch.cuda.is_available()
    if gpu_ids == 1 and use_cuda:  # 一个gpu
        master_gpu_id = 0  # 主master
        model = model.cuda(0) if use_cuda else model
    else:  
        # 不使用gpu
        master_gpu_id = None

    #训练模型
    TrainModule.trains(master_gpu_id=master_gpu_id, model=model, epochs=args.epochs,
        optimizer=optimizer, train_data=TrainSet, dev_set=DevSet, test_data=TestSet,
        batch_size=args.batch_size, gradient_accumulation_steps=args.gradient_accumulation_steps,
        use_cuda=use_cuda, num_workers=1)

if __name__ == '__main__':
    main()
