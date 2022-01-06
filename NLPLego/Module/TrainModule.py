import tqdm
import logging
import torch.utils.data as data
import time
import torch.nn.functional as F
from . import EvalModule

logging.basicConfig(level=logging.INFO,format="%(asctime)s - %(levelname)s - %(message)s")  # 使用指定字符串格式

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


def trains(master_gpu_id, model, epochs, optimizer, train_data, dev_set, test_data,batch_size, gradient_accumulation_steps=1, use_cuda=False, num_workers=4, modelArgs=None):
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
        EvalModule.evaluates(master_gpu_id, model, dev_set, batch_size, use_cuda, num_workers, modelArgs)

        logging.info("Evaluating Model in Test set".center(60, "="))
        EvalModule.evaluates(master_gpu_id, model, test_data, batch_size, use_cuda, num_workers, modelArgs)