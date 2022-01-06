import tqdm
import torch
import torch.utils.data as data
import logging
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score

def evaluates_pr(labels, predictions, modelArgs):
    label=[]
    pred=[]
    for lb, pd in zip(labels, predictions):
        label.append(lb.cpu().numpy().tolist())
        pred.append(pd.cpu().numpy().tolist())
    if modelArgs['label_size'] > 2:
        precise = precision_score(label, pred, average='weighted')
        recall = recall_score(label, pred, average='weighted')
    else:
        precise = precision_score(label, pred)
        recall = recall_score(label, pred)

    return precise, recall


def evaluates(master_gpu_id, model, test_data, batch_size = 1,use_cuda=False, num_workers=4, modelArgs=None):
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
    precision, recall = evaluates_pr(labels, predictions, modelArgs)
    F1 = 2 * precision * recall / (precision + recall)
    logging.info("precision: " + str(precision))
    logging.info("recall: " + str(recall))
    logging.info('F1-Score: ' + str(F1))