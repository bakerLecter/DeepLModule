import torch
import numpy as np
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
# from sklearn.metrics import confusion_matrix,precision_score,recall_score,accuracy_score

model_name = 'bert-base-chinese'  # 指定需下载的预训练模型参数

# # # BERT 在预训练中引入了 [CLS] 和 [SEP] 标记句子的开头和结尾
# samples = ['[CLS] 今天中午吃什么好呢？不如去吃火锅吧？ [SEP]  不想吃[MASK][MASK] [SEP]']  # 准备输入模型的语句
samples = ['今天中午吃什么好呢？不如去吃火锅吧？[SEP]今天中午吃什么好呢？']  # 准备输入模型的语句

# label=[0,2,1,1,2,0,0]
# pred=[0,2,0,1,1,0,1]

# print(precision_score(label,pred,average='weighted'))
# print(recall_score(label,pred,average='weighted'))
# print(accuracy_score(label,pred))
# print(confusion_matrix(label,pred))
tokenizer = BertTokenizer.from_pretrained(model_name)
tokenized_text = tokenizer(samples, return_tensors='pt', padding='max_length',truncation=True, max_length=50)
tmp = tokenizer.convert_ids_to_tokens(tokenized_text['input_ids'].numpy()[0])
# print(tmp)
# exit()
input_ids = [tokenizer.convert_tokens_to_ids(i) for i in tokenized_text]
# print(input_ids)
input_ids = torch.LongTensor(input_ids)
# # print(tokenized_text,input_ids)

# # # # 读取预训练模型
model = BertForSequenceClassification.from_pretrained(model_name, cache_dir="./bert/")
model.eval()
labels = torch.tensor([1]).unsqueeze(0)
loss, logits = model(input_ids,labels)

# sample = prediction_scores[0].detach().numpy()
# pred = np.argmax(sample, axis=1)
# tmp = tokenizer.convert_ids_to_tokens(pred)

print(loss, logits)
# print(tmp)

# m = torch.nn.ZeroPad2d()
# print(torch.cuda.device_count(),torch.cuda.is_available(),torch.cuda.get_device_name())
# exit()