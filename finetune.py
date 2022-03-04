from matplotlib.pyplot import get
import torch
from transformers import AutoTokenizer,BertForSequenceClassification,AdamW
import json
import os 
import torch.utils.data as Data
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = '6'

def get_examples(path,set_type):
    file_path = os.path.join(path,set_type)
    texts=[]
    labels=[]
    with open(file_path) as f:
        load_data = json.load(f)

        for info in load_data.values():
            label = info['polarity']
            
            text_a = info['sentence']
            texts.append(text_a)
            labels.append(0 if label=='-1' else 1)

    return texts,labels

data_dir = '/home/hanzhang/PyContinual-main/src/dat/dsc/Amazon_Instant_Video'

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
tokenizer = AutoTokenizer.from_pretrained('/home/hanzhang/bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('/home/hanzhang/bert-base-uncased',num_labels = 2).to(device)
optim = AdamW(model.parameters(), lr=5e-5)


train_texts,train_labels = get_examples(path = data_dir,set_type="train.json")
dev_texts,dev_labels = get_examples(path = data_dir,set_type="dev.json")
test_texts,test_labels = get_examples(path = data_dir,set_type="test.json")
# print(train_labels)
def make_data(sentences,labels):
    input_ids,token_type_ids,attention_mask=[],[],[]
    #input_ids是每个词对应的索引idx ;token_type_ids是对应的0和1，标识是第几个句子；attention_mask是对句子长度做pad
    #input_ids=[22,21,...499] token_type_ids=[0,0,0,0,1,1,1,1] ;attention_mask=[1,1,1,1,1,0,0,0]补零
    for i in range(len(sentences)):
        encoded_dict = tokenizer.encode_plus(
        sentences[i],
        add_special_tokens = True,      # 添加 '[CLS]' 和 '[SEP]'
        max_length = 256,           # 填充 & 截断长度
        pad_to_max_length = True,
        return_tensors = 'pt',         # 返回 pytorch tensors 格式的数据
        )
        input_ids.append(encoded_dict['input_ids'])
        token_type_ids.append(encoded_dict['token_type_ids'])
        attention_mask.append(encoded_dict['attention_mask'])

    # print(labels)
    input_ids = torch.cat(input_ids,dim=0)#每个词对应的索引
    token_type_ids = torch.cat(token_type_ids,dim=0)#0&1标识是哪个句子
    attention_mask = torch.cat(attention_mask,dim=0)#[11100]padding之后的句子
    labels = torch.LongTensor(labels)#所有实例的label对应的索引idx

    return input_ids, token_type_ids, attention_mask, labels

train_inputs, train_token, train_mask, train_labels=make_data(train_texts,train_labels)
train_data = Data.TensorDataset(train_inputs,train_mask, train_labels)
train_dataloader = Data.DataLoader(train_data, batch_size=64, shuffle=True)

dev_inputs, dev_token, dev_mask, dev_labels=make_data(dev_texts,dev_labels)
dev_data = Data.TensorDataset(dev_inputs,  dev_mask, dev_labels)
dev_dataloader = Data.DataLoader(dev_data, batch_size=64, shuffle=True)

test_inputs, test_token, test_mask, test_labels=make_data(test_texts,test_labels)
test_data = Data.TensorDataset(test_inputs,  test_mask, test_labels)
test_dataloader = Data.DataLoader(test_data, batch_size=64, shuffle=True)

#accuracy计算
def flat_accuracy(preds,labels):
 
    pred_flat=np.argmax(preds,axis=1).flatten()
    labels_flat=labels.flatten()
    return np.sum(pred_flat==labels_flat)/len(labels_flat)



def eval(model, val_loader):
  model.eval()
  eval_loss, eval_accuracy, nb_eval_steps = 0, 0, 0
  for batch in val_loader:
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            logits = model(batch[0], attention_mask=batch[1])[0]#注意跟下面的参数区别，这个地方model.eval()指定是测试，所以没有添加label 
            logits = logits.detach().cpu().numpy()
            label_ids = batch[2].cpu().numpy()
            tmp_eval_accuracy = flat_accuracy(logits, label_ids)
            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1
            
  print("Validation Accuracy: {}".format(eval_accuracy / nb_eval_steps))
#   global best_score
#   if best_score < eval_accuracy / nb_eval_steps:
#       best_score = eval_accuracy / nb_eval_steps
#       save(model)

def test(model, test_loader):
  model.eval()
  eval_loss, eval_accuracy, nb_eval_steps = 0, 0, 0
  for batch in test_loader:
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            logits = model(batch[0], attention_mask=batch[1])[0]#注意跟下面的参数区别，这个地方model.eval()指定是测试，所以没有添加label 
            logits = logits.detach().cpu().numpy()
            label_ids = batch[2].cpu().numpy()
            tmp_eval_accuracy = flat_accuracy(logits, label_ids)
            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1
            
  print("test Accuracy: {}".format(eval_accuracy / nb_eval_steps))

def save(model):
    # save
    torch.save(model.state_dict(), './res')
    model.config.to_json_file('./res')

epoch = 3
for epoch in range(epoch):
  for i,batch in enumerate(train_dataloader):
    batch = tuple(t.to(device) for t in batch)
    loss=model(input_ids=batch[0], attention_mask=batch[1], labels=batch[2])[0]
    print('epoch: ',epoch,'loss: ',loss.item())
    optim.zero_grad()
    loss.backward()
    optim.step()

    if i % 100 == 0:
        eval(model, dev_dataloader)

test(model, test_dataloader)