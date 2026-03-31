import torch
import torch.nn as nn
import numpy as np
import random
import os
from transformers import BertTokenizer, BertModel


class LanguageModel(nn.Module):
    def __init__(self, hidden_size, vocab_size, pretrain_model_path):
        super(LanguageModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrain_model_path, return_dict=False, attn_implementation='eager') #本来的字典输出改成元组tuple输出，注意力计算方式用标准模式，考虑兼容性
        self.classify = nn.Linear(hidden_size, vocab_size)
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y=None):
        if y is not None:
            mask = torch.tril(torch.ones((x.shape[0], x.shape[1], x.shape[1]))) #下三角矩阵，形状为batch_size * seq_len * seq_len,因为自注意力权重矩阵是每个位置对其他位置的所有注意力关系，所以有两个维度是seq_len
            # print(mask, mask.shape)
            if torch.cuda.is_available():
                mask = mask.cuda()
            x,_ = self.bert(x, attention_mask = mask)
            y_pred = self.classify(x)
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            x,_ = self.bert(x)
            y_pred = self.classify(x)
            return torch.softmax(y_pred, dim= -1)

def load_corpus(corpus_path):
    corpus = ""
    with open(corpus_path, "r", encoding="gbk") as f:
        for line in f:
            corpus += line.strip()
    return corpus

def build_sample(tokenizer, window_size, corpus):
    start = random.randint(0, len(corpus) - window_size - 1)
    end = start + window_size
    window = corpus[start:end]
    target = corpus[start + 1: end + 1] #每一个字对应前面输入的每一个字

    x = tokenizer.encode(window, add_special_tokens=False, padding = 'max_length', truncation=True, max_length = 10) #不要CLS和SEP
    y = tokenizer.encode(target, add_special_tokens=False, padding = 'max_length', truncation=True, max_length = 10)

    return x, y

def build_dataset(sample_length, tokenizer,window_size,corpus):
    dataset_x, dataset_y = [], []
    for i in range(sample_length):
        x, y = build_sample(tokenizer, window_size, corpus)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

def build_model(vocab,char_dim, pretrain_model_path):
    model = LanguageModel(768, 21128, pretrain_model_path)
    return model

# 根据窗口自动生成文本，生成文本需要小于30
def generate_sentence(openings, model, tokenizer,window_size):
    model.eval()
    with torch.no_grad():
        pred_char = ""
        while pred_char != "\n" and len(openings) <= 30: #如果生成了换行符说明文本终结，超过30个字也该停止
            openings += pred_char
            x = tokenizer.encode(openings, add_special_tokens=False)
            x = torch.LongTensor([x])
            if torch.cuda.is_available():
                x = x.cuda()
            y = model(x)[0][-1] #取元组的hidden_states的最后一个token表示
            index = sampling_strategy(y) #调用采样策略，不一定取概率最大的字预测
            pred_char = ''.join(tokenizer.decode(index))
    return openings

def sampling_strategy(prob_distribution):
    if random.random() > 0.1:
        strategy = "greedy"
    else:
        strategy = "sampling"
    if strategy == "greedy":
        return int(torch.argmax(prob_distribution))
    elif strategy == "sampling":
        prob_distribution = prob_distribution.cpu().numpy()
        return np.random.choice(list(range(len(prob_distribution))), p=prob_distribution)

def train(corpus_path, save_weight = True):
    epoch_num = 20
    batch_size = 128
    train_sample = 10000
    char_dim = 768
    window_size = 10
    vocab_size = 21128
    learning_rate = 0.001

    pretrain_model_path = r"G:\AI\programs\review\bert"
    tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)
    corpus = load_corpus(corpus_path)
    model = build_model(vocab_size, char_dim, pretrain_model_path)
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, tokenizer, window_size, corpus)
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        print("epoch:", epoch + 1, "loss:", np.mean(watch_loss))
        print(generate_sentence("让他在半年之前，就不能做出",model, tokenizer, window_size))
    if not save_weight:
        base_name = os.path.basename(corpus_path).replace("txt","pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return

if __name__ == "__main__":
    train(r"G:\AI\programs\review\week10\corpus.txt", False)

