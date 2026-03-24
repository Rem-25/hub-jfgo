import json
import random
import torch
import jieba
from collections import defaultdict
from torch.utils.data import DataLoader


class DataGenerator:
    def __init__(self, data_path, config):
        self.data_path = data_path
        self.config = config
        self.vocab = load_vocab(config['vocab_path'])
        self.schema = load_schema(config['schema_path']) #向量数据库
        self.train_data_size = config['train_data_size']
        self.data_type = None
        self.load()

    def load(self):
        self.data = []
        self.knwb = defaultdict(list)
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = json.loads(line)
                if isinstance(line, dict):
                    self.data_type = 'train'
                    questions = line['questions']
                    label = line['target']
                    for question in questions:
                        input_id = self.encode_sentence(question)
                        input_id = torch.LongTensor(input_id)
                        self.knwb[self.schema[label]].append(input_id)
                else:
                    self.data_type = 'test'
                    assert isinstance(line, list)
                    question, label = line
                    input_id = self.encode_sentence(question)
                    input_id = torch.LongTensor(input_id)
                    label_index = torch.LongTensor(self.schema[label])
                    self.data.append((input_id, label_index))
        return

    def encode_sentence(self, text):
        input_id = []
        if self.config['vocab_path'] == 'words.txt':
            for word in jieba.cut(text):
                input_id.append(self.vocab.get(word, self.vocab['[UNK]']))
        else:
            for char in text:
                input_id.append(self.vocab.get(char, self.vocab['[UNK]']))
        input_id = self.padding(input_id)
        return input_id

    def padding(self, input_id):
        input_id = input_id[:self.config['max_length']]
        input_id += [0] * (self.config['max_length'] - len(input_id))
        return input_id

    def __len__(self):
        if self.data_type == 'train':
            return self.config['epoch_data_size']
        else:
            assert self.data_type == 'test', self.data_type #如果不是test，抛出现在的数据类型
            return len(self.data)

    def __getitem__(self, index):
        if self.data_type == 'train':
            return self.random_train_sample()
        else:
            return self.data[index]

    def random_train_sample(self):
        standard_q_index = list(self.knwb.keys())
        p, n = random.sample(standard_q_index, 2)
        if len(self.knwb[p]) == 1:
            s1 = s2 = self.knwb[p][0]
        else:
            s1, s2 = random.sample(self.knwb[p], 2)
        s3 = random.choice(self.knwb[n])
        return [s1, s2, s3]

def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1
    return token_dict

def load_schema(schema_path):
    with open(schema_path,encoding='utf-8') as f:
        return json.loads(f.read())

def load_data(data_path, config,shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size = config['batch_size'],shuffle=shuffle)
    return dl

if __name__ == '__main__':
    from config import Config
    dg = DataGenerator("valid.json", Config)
    print(dg[1])
