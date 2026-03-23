import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer


class DataGenerator:
    def __init__(self, data_path, config):
        self.path = data_path
        self.config = config
        self.index_to_label = {0:'差评', 1:'好评'}
        self.label_to_index = dict((y,x) for x,y in self.index_to_label.items())
        self.config["class_num"] = len(self.index_to_label)
        if self.config['model_type'] == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained(config['pretrain_model_path'])
        self.vocab = load_vocab(config['vocab_path'])
        self.config["vocab_size"] = len(self.vocab)
        self.sentences = []
        self.load()

    def load(self):
        self.data = []
        with open(self.path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith("0,"):
                    label = 0
                elif line.startswith("1,"):
                    label = 1
                else:
                    continue

                title = line[2:].strip()
                if self.config["model_type"] == "bert":
                    input_id = self.tokenizer.encode(title, max_length=self.config['max_length'],padding='max_length',truncation=True,return_tensors='pt')

                else:
                    input_id = self.encode_sentence(title)
                self.sentences.append(title)
                input_id = torch.LongTensor(input_id)
                label_index = torch.LongTensor([label])
                self.data.append((input_id, label_index))

        return

    def encode_sentence(self, text):
        input_id = []
        for char in text:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        input_id = self.padding(input_id)
        return input_id

    def padding(self, input_ids):
        input_ids = input_ids[:self.config["max_length"]]
        input_ids += [0] * (self.config["max_length"] - len(input_ids))
        return input_ids

    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        return self.data[index]

def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding='utf-8') as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1
        return token_dict

def load_data(data_path, config, shuffle = True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size = config['batch_size'], shuffle = shuffle)
    return dl

if __name__ == '__main__':
    from config import Config
    dg = DataGenerator("文本分类练习.csv", Config)
    print(dg)
