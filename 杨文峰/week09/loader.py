import torch
import json
from torch.utils.data import DataLoader
from transformers import BertTokenizer

class DataGenerator:
    def __init__(self, data_path, config):
        self.data_path = data_path
        self.config = config
        self.tokenizer = load_vocab(config["bert_path"])
        self.sentences = []
        self.schema = self.load_schema(config["schema_path"])
        self.load()

    def load(self):
        self.data = []
        with open(self.data_path, "r", encoding="utf-8") as f:
            segments = f.read().split("\n\n")
            for segment in segments:
                sentence = []
                labels = []
                for line in segment.split("\n"):
                    if line.strip() == '':
                        continue
                    char, label = line.split()
                    sentence.append(char)
                    labels.append(self.schema[label])
                sentence = "".join(sentence)
                self.sentences.append(sentence)
                input_ids = self.encode_sentence(sentence)
                labels = self.padding(labels, -1)

                self.data.append(([torch.LongTensor(input_ids), torch.LongTensor(labels)]))
        return

    def encode_sentence(self, text, padding=True):
        input_ids = [self.tokenizer.vocab.get(char, self.tokenizer.vocab["[UNK]"]) for char in text]
        if padding:
            input_ids = self.padding(input_ids, self.tokenizer.vocab["[PAD]"])
        return input_ids

    def padding(self, input_id, pad_token = 0):
        input_id = input_id[:self.config["max_length"]]
        input_id += [pad_token] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def load_schema(self, schema_path):
        with open(schema_path, "r", encoding="utf-8") as f:
            return json.load(f)

def load_vocab(vocab_path):
    return  BertTokenizer.from_pretrained(vocab_path)

def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl

if __name__ == "__main__":
    from config import Config
    dg = DataGenerator("ner_data/train", Config)
    dl = DataLoader(dg, batch_size= 32)
    for x,y in dl:
        print(x.shape, y.shape)
        print(f"第三个数据“：{x[2]},\n数据标签：{y[2]}")
        input()

