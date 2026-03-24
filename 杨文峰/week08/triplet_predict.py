import jieba
import torch
from loader import load_data
from config import Config
from model import SiameseNetwork, choose_optimizer

from review.week04 import sentence


class Predictor:
    def __init__(self, config, model, knwb_data):
        self.config = config
        self.model = model
        self.train_data = knwb_data
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        else:
            self.model = self.model.cpu()
        self.model.eval()
        self.knwb_to_vector()

    def knwb_to_vector(self):
        #初始化映射关系和储存列表
        self.quetion_index_to_standard_question_index = {} #存储 问题的序号 对应 上标准问题的序号的字典
        self.question_ids = [] #存储问题向量的列表
        self.standard_question_ids = []

        #获取训练集数据的词汇表和模式定义
        self.vocab = self.train_data.dataset.vocab
        self.schema = self.train_data.dataset.schema

        #从向量数据库中选取，创建索引到标准问题的映射字典
        self.index_to_standard_question = dict((y,x) for x,y in self.schema.items())
        # 从知识库中的序列号和问题中，建立问题索引到标准问题
        for standard_question_index, question_ids in self.train_data.dataset.knwb.items():
            for question_id in question_ids:
                self.quetion_index_to_standard_question_index[len(self.question_ids)] = standard_question_index
                self.question_ids.append(question_id)
        with torch.no_grad():
            question_matrix = torch.stack(self.question_ids, dim=0)
            if torch.cuda.is_available():
                question_matrix = question_matrix.cuda()
            self.knwb_vectors = self.model(question_matrix)
            self.knwb_vectors = torch.nn.functional.normalize(self.knwb_vectors,dim = -1)
        return

    def encode_sentence(self, text):
        input_id =[]
        if self.config["vocab_path"] == 'word.txt':
            for word in jieba.cut(text):
                input_id.append(self.vocab.get(word, self.vocab['[UNK]']))
            else:
                for char in text:
                    input_id.append(self.vocab.get(char, self.vocab['[UNK]']))
        return input_id

    def predict(self,sentence):
        input_id = self.encode_sentence(sentence)
        input_id = torch.LongTensor([input_id])
        if torch.cuda.is_available():
            input_id = input_id.cuda()
        with torch.no_grad():
            test_question_vector =  self.model(input_id)
            res = torch.mm(test_question_vector.unsqueeze(0), self.knwb_vectors.T) #1*vector_size x vector_size*n,代表问题矩阵加维度 x 知识库所有问题的矩阵，得到1 * n的概率
            hit_index =int(torch.argmax(res.squeeze()))
            hit_index = self.quetion_index_to_standard_question_index[hit_index]
        return self.index_to_standard_question[hit_index]

if __name__ == '__main__':
    knwb_data = load_data(Config["train_data_path"],Config)
    model = SiameseNetwork(Config)
    model.load_state_dict(torch.load(Config["model_path"]))
    pd = Predictor(Config, model, knwb_data)
    sentence = '固定宽带密码修改'
    res = pd.predict(sentence)
    print(res)
