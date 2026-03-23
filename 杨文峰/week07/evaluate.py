import openpyxl
import torch
from loader import load_data

class Evaluator:
    def __init__(self, config,model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config['valid_data_path'],config,shuffle=False)
        self.sentences = self.valid_data.dataset.sentences
        self.stats_dict = {'correct':0,'wrong':0}

    def eval(self,epoch):
        self.logger.info('进行第%d轮模型测试：'% epoch)
        self.model.eval()
        self.stats_dict = {'correct':0,'wrong':0}

        self.writer = openpyxl.Workbook()
        self.sheet = self.writer.active
        self.sheet.append(["sentence", "true_label", "pred_label", "is_correct"])
        for index, batch_data in enumerate(self.valid_data):
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            input_ids, labels = batch_data
            input_ids = input_ids.squeeze()
            with torch.no_grad():
                pred_results = self.model(input_ids)
            self.write_stats(labels,pred_results, self.sentences[index * self.config['batch_size']:(index + 1) * self.config['batch_size']])
        acc = self.show_stats()
        self.writer.save('valid_result.xlsx')
        return acc

    def write_stats(self, labels, pred_results, sentences):
        assert len(labels) == len(pred_results)
        for true_label, pred_label, sentence in zip(labels, pred_results, sentences):
            pred_label = torch.argmax(pred_label)
            self.sheet.append([sentence, int(true_label), int(pred_label), int(true_label) == int(pred_label)])
            if int(true_label) == int(pred_label):
                self.stats_dict["correct"] += 1
            else:
                self.stats_dict["wrong"] += 1
        return

    def show_stats(self):
        correct = self.stats_dict['correct']
        wrong = self.stats_dict['wrong']
        self.logger.info('预测集合条目数:%d'%(correct + wrong))
        self.logger.info('预测正确条目数:%d,错误条目数:%d'%(correct, wrong))
        self.logger.info('预测正确率:%f'% (correct / (wrong + correct)))
        self.logger.info("_"*10 + '分割线' + '_'*10)
        return correct / (wrong + correct)
