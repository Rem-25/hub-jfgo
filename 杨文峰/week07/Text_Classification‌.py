"""
本次训练分为3部分
1.数据分析和预处理模块 loader.py
2.BERT模型训练模块 model.py
3.主程序入口 main.py
"""

#part 1
import pandas as pd
import numpy as np
from transformers import BertTokenizer
import time
from datetime import datetime

class DataAnalyzer:
    def __init__(self,csv_path):
        self.csv_path = csv_path
        self.df = None
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    def load_data(self):
        self.df = pd.read_csv(self.csv_path)
        return self.df

    def analyze_dataset(self):
        if self.df is None:
            self.df = self.load_data()

        positive_count = (self.df['label'] == 1).sum()
        negative_count = (self.df['label'] == 0).sum()
        total_samples = len(self.df)

        text_length = self.df['review'].apply(len)
        avg_length = text_length.mean()

        print('== 数据分析结果 ==')
        print(f"正样本数（好评）: {positive_count}")
        print(f"负样本数（差评）: {negative_count}")
        print(f"总样本数: {total_samples}")
        print(f"文本平均长度: {avg_length:.2f} 字符")

        return {
            'positive_count': positive_count,
            'negative_count': negative_count,
            'total_samples': total_samples,
            'avg_length': avg_length
        }

#part 2
import torch
import torch.nn as nn
from transformers import BertModel, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader,Dataset
from sklearn.metrics import accuracy_score

class ReviewDataset(Dataset):
    def __init__(self, reviews, labels, tokenizer,max_len = 128):
        self.reviews = reviews
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        review = str(self.reviews[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
        review,
        add_special_tokens=True,
        max_length=self.max_len,
        padding='max_length',
        truncation=True,
        return_attention_mask= True,
        return_tensors= 'pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class BERTClassifier(nn.Module):
    def __init__(self, n_classes=2):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.pooler_output
        output = self.drop(pooled_output)
        return self.out(output)


class ModelTrainer:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.model = None
        self.train_loader = None
        self.val_loader = None

    def prepare_data(self, train_ratio=0.8, batch_size=16):
        df = pd.read_csv(self.csv_path)

        train_size = int(len(df) * train_ratio)
        train_df = df[:train_size]
        val_df = df[train_size:]

        train_dataset = ReviewDataset(
            reviews=train_df['review'].values,
            labels=train_df['label'].values,
            tokenizer=self.tokenizer
        )

        val_dataset = ReviewDataset(
            reviews=val_df['review'].values,
            labels=val_df['label'].values,
            tokenizer=self.tokenizer
        )

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        return train_dataset, val_dataset

    def train_model(self, epochs=3):
        self.model = BERTClassifier().to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=2e-5)
        total_steps = len(self.train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )

        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            print(f'Epoch {epoch + 1}/{epochs}')
            self.model.train()
            total_loss = 0

            for batch in self.train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                loss.backward()
                optimizer.step()
                scheduler.step()

            avg_loss = total_loss / len(self.train_loader)
            print(f'Training loss: {avg_loss:.4f}')

            accuracy = self.evaluate_model()
            print(f'Validation Accuracy: {accuracy:.4f}')

    def evaluate_model(self):
        self.model.eval()
        predictions = []
        actual_labels = []

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids, attention_mask)
                _, preds = torch.max(outputs, dim=1)

                predictions.extend(preds.cpu().tolist())
                actual_labels.extend(labels.cpu().tolist())

        accuracy = accuracy_score(actual_labels, predictions)
        return accuracy

#part 3
import torch
import time
from transformers import BertTokenizer


def main():
    csv_path = r"G:\AI\每周的课\11.30第七周 文本分类问题\文本分类练习.csv"

    print("开始BERT情感分析项目...")

    # 数据分析
    analyzer = DataAnalyzer(csv_path)
    stats = analyzer.analyze_dataset()

    # 模型训练和验证
    print("\n开始模型训练...")
    trainer = ModelTrainer(csv_path)
    trainer.prepare_data()
    trainer.train_model(epochs=3)

    # 计算准确率
    accuracy = trainer.evaluate_model()
    print(f"\n模型验证准确率: {accuracy:.4f}")

    # 预测100条文本耗时测试
    print("\n开始性能测试...")
    test_reviews = analyzer.df['review'].sample(100, random_state=42).tolist()

    start_time = time.time()

    predictions = []
    for review in test_reviews:
        encoding = trainer.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        input_ids = encoding['input_ids'].to(trainer.device)
        attention_mask = encoding['attention_mask'].to(trainer.device)

        with torch.no_grad():
            outputs = trainer.model(input_ids, attention_mask)
            _, pred = torch.max(outputs, dim=1)
            predictions.append(pred.cpu().item())

    end_time = time.time()
    prediction_time = end_time - start_time

    print("\n=== 最终结果汇总 ===")
    print(f"正样本数: {stats['positive_count']}")
    print(f"负样本数: {stats['negative_count']}")
    print(f"文本平均长度: {stats['avg_length']:.2f} 字符")
    print(f"预测准确率: {accuracy:.4f}")
    print(f"预测100条文本耗时: {prediction_time:.2f} 秒")
    print(f"平均每条预测时间: {prediction_time / 100:.4f} 秒")

    return {
        'positive_count': stats['positive_count'],
        'negative_count': stats['negative_count'],
        'avg_length': stats['avg_length'],
        'accuracy': accuracy,
        'prediction_time_100': prediction_time
    }


if __name__ == "__main__":
    main()
