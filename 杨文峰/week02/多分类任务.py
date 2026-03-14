import torch
import numpy as np
import torch.nn as nn

class MultiClassfication(nn.Module):
    def __init__(self, input_size):
        super(MultiClassfication, self).__init__()
        self.linear = nn.Linear(input_size,5)
        self.loss = nn.functional.cross_entropy

    def forward(self,x,y = None):
        y_pred = self.linear(x)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return torch.softmax(y_pred, axis = -1)

def build_sample():
    x = np.random.random(5)
    max_index = np.argmax(x)
    return x, max_index

def build_dataset(sample_num):
    X = []
    Y = []
    for i in range(sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)

def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        for y_p, y_t in zip(y_pred, y):
            if torch.argmax(y_p) == int(y_t):
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率: %f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)

def main():
    epoch_num = 20
    batch_size = 20
    train_sample_num = 3000
    input_size = 5
    learning_rate = 0.001

    model = MultiClassfication(input_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    train_x, train_y = build_dataset(train_sample_num)
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample_num // batch_size):
            x = train_x[batch_index * batch_size:(batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size:(batch_index + 1) * batch_size]
            loss = model(x, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            watch_loss.append(loss.item())
        print("====================\n第%d轮平均loss:%f" % (epoch+1, np.mean(watch_loss)))
        acc = evaluate(model)
        log.append([acc, float(np.mean(watch_loss))])
    torch.save(model.state_dict(), "model.pt")
    print(log)
    return

def predict(model_path , input_vector):
    input_size = 5
    model = MultiClassfication(input_size)
    model.load_state_dict(torch.load(model_path))
    print(model.state_dict())

    model.eval()
    with torch.no_grad():
        input_tensor = torch.FloatTensor(input_vector).float()  # 确保是Float类型
        result = model(input_tensor)
        for vec,res in zip(input_vector, result):
            print("输入：%s, 预测类别：%s,概率值：%s" % (vec,torch.argmax(res),res))

if __name__ == "__main__":
    main()
    test_vec = [[0.47889086, 0.15229675, 0.31082123, 0.03504317, 0.18920843],
                [0.4963533, 0.5524256, 0.95758807, 0.65520434, 0.84890681],
                [0.48797868, 0.67482528, 0.13625847, 0.34675372, 0.09871392],
                [0.49349776, 0.59416669, 0.92579291, 0.41567412, 0.7358894]]

    predict("model.pt", test_vec)
