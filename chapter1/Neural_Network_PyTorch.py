import torch
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    # 使用するオブジェクトを定義
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(13, 5)
        self.fc2 = nn.Linear(5, 1)

    # 順伝播
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

    def loss_fn(self, y, t):
        return F.mse_loss(y, t)

def convert_categorical_to_numeric(data):
    for row in data:
        for i in range(len(row)):
            try:
                row[i] = float(row[i])
            except ValueError:
                # 数値に変換できない場合はそのままにする
                pass
    return data

torch.manual_seed(0)
# Boston Housing Datasetを取得
boston = fetch_openml(name='boston')

# 特徴量とターゲットを取得
x, y = boston.data, boston.target
x0_list = x.values.tolist()
x_list = convert_categorical_to_numeric(x0_list)
y_list = y.values.tolist()

x = torch.tensor(x_list, dtype=torch.float32)
y = torch.tensor(y_list, dtype=torch.float32)

# 学習データとテストデータに分割
dataset = torch.utils.data.TensorDataset(x, y)
n_train = int(len(dataset) * 0.6)
n_val = int(len(dataset) * 0.2)
n_test = len(dataset) - n_train - n_val
train, val, test = torch.utils.data.random_split(dataset, [n_train, n_val, n_test])
batch_size = 1
train_loader = torch.utils.data.DataLoader(train, batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val, batch_size)
test_loader = torch.utils.data.DataLoader(test, batch_size)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=1.e-3)

epoches = 100
for epoch in range(epoches):
  total_loss = 0.0
  num = 0
  for batch in train_loader:
    num += 1
    x, target = batch
    x = x.to(device)
    target = target.to(device)

    # パラメータの勾配を初期化
    optimizer.zero_grad()

    y = model(x)
    loss = model.loss_fn(y, target)
    total_loss += loss
    #print(y, target)
    #print('loss: ', loss.item())

    loss.backward()

    optimizer.step()
  print("epoch : ", epoch, " loss = ", (total_loss/num).item())
