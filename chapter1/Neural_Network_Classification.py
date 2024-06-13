import torch
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt

class Net(nn.Module):

    # 使用するオブジェクトを定義
    def __init__(self, n_input, n_output):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_input, 10)
        self.fc2 = nn.Linear(10, n_output)

    # 順伝播
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

    def loss_fn(self, y, t):
        return F.cross_entropy(y, t)

torch.manual_seed(0)

# データをロードし、訓練データとテストデータに分割
iris = load_iris()
x0, y = iris.data, iris.target

# 特徴量の標準化
scaler = StandardScaler()
x0 = scaler.fit_transform(x0)

# データをテンソルに変換
x = torch.tensor(x0, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.int64)

# 入力と出力のサイズ
n_input = len(x[0])
n_output = (torch.max(y)+1).item()

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
model = Net(n_input, n_output).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1.e-3, eps=1e-8)

epochs = 200
train_epoch_list = []
val_epoch_list = []
total_loss_list = []
valid_loss_list = []
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    num = 0
    for batch in train_loader:
        num += 1
        Input, Target = batch
        Input = Input.to(device)
        Target = Target.to(device)

        optimizer.zero_grad()
        non_active_output, Prediction = model(Input)
        #print(Prediction)
        loss = model.loss_fn(non_active_output, Target)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    print("Epoch: ", epoch + 1, " Loss: ", total_loss / num)
    train_epoch_list.append(epoch)
    total_loss_list.append(total_loss/num)

    if (epoch%5)==0:
      val_loss = 0.0
      val_num = 0
      correct = 0
      total = 0
      for batch in val_loader:
        val_num += 1
        Input, Target = batch
        Input = Input.to(device)
        Target = Target.to(device)

        optimizer.zero_grad()
        non_active_output, Prediction = model(Input)
        loss = model.loss_fn(non_active_output, Target)
        val_loss += loss.item()
        #total += Target.size(0)
        #_, predicted = torch.max(non_active_output.data, 1)
        #correct += (predicted == Target).sum().item()
      val_epoch_list.append(epoch)
      valid_loss_list.append(val_loss/val_num)
      print("Epoch: ", epoch + 1, " Validation Loss: ", val_loss / val_num)
      #print('Accuracy: {:.2f}%'.format(100 * correct / total))

#plt.xscale('log')
#plt.yscale('log')
plt.plot(train_epoch_list, total_loss_list, label="train", color="blue")
plt.plot(val_epoch_list, valid_loss_list, label="validation", color="darkorange")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xlim(0,200)
plt.ylim(0, 1.e+0)
plt.legend(ncol=1, loc=1, borderaxespad=0, fontsize=15,frameon=True)
plt.show()

# モデルの評価
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch in test_loader:
        Input, Target = batch
        Input = Input.to(device)
        Target = Target.to(device)

        non_active_output, Prediction = model(Input)
        _, predicted = torch.max(non_active_output.data, 1)
        total += Target.size(0)
        correct += (predicted == Target).sum().item()

print('Accuracy: {:.2f}%'.format(100 * correct / total))
