import torch, torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

# データ読み込み時の処理
from torchvision import transforms

class Net(nn.Module):
    def __init__(self, input_size=784, hidden_size=100, output_size=10, batch_size=256):
        super(Net, self).__init__()
        self.batch_size = batch_size
        # 使用する層の宣言
        self.conv = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def loss_fn(self, y, t):
        return F.cross_entropy(y, t)

    def forward(self, x):
        x = self.conv(x) #畳み込み処理
        x = F.max_pool2d(x, 2, 2) #プーリング処理
        x = x.view(-1, x.size(0)) #tensor -> vectorの処理
        x = F.relu(self.fc1(x)) #
        x = self.fc2(x)
        return x

# データ読み込み時に行う処理
transform = transforms.Compose([
    transforms.ToTensor()
])

# データセットの取得
train_val = torchvision.datasets.MNIST(
    root='.',
    train=True,
    download=True,
    transform=transform)

test = torchvision.datasets.MNIST(
    root='.',
    train=False,
    download=True,
    transform=transform)
# train : val = 80% : 20%
n_train = int(len(train_val) * 0.8)
n_val = len(train_val) - n_train
    
# データをランダムに分割
torch.manual_seed(0)
train, val = torch.utils.data.random_split(train_val, [n_train, n_val])
len(train), len(val), len(test)
batch_size = 100
train_loader = torch.utils.data.DataLoader(train, batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val, batch_size)
test_loader = torch.utils.data.DataLoader(test, batch_size)

# cuDNN に対する再現性の確保

torch.backends.cudnn.deterministic = True
# CuDNNは内部で非決定的なアルゴリズム（例えば，畳み込みやプーリングの操作）を使用することがある．これにより，同じ入力データを使用しても，異なる実行結果が得られる場合がある．
# torch.backends.cudnn.deterministic = Trueと設定することで，CuDNNは決定的なアルゴリズムを使用し，再現性を確保する．

torch.backends.cudnn.benchmark = False
# CuDNNは、与えられた入力サイズやレイヤー構造に最適なアルゴリズムを選択するために，ベンチマークを実行することがある．
# このベンチマークは初回実行時に行われ，その後は最適なアルゴリズムが使用される．
# ベンチマークを無効にし、事前定義されたアルゴリズムを使用する．

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1.e-3, eps=1e-8)

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
    print("epoch : ", epoch+1, " loss = ", (total_loss/num).item())
