import torch
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt

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
