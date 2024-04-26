import torch
import random
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

class Neural_Network:
  def __init__(self, layers, init_seed):
    self.layers = layers
    self.num_layers = len(layers)
    self.weights = []
    self.biases = []
    np.random.seed(init_seed)  # 乱数生成器にシードを設定
    for i in range(1, self.num_layers):
      w_layer = []
      b_layer = []
      for j in range(int(self.layers[i])):
        b_layer.append(0.0)
        w_node = [random.gauss(0.0, (2.0/self.layers[i-1])**0.5) for _ in range(int(self.layers[i-1]))] #He初期化
        w_layer.append(w_node)
      self.weights.append(w_layer)
      self.biases.append(b_layer)

  def ReLU(self, x):
    return x * (x > 0)

  def forward_prop(self, x, cache_mode=False):
    cached_sums = []  # 記録した全ノードの線形和（Σ）の値
    cached_outs = []  # 記録した全ノードの活性化関数の出力値

    #input layer
    cached_outs.append(x)  # 何も処理せずに出力値を記録
    next_x = x  # 現在の層の出力（x）＝次の層への入力（next_x）

    SKIP_INPUT_LAYER = 1
    for layer_i, layer in enumerate(self.layers):
        if layer_i == 0:
            continue  #skip input lawyer

        w_tensor = torch.tensor(self.weights[layer_i - SKIP_INPUT_LAYER]).float()
        x_tensor = ((torch.tensor(next_x)).reshape(len(next_x),1)).float()
        b_tensor = torch.tensor(self.biases[layer_i - SKIP_INPUT_LAYER]).reshape(1,len(self.biases[layer_i - SKIP_INPUT_LAYER])).float()
        node_sum_tensor = (torch.mm(w_tensor,x_tensor).t()+b_tensor)
        if layer_i < len(layers)-1: #except output layer
          node_out_tensor = self.ReLU(node_sum_tensor)
        else:
          node_out_tensor = node_sum_tensor
        node_sum_list = (node_sum_tensor.view(-1)).tolist()
        node_out_list = (node_out_tensor.view(-1)).tolist()
        cached_sums.append(node_sum_list)
        cached_outs.append(node_out_list)
        next_x = node_out_list

    if cache_mode:
        return (cached_outs[-1], cached_outs, cached_sums)

    return cached_outs[-1]

  def MSE_loss(self, y_pred, y_true):
    return 0.5*(y_pred-y_true)**2

# Boston Housing Datasetを取得
boston = fetch_openml(name='boston')

# 特徴量とターゲットを取得
x, y = boston.data, boston.target

# 学習データとテストデータに分割
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# DataFrameからリストに変換
x_train_list0 = x_train.values.tolist()
y_train_list = y_train.values.tolist()
x_test_list0 = x_test.values.tolist()
y_test_list = y_test.values.tolist()

def convert_categorical_to_numeric(data):
    for row in data:
        for i in range(len(row)):
            try:
                row[i] = float(row[i])
            except ValueError:
                # 数値に変換できない場合はそのままにする
                pass
    return data

# カテゴリーを数値に変換
x_train_list = convert_categorical_to_numeric(x_train_list0)
x_test_list = convert_categorical_to_numeric(x_test_list0)

num_node_input = len(x_train_list[0])
layers = [num_node_input,5,1]

model = Neural_Network(layers, init_seed=1)
y_pred, outs, sums = model.forward_prop(x_train_list[0], cache_mode=True)
print(y_pred, [y_train_list[0]])
target = [y_train_list[0]]
loss = 0.0
for output, ground_truth in zip(y_pred, target):
  loss += model.MSE_loss(output, ground_truth)
loss /= len(y_pred)
print("loss = ", loss)
