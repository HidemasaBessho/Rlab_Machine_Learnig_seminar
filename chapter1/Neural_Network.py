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
    random.seed(init_seed)
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
    node_out = []
    for i in range(len(x[0])):
      if x[0, i] > 0.0:
        node_out.append(x[0, i])
      else:
        node_out.append(0.0)
    return torch.tensor(node_out).reshape(1, len(x[0]))

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

  def MSE_loss_der(self, y_pred, y_true):
    return y_pred - y_true

  def identity_der(self, x):
    return 1.0

  def ReLU_der(self, x):
    return 0.0 if x < 0.0 else 1.0

  def sum_der(self, x, weights, bias, with_respect_to='w'):
    # ※1データ分、つまりxとweightsは「一次元リスト」という前提。
    if with_respect_to == 'w':
        return x  # 線形和uを各重みw_iで偏微分するとx_iになる（iはノード番号）
    elif with_respect_to == 'b':
        return 1.0  # 線形和uをバイアスbで偏微分すると1になる
    elif with_respect_to == 'x':
        return weights  # 線形和uを各入力x_iで偏微分するとw_iになる

  def back_prop(self, y_true, cached_outs, cached_sums):
    grads_w =[]  # 重みの勾配
    grads_b = []  # バイアスの勾配
    grads_x = []  # 入力の勾配

    layer_count = len(self.layers)
    layer_max_i = layer_count-1
    SKIP_INPUT_LAYER = 1
    PREV_LAYER = 1
    rng = range(SKIP_INPUT_LAYER, layer_count)
    for layer_i in reversed(rng):
        is_output_layer = (layer_i == layer_max_i)
        layer_grads_w = []
        layer_grads_b = []
        layer_grads_x = [] #layerごとのgrad

        if is_output_layer:
            back_error = []
            y_pred = cached_outs[layer_i]
            for output, target in zip(y_pred, y_true):
                loss_der = self.MSE_loss_der(output, target)
                back_error.append(loss_der)
        else:
            back_error = grads_x[-1]

        node_sums = cached_sums[layer_i - SKIP_INPUT_LAYER]
        for node_i, node_sum in enumerate(node_sums):
            if is_output_layer:
                active_der = self.identity_der(node_sum)
            else:
                active_der = self.ReLU_der(node_sum)
            w = self.weights[layer_i - SKIP_INPUT_LAYER][node_i]
            b = self.biases[layer_i - SKIP_INPUT_LAYER][node_i]
            x = cached_outs[layer_i - PREV_LAYER]
            sum_der_w = self.sum_der(x, w, b, with_respect_to='w')
            sum_der_b = self.sum_der(x, w, b, with_respect_to='b')
            sum_der_x = self.sum_der(x, w, b, with_respect_to='x')

            delta = back_error[node_i] * active_der

            grad_b = delta * sum_der_b
            layer_grads_b.append(grad_b)

            node_grads_w = []
            for x_i, (each_dw, each_dx) in enumerate(zip(sum_der_w, sum_der_x)):
                grad_w = delta * each_dw
                node_grads_w.append(grad_w)

                grad_x = delta * each_dx
                if node_i == 0:
                    layer_grads_x.append(grad_x)
                else:
                    layer_grads_x[x_i] += grad_x
            layer_grads_w.append(node_grads_w)

        grads_w.append(layer_grads_w)
        grads_b.append(layer_grads_b)
        grads_x.append(layer_grads_x)

    grads_w.reverse()
    grads_b.reverse()
    return grads_w, grads_b

  def update_params(self,grads_w, grads_b,lr):
    # ネットワーク全体で勾配を保持するためのリスト
    new_weights = [] # 重み
    new_biases = [] # バイアス

    SKIP_INPUT_LAYER = 1
    for layer_i, layer in enumerate(layers):  # 各層を処理
        if layer_i == 0:
            continue  # 入力層はスキップ

        # 層ごとで勾配を保持するためのリスト
        layer_w = []
        layer_b = []

        for node_i in range(layer):  # 層の中の各ノードを処理
            b = self.biases[layer_i - SKIP_INPUT_LAYER][node_i]
            grad_b = grads_b[layer_i - SKIP_INPUT_LAYER][node_i]
            b = b - lr*grad_b
            layer_b.append(b)

            node_weights = self.weights[layer_i - SKIP_INPUT_LAYER][node_i]
            node_w = []
            for each_w_i, w in enumerate(node_weights):
                grad_w = grads_w[layer_i - SKIP_INPUT_LAYER][node_i][each_w_i]
                w = w - lr*grad_w
                node_w.append(w)
            layer_w.append(node_w)

        new_weights.append(layer_w)
        new_biases.append(layer_b)

    return new_weights, new_biases

  def train(self, x, y, epochs, lr, train_seed):
    losses = []
    epoches = []
    data = list(zip(x, y))  # xとyをペアにしてリスト化
    out_count = 1.0
    out_num = 10
    for epoch in range(1, epochs+1):
        total_loss = 0.0
        if epoch != 0:
            random.shuffle(data)  # データをランダムにシャッフル
        np.random.seed(train_seed)
        for input, target in data:

            # 順伝播
            y_pred, outs, sums = self.forward_prop(input, True)

            # 誤差計算
            #for output, ground_truth in zip(y_pred, target):
                #loss = self.sseloss(output, ground_truth)
                #total_loss += loss

                # 誤差逆伝播
                grads_w, grads_b = self.back_prop(target, outs, sums)

                # パラメータ更新
                new_w, new_b = self.update_params(grads_w, grads_b, lr)

                # 更新したパラメータを反映
                self.weights = new_w
                self.biases = new_b

            for input, target in data:

                # 順伝播
                y_pred, outs, sums = self.forward_prop(input, True)

                # 誤差計算
                for output, ground_truth in zip(y_pred, target):
                  loss = self.sseloss(output, ground_truth)
                  total_loss += loss

            # エポックごとに平均誤差を表示
            avg_loss = total_loss / len(x) /50.0
            losses.append(avg_loss)
            epoches.append(epoch)

        plt.xscale('log')
        plt.yscale('log')
        plt.plot(losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Time')
        plt.xlim(1,1.e+4)
        plt.ylim(1.e+0, 3.e+5)
        plt.show()
    

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
