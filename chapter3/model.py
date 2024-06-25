import torch
import torch_geometric
from torch import Tensor
from torch_scatter import scatter
from typing import Optional, Tuple

class Model(torch.nn.Module):

    def __init__(self):

        super().__init__()

        n_node_feature=1
        n_edge_feature=3
        mlp_size=64

        self.node_encoder = MLP(n_node_feature, mlp_size) #ノードのエンコーダーの初期化 : inputノードの次元を圧縮する
        self.edge_encoder = MLP(n_edge_feature, mlp_size) #エッジのエンコーダーの初期化 : inputエッジの次元を圧縮する

        self.decoder = MLP(mlp_size, mlp_size, decoder_layer=True) #デコーダーの初期化 : 次元を復元する
        
        self.num_message_passing_steps = 7     
        self.network = Iterative_Layers(edge_model=EdgeUpdate(mlp_size),
                                        node_model=NodeUpdate(mlp_size),
                                        steps=self.num_message_passing_steps) #Iterative_Layersの初期化

    def forward(self, data):   

        edge_index = data.edge_index    
        batch = data.batch

        encoded_x = self.node_encoder(data.x) #MLPのforwardをやっている
        encoded_edge_attr = self.edge_encoder(data.edge_attr) #MLPのforwardをやっている

        x, edge_attr = self.network(encoded_x,
                                    edge_index,
                                    encoded_edge_attr,
                                    batch) #Iterative_Layersのforwardをやっている
                
        return self.decoder(x)
        
    
class EdgeUpdate(torch.nn.Module):

    def __init__(self, mlp_size):
        super().__init__()
        self.mlp = MLP(4*mlp_size, mlp_size) 
        
    def forward(self, src, dest, edge_attr, encoded_edge, batch):        
        edge_input = torch.cat([src, dest, edge_attr, encoded_edge], dim=1) #inputした引数をひとまとめにする
        out = self.mlp(edge_input) # = MLP(4*mlp_size, mlp_size) = MLPのforward
        return out


class NodeUpdate(torch.nn.Module):

    def __init__(self, mlp_size):
        
        super().__init__()
        self.mlp = MLP(3*mlp_size, mlp_size)

    def forward(self, x, edge_index, edge_attr, encoded_x, batch):
        row, col = edge_index
        recv = scatter(edge_attr, col, dim=0, dim_size=x.size(0))
        out = torch.cat([x, encoded_x, recv], dim=1) 
        out = self.mlp(out)

        return out


class MLP(torch.nn.Module):

    def __init__(self, initial_size, mlp_size, decoder_layer=False):
        super().__init__()
        
        self.fc1 = torch.nn.Linear(initial_size, mlp_size)
        self.fc2 = torch.nn.Linear(mlp_size, mlp_size)
        self.decoder_layer = decoder_layer
        if decoder_layer == True:
            self.fc3 = torch.nn.Linear(mlp_size, 1)

    def forward(self, x):

        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        x = torch.nn.functional.relu(x)
        if self.decoder_layer == True:
            x = self.fc3(x)

        return x


class Iterative_Layers(torch.nn.Module):

    def __init__(self, edge_model, node_model, steps=7):
        super().__init__()
        self.edge_model = edge_model #Modelクラスを見ると，edge_modelはEdgeUpdate(mlp_size)
        self.node_model = node_model #Modelクラスを見ると，node_modelはNodeUpdate(mlp_size)
        self.num_message_passing_steps = steps        
        self.reset_parameters()

    def reset_parameters(self):
        for item in [self.node_model, self.edge_model]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()

    def forward(
            self,
            x: Tensor,
            edge_index: Tensor,
            edge_attr: Tensor,
            batch: Optional[Tensor] = None)  -> Tuple[Tensor, Tensor]:

        row = edge_index[0]   #src(送信側)
        col = edge_index[1]   #dest(受信側)
        encoded_x = x
        encoded_edge = edge_attr

        for _ in range(self.num_message_passing_steps):

            if self.edge_model is not None:

                edge_attr = self.edge_model(x[row],
                                            x[col],
                                            edge_attr,
                                            encoded_edge,
                                            batch)

                
            if self.node_model is not None:

                x = self.node_model(x,
                                    edge_index,
                                    edge_attr,
                                    encoded_x,
                                    batch)

        return x, edge_attr
