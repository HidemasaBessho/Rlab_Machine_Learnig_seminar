import glob
import re
import numpy as np
import torch
import torch_geometric


class Dataset(torch.nn.Module):

    def __init__(self,
                 file_pattern,
                 max_files_to_load=None,
                 n_particles=4096, 
                 edge_threshold=2.0,                 
                 if_train=True
                 ):

        print("max_files_to_load = ", max_files_to_load)        

        filenames = sorted(glob.glob(file_pattern), key=self.natural_keys) #ソートする
        if max_files_to_load: filenames = filenames[:max_files_to_load]

        self.filenames = filenames    
        self.edge_threshold = edge_threshold
        self.num_nodes = n_particles        # number of particles: fixed
        self.phi = 1.2                      # number density of the system
        self.box = np.full(3, (float(self.num_nodes) / self.phi )**(1.0/3.0) ) # length of simulation box, shapeが[1,3]のndarray

        self.data = [] #inputするグラフデータを保持する
        self.init_positions_data = [] #初期配置を保持する

        for filename in self.filenames:
            print("loading: ", filename)

            npz = np.load(filename) #npz : ndarrayを名前付きで保存することができる
            types = npz['types'] # 'types'という名前がつけられて保存されたadarrayを、typesという名前で取り出す
            initial_positions = npz['initial_positions']
            positions = npz['positions']
        
            node_feature, edge_index, edge_feature = self.make_graph_from_static_structure(initial_positions,types)

            target_n, mask_n = self.get_targets_node(initial_positions, positions, types, if_train)

            target_n = torch.tensor(target_n[:, None], dtype=torch.float)
            mask_n = torch.tensor(mask_n[:, None], dtype=torch.bool)
          
            graph = torch_geometric.data.Data(x=node_feature,
                                              edge_index=edge_index,
                                              edge_attr=edge_feature,
                                              y=target_n,
                                              mask=mask_n)

            self.init_positions_data.append(initial_positions)
            self.data.append(graph)

    
    def atoi(self, text):
        return int(text) if text.isdigit() else text

    def natural_keys(self, text):
        return [ self.atoi(c) for c in re.split(r'(\d+)', text) ]
        
    def __len__(self):
        return len( self.filenames )

    def get_init_positions(self, idx):
        return self.init_positions_data[idx]
    
    def __getitem__(self, idx):
        graph = self.data[idx]
        graph = self.apply_random_rotation(graph)
        return graph

    def get_targets_node(self, initial_positions, trajectory_target_positions, types, if_train=True):
        """Returns the averaged particle mobilities from the sampled trajectories.  
        """    
        targets = np.mean([ np.linalg.norm(t - initial_positions, axis=-1)
                            for t in trajectory_target_positions], axis=0) #ノルムを計算し，平均値をとっている? アイソコンフィギュレーションだけ？ list内包表記

        mask = np.ones(len(targets), dtype=bool) #長さがlen(targets)で, 全ての要素がTrueであるブール型の配列
        if (if_train == False):
            mask = (types == 0).astype(bool) #if_trainがFalseの時, つまりvalidationの時, typesの値が0の要素に対応する位置がTrue, それ以外の位置がFalse
        
        return targets.astype(np.float32), mask
    
    def make_graph_from_static_structure(self, positions, types):

        cross_positions = positions[None, :, :] - positions[:, None, :]
      
        box_ = self.box[None, None, :]
        cross_positions += (cross_positions < -box_ / 2.).astype(np.float32) * box_
        cross_positions -= (cross_positions > box_ / 2.).astype(np.float32) * box_
                                                                           
        distances = np.linalg.norm(cross_positions, axis=-1)
        indices = np.where((distances < self.edge_threshold) & (distances > 1e-6)) #条件を満たすインデックスのみを取り出す

        node_feature = torch.tensor(types[:,None], dtype=torch.float) #node_featureの形状はノード数×ノードの特徴量の種類数, [num_nodes, num_node_features]
        edge_index = torch.tensor(np.array(indices), dtype=torch.long) #edge_indexの形状は2×エッジの数, [2, num_edge]
        edge_feature = torch.tensor(cross_positions[indices], dtype=torch.float) #edge_indexに該当するcross_positionsのみを取り出す
        
        return node_feature, edge_index, edge_feature



    def apply_random_rotation(self, graph):
        # Transposes edge features, so that the axes are in the first dimension.
        xyz = torch.transpose(graph.edge_attr, 0, 1)
        xyz = xyz[torch.randperm(3)]
        # Random reflections.
        symmetry =  np.random.randint(0, 2, [3])
        symmetry = torch.tensor( 1 - 2 * np.reshape(symmetry, [3, 1]), dtype=torch.float)
        xyz = xyz*symmetry 
        graph.edge_attr = torch.transpose(xyz, 0, 1)
    
        return graph
