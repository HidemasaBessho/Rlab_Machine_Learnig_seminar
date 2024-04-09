import random
import logging
import numpy as np
import torch
import torch_geometric
import torch.nn.functional as F 

from model import Model 
from load_data import Dataset


def train_model(p_frac,
                temperature,
                train_file_pattern,
                test_file_pattern,
                max_files_to_load=None,
                n_epochs=1000,
                n_particles=4096, 
                tcl=7,
                learning_rate=1e-4,
                grad_clip=1.0,
                measurement_store_interval=5,
                seed=0
                ):


    torch.manual_seed(seed) 
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    trainset = Dataset(train_file_pattern,
                       max_files_to_load,
                       n_particles,
                       edge_threshold=2.0
                       )
    
    testset = Dataset(test_file_pattern,
                      max_files_to_load,
                      n_particles, 
                      edge_threshold=2.0,                      
                      if_train=False)

    train_batch_size = 1 if (p_frac > 1.0 - 1e-6) else 1
    
    train_data_loader = torch_geometric.loader.DataLoader(trainset, batch_size=train_batch_size, shuffle=True)
    test_data_loader = torch_geometric.loader.DataLoader(testset, batch_size=1, shuffle=False)
    
    model = Model().to(device)
    
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1e-8)
    
    for epoch in range(n_epochs+1):

        train_losses = []        
        for data in train_data_loader:        
            data = data.to(device) 
            optimizer.zero_grad()                        
            params_node = model(data)
            loss_node = loss_fn(params_node, data.y)
            loss = loss_node
            loss.backward()

            if grad_clip != float('inf'):
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip) #勾配爆発防止

            optimizer.step()
            train_losses.append( loss.item() )
            

        if epoch%5==0 and test_data_loader is not None:    

            valid_losses = []
            valid_stats_node = []

            cnt = 0
            for data in test_data_loader:

                data = data.to(device)                
                prediction_node = model(data)
                loss_node = loss_fn(prediction_node, data.y)
                loss = loss_node               
                loss_value = loss.item()            
                valid_losses.append(loss_value)

                prediction_node = torch.squeeze(prediction_node).to('cpu').detach().numpy().copy()
                target_node = torch.squeeze(data.y).to('cpu').detach().numpy().copy()
                mask_node = torch.squeeze(data.mask).to('cpu').detach().numpy().copy()

                valid_stats_node.append(np.corrcoef(prediction_node[mask_node == True],target_node[mask_node == True])[0, 1] )
              
                if (cnt==0 and epoch%500 == 0):
                    fn = "./result/T" + '{:.2f}'.format(temperature) + "/T" + '{:.2f}'.format(temperature) + "_" + str(cnt) + "_tc" + str(tcl) \
                        + "_pred_n_" + str(epoch) + "_" + str(seed+1) + ".dat"
                    prediction_output(testset, prediction_node[0:4096], target_node[0:4096],  fn)
                cnt = cnt + 1

            fm = "./loss/T" + '{:.2f}'.format(temperature) + "/loss_frac" + '{:.2f}'.format(p_frac)   + "_T" + '{:.2f}'.format(temperature) \
                 + "_tc" + str(tcl) + "_" + str(seed+1) + ".dat"
            with open(fm, 'a') as f:
                f.write( str(epoch) + "," + str(np.mean(train_losses)) + "," + str( np.mean(valid_losses)) + "," + str(np.mean(valid_stats_node)) + "\n")

    torch.save(model.state_dict(),'./save_train_model/model_weight/T' + '{:.2f}'.format(temperature) + '/model_weight'+"_T"+ '{:.2f}'.format(temperature) + "_tc" + str(tcl)+'.pth')


def prediction_output(testset, predictions, targets, filename):
    with open(filename, 'w') as f:
        f.write("x,y,z,prediction,targets\n")
        ralist = testset.get_init_positions(0)
        for i,(pdt,tgt) in enumerate( zip(predictions, targets) ):
            ra = ralist[i]
            f.write( str(ra[0]) +","+ str(ra[1]) +","+ str(ra[2]) +","+ str(pdt) +","+ str(tgt) + "\n" )
