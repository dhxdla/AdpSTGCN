import torch.optim as optim
import utils.loss_box as loss_box
import numpy as np
from utils.loadAsave_tools import load_topo_adjs,load_best_model,save_model,save_dict
from model.ada_stgcn.ada_STGCN import *
import math
class Engine():
    def __init__(self, args, dataset, dataloader, scaler):
        self.configs = vars(args)
        self.dataset = dataset
        self.dataloader = dataloader
        self.scaler = scaler
        self.load_model(self.configs)
        self.loss = loss_box.masked_mae
        self.optimizer = self.get_optimizer(self.model, self.configs['optimizer'], self.configs['learning_rate'], self.configs['weight_decay'])


    def load_model(self, configs):
        topo_graph = load_topo_adjs(configs['root_path']+configs['adj_path'],configs['n_nodes']).to(self.configs['device'])
        self.model = ada_gcn(configs, topo_graph).to(configs['device'])
        
    def get_optimizer(self, model, opt_name, learning_rate, weight_decay):
        if opt_name == 'adam':
            return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    def run_exp(self):
        val_mae = []
        val_rmse = []
        val_mape = []
        test = []
        mean_test = []
        OUT = []
        for epoch in range(1, self.configs['epochs']+1):
            self.model.train()
            for iters, (seq_x, seq_y, seq_x_mark, seq_y_mark) in enumerate(self.dataloader['train']):
                # seq_x: 输入序列, seq_y: 真实序列, xxx_mark: 相应序列的时间信息 
                self.optimizer.zero_grad()
                # model output
                output = self.model(seq_x)
                predict = self.scaler.inverse_transform(output)
                real = self.scaler.inverse_transform(seq_y)
                loss = self.loss(predict,real,0.0)
                loss.backward() # optimization
                self.optimizer.step()
            
            
            
            self.model.eval()
            mae = 0
            mape = 0
            mse = 0
            for iters, (seq_x, seq_y, seq_x_mark, seq_y_mark) in enumerate(self.dataloader['vali']):
                # seq_x: 输入序列, seq_y: 真实序列, xxx_mark: 相应序列的时间信息 
                with torch.no_grad():
                # model output
                    output = self.model(seq_x)
                    predict = self.scaler.inverse_transform(output)
                    real = self.scaler.inverse_transform(seq_y)
                    info = loss_box.metric(predict,real,0.0)
                    mae += info[0]
                    mape += info[1]
                    mse += info[2]
            L = len(self.dataloader['vali'])
            mae = mae/L
            mape = mape/L
            rmse = math.sqrt(mse/L)
            val_mae.append(mae)
            val_mape.append(mape)
            val_rmse.append(rmse)
            print('vali',mae,mape,rmse)


            mae = 0
            mape = 0
            mse = 0
            amae = []
            amape = []
            amse = []
            out = []
            rrreal = []
            for iters, (seq_x, seq_y, seq_x_mark, seq_y_mark) in enumerate(self.dataloader['test']):
                # seq_x: 输入序列, seq_y: 真实序列, xxx_mark: 相应序列的时间信息 
                with torch.no_grad():
                # model output
                    output = self.model(seq_x)
                    predict = self.scaler.inverse_transform(output)
                    
                    real = self.scaler.inverse_transform(seq_y)
                    out.append(predict.cpu())
                    rrreal.append(real.cpu())

                    info = loss_box.metric(predict,real,0.0)
                    mae += info[0]
                    mape += info[1]
                    mse += info[2]
                    temp1 = []
                    temp2 = []
                    temp3 =[]
                    for i in range(12):
                        metrics = loss_box.metric(predict[:,i,:],real[:,i,:],0.0)
                        temp1.append(metrics[0])
                        temp2.append(metrics[1])
                        temp3.append(metrics[2])
                    amae.append(temp1)
                    amape.append(temp2)
                    amse.append(temp3)
            L = len(self.dataloader['test'])
            mae = mae/L
            mape = mape/L
            rmse = math.sqrt(mse/L)
            OUT.append([out,rrreal])
            
            mean_test.append([mae,mape,rmse])

            mae12 = torch.mean(torch.tensor(amae), dim=0)
            mape12 = torch.mean(torch.tensor(amape), dim=0)
            rmse12 = torch.sqrt(torch.mean(torch.tensor(amse), dim=0))

            
            print(mae12)
            print(mape12)
            print(rmse12)
            test.append([mae12,mape12,rmse12])

        


        index3= np.argmin(val_rmse)

        pre = np.array(OUT[index3])
        print(pre.shape)
        np.save('./huatuyongde',pre)
        print("最佳结果为：")
        print(mean_test[index3])
        print(test[index3])






        
