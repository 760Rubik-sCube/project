import torch 
from torch.utils.data import DataLoader,TensorDataset
import torch.nn as nn
import torch.optim as optim

class CubeNN(nn.Module):
    def __init__(self,level):
        super(CubeNN, self).__init__() 
        #input layer -> h1
        self.c_input = nn.Linear(144, 6000)
        self.s_inputs = nn.ModuleList()
        for _ in range(level-2):
            self.s_inputs.append(nn.Linear(144, 6000))
        self.f_inputs = nn.ModuleList()    
        for _ in range((level-2)**2):
            self.f_inputs.append(nn.Linear(36, 6000))
        self.bn1 = nn.BatchNorm1d(6000)
        #input h1 -> block
        self.hidden1 = nn.Linear(6000, 2000) 
        self.bn2 = nn.BatchNorm1d(2000)

        #input  blocks
        self.blocks = nn.ModuleList()
        for block_num in range(5):
            res_fc1 = nn.Linear(2000, 2000)
            res_bn1 = nn.BatchNorm1d(2000)
            res_fc2 = nn.Linear(2000, 2000)
            res_bn2 = nn.BatchNorm1d(2000)
            self.blocks.append(nn.ModuleList([res_fc1, res_bn1, res_fc2, res_bn2]))
        self.output = nn.Linear(2000, 1)

    def one_hot(self,data):
        x,y = data.size()
        data = nn.functional.one_hot(data, 6)
        data = data.float()
        data = data.view(-1,y*6)
        return data

    def forward(self,c,s,f):
        y_pred = self.c_input(self.one_hot(c))
        for i in range(len(self.s_inputs)):
            y_pred += self.s_inputs[i](self.one_hot(s[i])) 
            
        for i in range(len(self.f_inputs)):
            y_pred += self.f_inputs[i](self.one_hot(f[i]))

        y_pred = self.bn1(y_pred)
        y_pred = nn.functional.relu(y_pred)

        y_pred = self.hidden1(y_pred)
        y_pred = self.bn2(y_pred)
        y_pred = nn.functional.relu(y_pred)
        
        for block_num in range(len(self.blocks)):
            res_inp = y_pred
            y_pred = self.blocks[block_num][0](y_pred)
            y_pred = self.blocks[block_num][1](y_pred)
            y_pred = nn.functional.relu(y_pred)
            y_pred = self.blocks[block_num][2](y_pred)
            y_pred = self.blocks[block_num][3](y_pred)
            y_pred = nn.functional.relu(y_pred + res_inp)
        
        

        y_pred = self.output(y_pred)
        return y_pred
    