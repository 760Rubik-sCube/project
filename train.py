import torch 
from torch.utils.data import DataLoader,TensorDataset
import torch.nn as nn
import torch.optim as optim
import time
import copy
import os
import random
from Cube import *
from CubeNN import *
import re
from RandomCube import *


print("Creating Model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CubeNN(3).to(device)

print("Read Saved model...")
epoch = 0
file = None
for item in os.listdir():
    if item.endswith(".pth") and item.startswith("cube_new_epoch="): 
        num_text = item.replace("cube_new_epoch=","").replace(".pth","")
        if num_text.isdigit():
            temp_num = int(num_text)
            if temp_num>epoch:
                epoch = temp_num
                file = item
if file != None:
    print(f"{file} found, Loading...")
    model.load_state_dict(torch.load(file))
    print(f"Model loading completed, epoch={epoch}")
else:
    print("Unable to find the saved model, start a new model training")
    


criterion = nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)



def input_int(display,min_value=-float("inf"),max_value=float("inf")):
    while(True):
        user_input=input(display)
        if user_input.isdigit():
            value = int(user_input)
            if min_value<=value<=max_value:
                return value
            else:
                print(f"Input value between {min_value}~{max_value}")
    

df = CubeDataset(3)






def train_model(epoch): 
    model.eval()

    move_count = len(children)
    print("        Start calculating value:")
    with torch.no_grad():
        y_result = model(children[0][0],children[0][1],children[0][2])
        print(f"                [1/{move_count}]")
        for i in range(1, move_count):
            next_result= model(children[i][0],children[i][1],children[i][2])
            y_results= torch.cat((y_result, next_result), dim=1)
            y_result, _ = torch.min(y_results, dim=1, keepdim=True)


            print(f"                [{i+1}/{move_count}]")

        y_result= y_result+1
        y_result = torch.cat((cost_0, y_result), dim=0)    
        y_result =y_result.detach()
    
   
    i = 0
    model.train()
    while(True):
        
        y_pred = model(chidren_with_goal[0], chidren_with_goal[1], chidren_with_goal[2])
        loss = criterion(y_pred, y_result)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_value= loss.item()
        if i%5==0:
            print(f'        [{i+1}] Loss: {round(loss_value,5)}')
        if(i % 5 == 0 and loss_value < 0.05):
            return i==0
        i+=1
     
    


while(True):
    total_epochs = input_int(f"Please enter the target epoch number，current: {epoch}：")
    if total_epochs <= epoch:
        break
    dataset_size = input_int("Please enter the data set size：",1,3000000) 
    move_depth = input_int("Please enter the maximum number of scrambled steps：",1,200)    
    need_new_dataset = True
    for epoch in range(epoch, total_epochs):
        
        
        
        
        if(need_new_dataset):
            print(f"Generate new data...")
            with torch.no_grad():
                cubes, children = df.get_train_dataset(dataset_size,move_depth,device)
                goals,cost_0 = df.get_goal_state(dataset_size,device)
                chidren_with_goal = (torch.cat((goals[0], cubes[0]),dim=0).detach(), torch.cat((goals[1], cubes[1]), dim=1).detach(),torch.cat((goals[2], cubes[2]), dim=1).detach())
            print("Data generation completed!")
        

        print(f'Start Epoch[{epoch+1}/{total_epochs}]')
        need_new_dataset = train_model(epoch)
        print(f'   End Epoch[{epoch+1}/{total_epochs}]')
        
        

        if((epoch+1)%50==0):
            torch.save(model.state_dict(), f'cube_new_epoch={epoch+1}.pth')
            del_file = f'cube_new_epoch={epoch+1-1000}.pth'
            if os.path.exists(del_file):
                os.remove(del_file)
            
    torch.save(model.state_dict(), f'cube_new_epoch={epoch+1}.pth')
    epoch+=1
    
    
