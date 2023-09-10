import torch 
from Cube import *
import random
import copy
class CubeDataset():
    def __init__(self,level):
        self.level = level
        self.move_data={"x":[0]*(level-1),"y":[0]*(level-1),"z":[0]*(level-1)}
        self.moves=[]
        for index in range(0,level-1):
            for axis in ["x","y","z"]:
                for clockwise in [True,False]:
                    self.moves.append(CubeMove(axis,index, clockwise))
    
    
    
    def get_train_dataset(self, size, depth, device):
        cubes, cube_data,_ = self.get_random_cubes(size, depth, device)
        children_moves =[]
        for move in self.moves:
            children_moves.append(self.__get_childern(cubes,move, device))
        return cube_data, children_moves
        
    def get_goal_state(self,size,device):
        corns=[]
        sideGroups = [[] for _ in range(self.level-2)]
        faceGroups = [[] for _ in range((self.level-2)**2)]
        costs = []
        cube = Cube(self.level)
        c,s,f = cube.train_data()
        
        for _ in range(size):
            costs.append([0])
            corns.append(c)         
            for i in range(len(sideGroups)):
                sideGroups[i].append(s[i])        
            for i in range(len(faceGroups)):
                faceGroups[i].append(f[i])
                
        corn_t =  torch.tensor(corns).to(device)
        side_t =  torch.tensor(sideGroups).to(device)
        face_t = torch.tensor(faceGroups).to(device)
        cost_t = torch.tensor(costs).to(device).float()
        return (corn_t, side_t, face_t) , cost_t          
    

    def get_random_cubes(self, size, depth, device):
        cubes=[]
        corns=[]
        sideGroups = [[] for _ in range(self.level-2)]
        faceGroups = [[] for _ in range((self.level-2)**2)]
        labels = []
        for _ in range(size):
            cube = Cube(self.level)
            self.__clear_move_data()
            for step in range(1,depth+1): 
                move = self.moves[random.randint(0, len(self.moves)-1)]
                while(not self.__try_move(cube,move)):
                    move = self.moves[random.randint(0, len(self.moves)-1)]
                if cube.is_solve():
                    continue
                cubes.append(copy.deepcopy(cube))
                c,s,f = cube.train_data()
                
                labels.append([step])
                corns.append(c)         
                for i in range(len(sideGroups)):
                    sideGroups[i].append(s[i])        
                for i in range(len(faceGroups)):
                    faceGroups[i].append(f[i])  
                    
        corn_t =  torch.tensor(corns).to(device)
        side_t =  torch.tensor(sideGroups).to(device)
        face_t = torch.tensor(faceGroups).to(device)
        labels_t =torch.tensor(labels).to(device).float()
        return cubes, (corn_t, side_t, face_t), labels_t

    
    
    def __get_childern(self,cubes,move,device):
        corns=[]
        sideGroups = [[] for _ in range(self.level-2)]
        faceGroups = [[] for _ in range((self.level-2)**2)]
        
        for cube in cubes:

            copy_cube =copy.deepcopy(cube)
            copy_cube.rotate(move)
            c, s, f = copy_cube.train_data()
            corns.append(c)         
            for i in range(len(sideGroups)):
                sideGroups[i].append(s[i])        
            for i in range(len(faceGroups)):
                faceGroups[i].append(f[i])    
        
        corn_t =  torch.tensor(corns).to(device)
        side_t =  torch.tensor(sideGroups).to(device)
        face_t = torch.tensor(faceGroups).to(device)
        return (corn_t, side_t, face_t)
            
        


    
    def __clear_move_data(self):
        for item in self.move_data.values():
            for i in range(len(item)):
                item[i]= 0
                
    def __try_move(self,cube,move):
        move_dir = self.move_data[move.axis][move.index]
        if move_dir == 0 or (move_dir == 1 and move.clockwise) or (move_dir == -1 and not move.clockwise):
            cube.rotate(move)
            if move.clockwise:
                self.move_data[move.axis][move.index]+=1
            else:
                self.move_data[move.axis][move.index]-=1
            for key,value in self.move_data.items():
                if key != move.axis:
                    for i in range(len(value)):
                        value[i] = 0
            return True
        else:
            return False

 
   
        
class CubeCreater():
    def __init__(self,level):
        self.level = level
        self.move_data={"x":[0]*(level-1),"y":[0]*(level-1),"z":[0]*(level-1)}
        
        self.moves=[]
        for index in range(0,level-1):
            for axis in ["x","y","z"]:
                for clockwise in [True,False]:
                    self.moves.append(CubeMove(axis,index, clockwise))
        

    def create(self,depth):
        self.__clear_move_data()
        cube = Cube(self.level)
        for step in range(depth):
            move = self.moves[random.randint(0, len(self.moves)-1)]
            while(not self.__try_move(cube,move)):
                move = self.moves[random.randint(0, len(self.moves)-1)]
        return cube


    def __clear_move_data(self):
        for item in self.move_data.values():
            for i in range(len(item)):
                item[i]= 0
                
    def __try_move(self,cube,move):
        move_dir = self.move_data[move.axis][move.index]
        if move_dir == 0 or (move_dir == 1 and move.clockwise) or (move_dir == -1 and not move.clockwise):
            cube.rotate(move)
            if move.clockwise:
                self.move_data[move.axis][move.index]+=1
            else:
                self.move_data[move.axis][move.index]-=1
            for key,value in self.move_data.items():
                if key != move.axis:
                    for i in range(len(value)):
                        value[i] = 0
            return True
        else:
            return False

 
   
            
    