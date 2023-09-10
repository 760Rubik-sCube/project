class CubeMove():
    def __init__(self,axis,index,clockwise):
        self.axis = axis #x,y,z
        self.index = index # 1~(level-1) -- don't include 0 because we want block(0,0,0) always never change its position
        self.clockwise = clockwise # Boolean
        
    def __eq__(self, other):
        if other ==None:
            return False
        return self.axis == other.axis and self.index== other.index and self.clockwise == other.clockwise 
    
    def __str__(self):
        output = f"({self.axis}{self.index}"
        output+=")" if self.clockwise else "\')"
        return output 
    

           

    def is_reverse(self,other):
        if other ==None:
            return False
        return self.axis == other.axis and self.index== other.index and self.clockwise != other.clockwise
    
    def set_reverse(self):
        self.clockwise = not self.clockwise
        
    def get_reverse(self):
        return CubeMove(self.axis,self.index,not self.clockwise)
    
    

class Cube(): 
    """
    allowed init input value 
        level of cube
        string cube data
        3D list data of cube
    """
    def __init__(self,level):  
        self.level = level
        self.biside =[0,level-1]
        self.cube =[[[color for x in range(level)] for y in range(level)] for color in range(6)]

    def __eq__(self, other):
        return self.cube == other.cube
    
    def serialize_data(self):
        data=""
        for i in range(6):
            for x in range(self.level):
                for y in range(self.level):
                    data+= str(self.cube[i][x][y])
        return data 

    def __str__(self):
        output = "\n"
        level = self.level 
        for y in range(self.level):
            output += " "*(2*self.level +1)
            for x in range(self.level):
                output += f"{self.cube[0][x][y]} "
            output += "\n"
        
        for y in range(self.level):
            for i in range(1,5):
                for x in range(self.level):
                    output += f"{self.cube[i][x][y]} "
                output += " "
            output +="\n"
        
        for y in range(self.level):
            output += " "*(2*self.level+1)
            for x in range(self.level):
                output += f"{self.cube[5][x][y]} "
            output+="\n"
        return output
    

    def is_solve(self):
        cube = self.cube
        for i in range(6):
            basic_color = cube[i][0][0]
            for x in range(self.level):
                for y in range(self.level):
                    if basic_color != cube[i][x][y]:
                        return False
        return True
    
    def rotate(self,move):
        self.__rotate_face(move)
        self.__rotate_side(move)
    
    def __rotate_side(self,move):
        cube =self.cube
        level =self.level
        i = move.index
        if move.axis == "y":
            if move.clockwise:
                for x in range(level):
                    cube[1][x][i],cube[2][x][i],cube[3][x][i],cube[4][x][i]=\
                    cube[2][x][i],cube[3][x][i],cube[4][x][i],cube[1][x][i]  
            else:
                for x in range(level):
                    cube[2][x][i],cube[3][x][i],cube[4][x][i],cube[1][x][i]=\
                    cube[1][x][i],cube[2][x][i],cube[3][x][i],cube[4][x][i]     
  
        elif move.axis =="x":
            if move.clockwise:  
                for x in range(level):
                    cube[0][i][x],cube[2][i][x],cube[5][i][x],cube[4][level-1-i][level-1-x]=\
                    cube[4][level-1-i][level-1-x],cube[0][i][x],cube[2][i][x],cube[5][i][x]
            else:
                for x in range(level):
                    cube[4][level-1-i][level-1-x],cube[0][i][x],cube[2][i][x],cube[5][i][x]=\
                    cube[0][i][x],cube[2][i][x],cube[5][i][x],cube[4][level-1-i][level-1-x]
        else: #axis = z
            if move.clockwise:
                for o in range(level):
                    cube[0][o][level-1-i],cube[3][i][o],cube[5][level-1-o][i],cube[1][level-1-i][level-1-o]=\
                    cube[1][level-1-i][level-1-o], cube[0][o][level-1-i],cube[3][i][o],cube[5][level-1-o][i]
            else:
                for o in range(level):
                    cube[1][level-1-i][level-1-o], cube[0][o][level-1-i],cube[3][i][o],cube[5][level-1-o][i]=\
                    cube[0][o][level-1-i],cube[3][i][o],cube[5][level-1-o][i],cube[1][level-1-i][level-1-o]
                    
               
    def __rotate_face(self,move):
        face= None
        toward_face = None #is this face towards view point
        if move.index == 0:
            toward_face = True
            if move.axis == "x":
                face = self.cube[1]
            elif move.axis =="y":
                face = self.cube[0]
            else: 
                face = self.cube[2]
        elif move.index == self.level-1: # is face is at back
            toward_face = False
            if move.axis == "x":
                face = self.cube[3]
            elif move.axis == "y":
                face = self.cube[5]
            else: 
                face = self.cube[4]       
        if face == None:
            return 
        clockwise = move.clockwise 
        if not toward_face:
            clockwise = not clockwise
        if clockwise:
            self.__clockwise_rotate_face(face)
        else:
            self.__anti_clockwise_rotate_face(face)
            
    def __anti_clockwise_rotate_face(self,face):
        max_i = self.level-1
        for a in range(self.level//2):
            for b in range((self.level+1)//2):
                face[a][b],face[b][max_i-a],face[max_i-a][max_i-b],face[max_i-b][a] = \
                face[max_i-b][a],face[a][b],face[b][max_i-a],face[max_i-a][max_i-b]  

    def __clockwise_rotate_face(self,face):
        max_i = self.level-1
        for a in range(self.level//2):
            for b in range((self.level+1)//2):
                face[max_i-b][a],face[a][b],face[b][max_i-a],face[max_i-a][max_i-b] = \
                face[a][b],face[b][max_i-a],face[max_i-a][max_i-b],face[max_i-b][a]



             
    def train_data(self):
        cube = self.cube
        level = self.level 
        max_i = level-1
        biside =self.biside
        corner=[]
        for i in range(6):
            for x in biside:
                for y in biside:
                    corner.append(cube[i][x][y])
               
        side_i=[]
        for x in range(1,level-1):
            side = []
            for y in biside:
                for i in range(6):
                    side.append(cube[i][y][x])
                    side.append(cube[i][x][y])
            side_i.append(side)
          
        face_i=[]
        for x in range(1,max_i):
            for y in range(1,max_i):
                face =[]
                for i in range(6):
                    face.append(cube[i][x][y])
                face_i.append(face)

        return corner,side_i,face_i


    
