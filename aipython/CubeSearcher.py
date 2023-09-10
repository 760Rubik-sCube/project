from searchGeneric import *
import time


class CubeSearcher(AStarSearcher):
    
    def __init__(self, problem, weight):
        super().__init__(problem)
        self.weight = weight
    
    def add_to_frontier(self,path):
        """add path to the frontier with the appropriate cost"""
        value = path.cost * 0.3 + self.problem.heuristic(path.end())
        self.frontier.add(path, value)
        
    def search(self,time_limit= float("inf")):
        startTime = time.time()
        while not self.empty_frontier():
            path = self.frontier.pop()
            self.num_expanded += 1
            neighs = self.problem.neighbors(path.end())
            for arc in reversed(list(neighs)):
                new_path =Path(path,arc)
                if self.problem.is_goal(new_path.end()):    # solution found
                    print( self.num_expanded, "paths have been expanded and",len(self.frontier), "paths remain in the frontier")
                    self.solution = new_path   # store the solution found
                    return new_path
                else:
                    self.add_to_frontier(new_path)
            
            if(time.time()-startTime > time_limit):
                break

        print("No (more) solutions. Total of",self.num_expanded,"paths expanded.")