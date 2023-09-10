from searchGeneric import *


class BidirectionalSearch(Searcher):
    def __init__(self, problem):
        self.problem = problem
        self.close_list =[]
        self.forward_frontier = [Path(problem.start_node())]
        self.backward_frontier = [Path(problem.goal_node())]
        self.num_expanded = 0
        self.size = 0
            
    def add_to_frontier(self,is_forward, path):
        
        if is_forward:
            self.forward_frontier.append(path)
        else:
            self.backward_frontier.append(path)
    
    def empty_frontier(self):
        return len(self.forward_frontier) == 0 or len(self.backward_frontier) == 0
        

    def search(self,time_limit= float("inf")):
        self.startTime = time.time()
        self.time_limit =time_limit
        is_forward = True
        while not self.empty_frontier():  
            result = self.search_one_side(is_forward)
            if(result != None):
                self.solution = result
                return result
            is_forward= not is_forward
            if(time.time()-self.startTime > time_limit):
                break
        
                  
    def search_one_side(self, is_forward):
        if is_forward:
            expand_list = self.forward_frontier
            compare_list = self.backward_frontier
        else: 
            expand_list = self.backward_frontier
            compare_list = self.forward_frontier
        
        if  self.size <= 0:
            self.size = len(expand_list)
            
        while self.size > 0:
            self.size -= 1
            path =expand_list.pop(0)
            self.close_list.append(path.end())
            self.num_expanded += 1
            for compare_path in compare_list:
                if compare_path.end() == path.end():
                    print( self.num_expanded, "paths have been expanded and",len(expand_list)+len(compare_list), "paths remain in the frontier")
                    if is_forward:
                        return self.generate_result(path, compare_path)
                    else:
                        return self.generate_result(compare_path, path)
                    
            neighs = self.problem.neighbors(path.end())
            for neigh in neighs:
                if neigh.to_node not in self.close_list:
                    self.add_to_frontier(is_forward,Path(path,neigh))  
            if(time.time()-self.startTime > self.time_limit ):
                break

                    
    def generate_result(self,start_path, end_path):
        while end_path.arc!= None:
            arc = end_path.arc
            new_arc =Arc(arc.to_node,arc.from_node,arc.cost,arc.action.get_reverse())
            start_path = Path(start_path,new_arc)
            end_path = end_path.initial
        return start_path
            
        
class UniformCostSearcher(AStarSearcher):
    """returns a greedy searcher for a problem.
       Paths can be found by repeatedly calling search().
    """
    def __init__(self, problem):
        super().__init__(problem)

    def add_to_frontier(self, path):
        """add path to the frontier with the appropriate cost"""
        value = path.cost
        self.frontier.add(path, value)
        return