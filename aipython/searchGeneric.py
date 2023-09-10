import time
class Search_problem(object):
    """A search problem consists of:
    * a start node
    * a neighbors function that gives the neighbors of a node
    * a specification of a goal
    * a (optional) heuristic function.
    The methods must be overridden to define a search problem."""

    def start_node(self):
        """returns start node"""
        raise NotImplementedError("start_node")   # abstract method
    
    def is_goal(self,node):
        """is True if node is a goal"""
        raise NotImplementedError("is_goal")   # abstract method

    def neighbors(self,node):
        """returns a list (or enumeration) of the arcs for the neighbors of node"""
        raise NotImplementedError("neighbors")   # abstract method

    def heuristic(self,n):
        """Gives the heuristic value of node n.
        Returns 0 if not overridden."""
        return 0

class Arc(object):
    """An arc has a from_node and a to_node node and a (non-negative) cost"""
    def __init__(self, from_node, to_node, cost=1, action=None):
        self.from_node = from_node
        self.to_node = to_node
        self.action = action
        self.cost=cost
        assert cost >= 0, (f"Cost cannot be negative: {self}, cost={cost}")

    def __repr__(self):
        """string representation of an arc"""
        if self.action:
            return f"{self.from_node} --{self.action}--> {self.to_node}"
        else:
            return f"{self.from_node} --> {self.to_node}"

class Search_problem_from_explicit_graph(Search_problem):
    """A search problem consists of:
    * a list or set of nodes
    * a list or set of arcs
    * a start node
    * a list or set of goal nodes
    * a dictionary that maps each node into its heuristic value.
    * a dictionary that maps each node into its (x,y) position
    """

    def __init__(self, nodes, arcs, start=None, goals=set(), hmap={}, positions={}):
        self.neighs = {}
        self.nodes = nodes
        for node in nodes:
            self.neighs[node]=[]
        self.arcs = arcs
        for arc in arcs:
            self.neighs[arc.from_node].append(arc)
        self.start = start
        self.goals = goals
        self.hmap = hmap
        self.positions = positions

    def start_node(self):
        """returns start node"""
        return self.start
    
    def is_goal(self,node):
        """is True if node is a goal"""
        return node in self.goals

    def neighbors(self,node):
        """returns the neighbors of node (a list of arcs)"""
        return self.neighs[node]

    def heuristic(self,node):
        """Gives the heuristic value of node n.
        Returns 0 if not overridden in the hmap."""
        if node in self.hmap:
            return self.hmap[node]
        else:
            return 0
        
    def __repr__(self):
        """returns a string representation of the search problem"""
        res=""
        for arc in self.arcs:
            res += f"{arc}.  "
        return res

class Path(object):
    """A path is either a node or a path followed by an arc"""
    
    def __init__(self,initial,arc=None):
        """initial is either a node (in which case arc is None) or
        a path (in which case arc is an object of type Arc)"""
        self.initial = initial
        self.arc=arc
        if arc is None:
            self.cost=0
        else:
            self.cost = initial.cost+arc.cost

    def end(self):
        """returns the node at the end of the path"""
        if self.arc is None:
            return self.initial
        else:
            return self.arc.to_node

    def nodes(self):
        """enumerates the nodes for the path.
        This enumerates the nodes in the path from the last elements backwards.
        """
        current = self
        while current.arc is not None:
            yield current.arc.to_node
            current = current.initial
        yield current.initial

    def initial_nodes(self):
        """enumerates the nodes for the path before the end node.
        This calls nodes() for the initial part of the path.
        """
        if self.arc is not None:
            yield from self.initial.nodes()
        
    def __repr__(self):
        """returns a string representation of a path"""
        if self.arc is None:
            return str(self.initial)
        elif self.arc.action:
            return f"{self.initial}\n   --{self.arc.action}--> {self.arc.to_node}"
        else:
            return f"{self.initial} --> {self.arc.to_node}"

problem1 = Search_problem_from_explicit_graph(
    {'A','B','C','D','G'},
    [Arc('A','B',3), Arc('A','C',1), Arc('B','D',1), Arc('B','G',3),
         Arc('C','B',1), Arc('C','D',3), Arc('D','G',1)],
    start = 'A',
    goals = {'G'},
    positions={'A': (0, 2), 'B': (1, 1), 'C': (0,1), 'D': (1,0), 'G': (2,0)})
problem2 = Search_problem_from_explicit_graph(
    {'a','b','c','d','e','g','h','j'},
    [Arc('a','b',1), Arc('b','c',3), Arc('b','d',1), Arc('d','e',3),
        Arc('d','g',1), Arc('a','h',3), Arc('h','j',1)],
    start = 'a',
    goals = {'g'},
    positions={'a': (0, 0), 'b': (0, 1), 'c': (0,4), 'd': (1,1), 'e': (1,4),
                   'g': (2,1), 'h': (3,0), 'j': (3,1)})

problem3 = Search_problem_from_explicit_graph(
    {'a','b','c','d','e','g','h','j'},
    [],
    start = 'g',
    goals = {'k','g'})

simp_delivery_graph = Search_problem_from_explicit_graph(
    {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J'},
    [    Arc('A', 'B', 2),
         Arc('A', 'C', 3),
         Arc('A', 'D', 4),
         Arc('B', 'E', 2),
         Arc('B', 'F', 3),
         Arc('C', 'J', 7),
         Arc('D', 'H', 4),
         Arc('F', 'D', 2),
         Arc('H', 'G', 3),
         Arc('J', 'G', 4)],
   start = 'A',
   goals = {'G'},
   hmap = {
        'A': 7,
        'B': 5,
        'C': 9,
        'D': 6,
        'E': 3,
        'F': 5,
        'G': 0,
        'H': 3,
        'J': 4,
    })
cyclic_simp_delivery_graph = Search_problem_from_explicit_graph(
    {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J'},
    [    Arc('A', 'B', 2),
         Arc('A', 'C', 3),
         Arc('A', 'D', 4),
         Arc('B', 'A', 2),
         Arc('B', 'E', 2),
         Arc('B', 'F', 3),
         Arc('C', 'A', 3),
         Arc('C', 'J', 7),
         Arc('D', 'A', 4),
         Arc('D', 'H', 4),
         Arc('F', 'B', 3),
         Arc('F', 'D', 2),
         Arc('G', 'H', 3),
         Arc('G', 'J', 4),
         Arc('H', 'D', 4),
         Arc('H', 'G', 3),
         Arc('J', 'C', 6),
         Arc('J', 'G', 4)],
   start = 'A',
   goals = {'G'},
   hmap = {
        'A': 7,
        'B': 5,
        'C': 9,
        'D': 6,
        'E': 3,
        'F': 5,
        'G': 0,
        'H': 3,
        'J': 4,
    })



# searchGeneric.py - Generic Searcher, including depth-first and A*
# AIFCA Python3 code Version 0.9.7 Documentation at http://aipython.org
# Download the zip file and read aipython.pdf for documentation

# Artificial Intelligence: Foundations of Computational Agents http://artint.info
# Copyright 2017-2023 David L Poole and Alan K Mackworth
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# See: http://creativecommons.org/licenses/by-nc-sa/4.0/deed.en


class Searcher():
    """returns a searcher for a problem.
    Paths can be found by repeatedly calling search().
    This does depth-first search unless overridden
    """
    def __init__(self, problem):
        """creates a searcher from a problem
        """
        self.problem = problem
        self.initialize_frontier()
        self.num_expanded = 0
        self.add_to_frontier(Path(problem.start_node()))
        super().__init__()

    def initialize_frontier(self):
        self.frontier = []
        
    def empty_frontier(self):
        return self.frontier == []
        
    def add_to_frontier(self,path):
        self.frontier.append(path)
        

    def search(self):
        """returns (next) path from the problem's start node
        to a goal node. 
        Returns None if no path exists.
        """
        while not self.empty_frontier():
            path = self.frontier.pop()
            self.num_expanded += 1
            if self.problem.is_goal(path.end()):    # solution found
                print(self.num_expanded, "paths have been expanded and",
                            len(self.frontier), "paths remain in the frontier")
                self.solution = path   # store the solution found
                return path
            else:
                neighs = self.problem.neighbors(path.end())
                for arc in reversed(list(neighs)):
                    self.add_to_frontier(Path(path,arc))
        print("No (more) solutions. Total of",
                     self.num_expanded,"paths expanded.")

# Depth-first search for problem1; do the following:
# searcher1 = Searcher(searchProblem.problem1)
# searcher1.search()  # find first solution
# searcher1.search()  # find next solution (repeat until no solutions)
# searcher_sdg = Searcher(searchProblem.simp_delivery_graph)
# searcher_sdg.search()  # find first or next solution

import heapq        # part of the Python standard library

class FrontierPQ(object):
    """A frontier consists of a priority queue (heap), frontierpq, of
        (value, index, path) triples, where
    * value is the value we want to minimize (e.g., path cost + h).
    * index is a unique index for each element
    * path is the path on the queue
    Note that the priority queue always returns the smallest element.
    """

    def __init__(self):
        """constructs the frontier, initially an empty priority queue 
        """
        self.frontier_index = 0  # the number of items added to the frontier
        self.frontierpq = []  # the frontier priority queue

    def empty(self):
        """is True if the priority queue is empty"""
        return self.frontierpq == []

    def add(self, path, value):
        """add a path to the priority queue
        value is the value to be minimized"""
        self.frontier_index += 1    # get a new unique index
        heapq.heappush(self.frontierpq,(value, -self.frontier_index, path))

    def pop(self):
        """returns and removes the path of the frontier with minimum value.
        """
        (_,_,path) = heapq.heappop(self.frontierpq)
        return path 

    def count(self,val):
        """returns the number of elements of the frontier with value=val"""
        return sum(1 for e in self.frontierpq if e[0]==val)

    def __repr__(self):
        """string representation of the frontier"""
        return str([(n,c,str(p)) for (n,c,p) in self.frontierpq])
    
    def __len__(self):
        """length of the frontier"""
        return len(self.frontierpq)

    def __iter__(self):
        """iterate through the paths in the frontier"""
        for (_,_,path) in self.frontierpq:
            yield path
    
class AStarSearcher(Searcher):
    """returns a searcher for a problem.
    Paths can be found by repeatedly calling search().
    """

    def __init__(self, problem):
        super().__init__(problem)

    def initialize_frontier(self):
        self.frontier = FrontierPQ()

    def empty_frontier(self):
        return self.frontier.empty()

    def add_to_frontier(self,path):
        """add path to the frontier with the appropriate cost"""
        value = path.cost+self.problem.heuristic(path.end())
        self.frontier.add(path, value)


