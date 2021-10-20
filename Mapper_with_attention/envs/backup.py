"""
Created on Wed Feb  5 20:42:28 2020

@author: lance
reference: https://github.com/samdjstephens/pydstarlite
"""
import numpy as np
import heapq
from collections import deque
from functools import partial

class PriorityQueue:
    def __init__(self):
        self.elements = []

    def empty(self):
        return len(self.elements) == 0

    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))

    def pop(self):
        item = heapq.heappop(self.elements)
        return item[1]

    def first_key(self):
        return heapq.nsmallest(1, self.elements)[0][0]

    def delete(self, node):
        self.elements = [e for e in self.elements if e[1] != node]
        heapq.heapify(self.elements)

    def __iter__(self):
        for key, node in self.elements:
            yield node

class D_star():
    def __init__(self, map_, idx_to_object):
        '''
          @param [array] map : static map with obstacle information.
        '''
        self.map = map_
        self.row = None
        self.col = None
        self.cost_map = None 

        self.idx_to_object = idx_to_object
        self.object_to_idx = dict(zip(self.idx_to_object.values(), self.idx_to_object.keys()))
        self.object_to_cost = {
          "free"     : 0,
          "obstacle" : float('inf'),
          "agent"    : 500,
          "dynamic obstacle" : 20,
          "unseen"   : 10,
          "goal"     : 0,
      }

        self.reset()

    def trans_cost(self, neighbour, node): ## problem here
        if self.cost_map[neighbour[1]][neighbour[0]] > 0 or self.cost_map[node[1]][node[0]]  > 0:
            return  self.cost_map[neighbour[1]][neighbour[0]]   #max(self.cost_map[neighbour[1]][neighbour[0]], self.cost_map[node[1]][node[0]] )
        else:
            return self.eucilidean_distance(neighbour, node) #1

    def lookahead_cost(self, node, neighbour):
        return self.g(neighbour) + self.trans_cost(neighbour, node)#problem

    def lowest_cost_neighbour(self, node):
        cost = partial(self.lookahead_cost, node)
        return min(self.get_neighbors(node), key=cost) 

    def g(self, node):
        return self.G_VALS.get(node, float('inf'))

    def rhs(self, node):
        return self.RHS_VALS.get(node, float('inf')) if node != self.goal else 0

    def calculate_rhs(self, node):
        lowest_cost_neighbour = self.lowest_cost_neighbour(node)
        self.back_pointers[node] = lowest_cost_neighbour
        return self.lookahead_cost(node, lowest_cost_neighbour)

    def calculate_key(self, node):
        '''
        node: tuple(x,y) /(row,column)
        '''
        g_rhs = min([self.g(node), self.rhs(node)])

        return (
            g_rhs + self.heuristic(node, self.position) + self.Km,
            g_rhs
        )

    def update_node(self, node):
        if node != self.goal:
            self.RHS_VALS[node] = self.calculate_rhs(node)
        self.frontier.delete(node)
        if self.g(node) != self.rhs(node):
            self.frontier.put(node, self.calculate_key(node))

    def update_nodes(self, nodes):
        [self.update_node(n) for n in nodes]

    def in_bounds(self, pos):
        (x, y) = pos
        return 0 <= x < self.col and 0 <= y < self.row

    def get_neighbors(self, pos):
        (x_, y_) = pos
        #results = [(x_+1,y_), (x_-1,y_), (x_, y_+1), (x_,y_-1)]
        results = [(x_+1,y_), (x_-1,y_), (x_, y_+1), (x_,y_-1), (x_+1,y_+1), (x_-1,y_+1), (x_+1, y_-1), (x_-1,y_-1)]
        results = filter(self.in_bounds, results)
        return list(results)

    def compute_shortest_path(self):
        #print("computing")
        l#ast_nodes = deque(maxlen=10)
        #len(self.frontier.elements) and 
        while not self.frontier.empty() and (self.frontier.first_key() < self.calculate_key(self.position) or self.rhs(self.position) != self.g(self.position)):
            k_old = self.frontier.first_key()
            node = self.frontier.pop()


            #loop check
#            last_nodes.append(node)
#            if len(last_nodes) == 10 and len(set(last_nodes)) < 3:
#                raise Exception("Fail! Stuck in a loop")


            k_new = self.calculate_key(node)
            if k_old < k_new:
                self.frontier.put(node, k_new)
            elif self.g(node) > self.rhs(node):
                self.G_VALS[node] = self.rhs(node)
                self.update_nodes(self.get_neighbors(node))
            else:
                self.G_VALS[node] = float('inf')
                self.update_nodes(self.get_neighbors(node) + [node])

        #print(self.back_pointers)
        self.path = self.reconstruct_path(self.position, self.goal)
        return self.back_pointers.copy(), self.G_VALS.copy()

    def plan(self, pose, goal, obs = None, debug = False):
        #print(obs)
        '''
        pose, goal are (2,) ndarray
        obs 2d array
        '''
        #draw itself
        #self.map[pose[1], pose[0]] = 2
        #update map
        '''
        if self.position != None:
            print(self.position)
            self.map[self.position[1]][self.position[0]] = 0
        self.map[pose[1]][pose[0]] = 3
        '''

        self.position = tuple(pose.tolist()) 
        self.goal = tuple(goal.tolist())
        update_obj = None

        if self.path is None: # first run compute shortest path or when last run cannot find solution
            #print("no path")
            if obs is not None:
                update_obj = self.map_changed(obs, self.position)
                if update_obj: self.update_cost_map(update_obj)

            if not self.valid_goal(goal):
                return None

            self.frontier.put(self.goal, self.calculate_key(self.goal))
            self.back_pointers[self.goal] = None
            self.last_node = self.position
            self.compute_shortest_path()

        else:
            #new observations changes the current map
            #print("check replan")
            update_obj = self.map_changed(obs, self.position)
            if update_obj: # found new obstacle 
                self.update_cost_map(update_obj)
                self.update_nodes({node for obj in update_obj
                   for node in self.get_neighbors((obj[1],obj[0])) })
                #print("update,node: ",{node for obj in update_obj
                #   for node in self.get_neighbors((obj[1],obj[0])) })
            else: # no new obstacle found, simply one step forward
                if not np.array_equal(self.path[0], self.position):  #environment tells the agent not to move
                    self.path = self.path[1:]
                    print("skipp")
            print(self.cost_map)
            if np.array_equal(self.path[0], self.position): #self.obs_on_path(update_obj) or 
                # need replan when object didn't move last step(environment does not allow it to move)
            #if self.obs_on_path(update_obj):
                if not self.valid_goal(goal):
                    return None

                print("replanning")
                self.Km += self.heuristic(self.last_node, self.position)
                self.last_node = self.position

                self.compute_shortest_path()
            else:
                self.path = self.path[1:]
        #print(self.path)
        print(self.path)
        if debug:
            print("******************************")
            print("new obs", len(update_obj) != 0)
            print(self.position)
            print(update_obj)
            #print(self.map)
            print(self.cost_map)
            print(self.path)

        return self.path

    def obs_on_path(self, new_obs):
        '''
        new_obs: a list of 3 tuple (x, y, type) of discoved obs
        '''
        for obs in new_obs:
            if (obs[2] in (1,2,3)) and any(np.array_equal( np.array([obs[1], obs[0]]), x) for x in self.path):
                return True
        return False

    def update_cost_map(self, new_obs):
        #print("cost map")
        for obj in new_obs:
            self.cost_map[obj[0]][obj[1]] = self.object_to_cost[self.idx_to_object[obj[2]]]
        #print(self.cost_map)

    def map_changed(self, obs, pos):
        if obs is None: return []

        visibility = int((len(obs) - 1)/2)

        y,x = pos
        xmin = max(int(x-visibility), 0)
        ymin = max(int(y-visibility), 0)
        xmax = min(int(x+1+visibility), self.row)
        ymax = min(int(y+1+visibility), self.col)

        current_map = self.map[xmin:xmax, ymin:ymax]

        height = xmax-xmin
        width = ymax-ymin
        obs_size = 2*visibility + 1
        obs_x_min, obs_x_max, obs_y_max, obs_y_min = (0, obs_size, obs_size, 0)
        x_offset = 0
        y_offset = 0

        if x-visibility<0:
            obs_x_min = obs_size - height
            obs_x_max = obs_size
            x_offset = 0
        elif x+1+visibility>self.row:
            obs_x_min = 0
            obs_x_max = height
            x_offset = x-visibility
        else:
            x_offset = x-visibility

        if y-visibility<0:
            obs_y_min = obs_size - width
            obs_y_max = obs_size
            y_offset = 0
        elif y+1+visibility>self.col:
            obs_y_min = 0
            obs_y_max = width
            y_offset = y-visibility
        else:
            y_offset = y-visibility
             
        cropped_obs = obs[obs_x_min:obs_x_max,obs_y_min:obs_y_max]
        #print("offset", x_offset, y_offset)
        update_obj = [(i+x_offset, j+y_offset, cropped_obs[i][j]) for i in range(0,len(current_map))
                        for j in range(0, len(current_map[1]))
                        if current_map[i][j] != cropped_obs[i][j]   ]

        #print(self.map[xmin:xmax, ymin:ymax], cropped_obs)
        if update_obj:
            self.map[xmin:xmax, ymin:ymax] = cropped_obs

        return update_obj


    def reconstruct_path(self, start, goal):
        """Reconstruct a shortest path from a dictionary of back-pointers"""
        traj = [np.array(start)]
        cur = start
        while (cur != goal):
            if self.g(cur) == float('inf'):
                print("can not find a path")
                self.reset()
                return None
            cur = self.back_pointers[cur]
            traj.append(np.array(cur))
        #print("Path Found !!")
        return traj
    '''
    def reconstruct_path(self, start, goal):
        """Reconstruct a shortest path from a dictionary of back-pointers"""
        traj = [start]
        cur = start
        while (cur != goal):
            if self.g(cur) == float('inf'):
                print("can not find a path")
                return None
            cur = self.lowest_cost_neighbour(cur)
            traj.append(cur)
        return traj
    '''
    def heuristic(self, a, b):
        return self.diagonal_distance(a ,b)

    def diagonal_distance(self, start, end):
        return max(abs(start[0]-end[0]), abs(start[1]-end[1]))

    def manhattan_distance(self, start, end):
        return (abs(start[0]-end[0]) + abs(start[1]-end[1]))

    def eucilidean_distance(self, start, end):
        return ( (start[0]-end[0])**2+(start[1]-end[1])**2 )**0.5

    def goal_reached(self, pose, goal):
        return (pose==goal).all()
    
    def valid_goal(self, goal):
        x = goal[0]
        y = goal[1]
        if x<0 or x>=self.col or y<0 or y>=self.row:
            print("Goal is out of map")
            return False
        if self.map[y,x]==self.object_to_idx["obstacle"]:
            #print(self.map[y,x])
            #print(self.map)
            print("Goal is occupied!")
            return False
        return True
    
    def render_path(self, cost_map, path):
        cmap = cost_map.copy()
        for pose in path:
            cmap[pose[1],pose[0]] = 1
        #print(cmap)
        return cmap

    def reset(self):
        self.compute_cost_map(self.map)

      #dstar initialize
        self.back_pointers = {}
        self.G_VALS = {}
        self.RHS_VALS = {}
        self.Km = 0
        self.position = None
        self.last_node = None
        self.goal = None
        self.path = None
        self.frontier = PriorityQueue()
        
    def set_map(self, map_):
        self.map = map_

    def compute_cost_map(self, map_):
        self.row = map_.shape[0]
        self.col = map_.shape[1]
        self.cost_map = np.zeros(map_.shape)
        for idx in self.idx_to_object:
            self.cost_map[map_==idx] = self.object_to_cost[self.idx_to_object[idx]]
