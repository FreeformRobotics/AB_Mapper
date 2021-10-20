'''
@Author: Zuxin Liu
@Email: zuxinl@andrew.cmu.edu
@Date:   2020-02-16 22:49:21
@LastEditTime: 2020-07-27 11:40:00
@Description:
'''

import numpy as np
from heapq import heappush, heappop

class Node(object):
    """Searching node"""
    def __init__(self, pose, g_value, h_value):
        self.pose = pose
        self.x = pose[0]
        self.y = pose[1]
        self.g_value = g_value
        self.f_value = g_value + h_value
        self.father = None

    def __lt__(self, b):
            # Heap will always pop the item with smallest f_value
        return self.f_value < b.f_value

class A_star():

    def __init__(self, map_, idx_to_object):
        '''
          @param [array] map : static map with obstacle information.
        '''
        self.map = map_
        self.row = None
        self.col = None
        self.open_list = []
        self.cost_map = None 
        self.closed_map = None
        self.path = []
        self.idx_to_object = idx_to_object
        self.object_to_idx = dict(zip(self.idx_to_object.values(), self.idx_to_object.keys()))
        self.object_to_cost = {
          "free"     : 0,
          "obstacle" : 10,
          "agent"    : 15,
          "dynamic obstacle" : 20,
          "unseen"   : 10,
          "goal"     : 0,
      }
        self.reset()

    def reset(self):
        self.open_list = []
        self.compute_cost_map(self.map)
        #Set the obstacle's pose in closed_map to 1 because we do not want to expand it.
        self.closed_map = np.zeros(self.map.shape)
        self.closed_map[self.map==self.object_to_idx["obstacle"]] = 1
        self.closed_map[self.map==self.object_to_idx["agent"]] = 1
        self.closed_map[self.map==self.object_to_idx["dynamic obstacle"]] = 1
        
    def update_map(self, map_):
        self.map = map_

    def compute_cost_map(self, map_):
        self.row = map_.shape[0]
        self.col = map_.shape[1]
        self.cost_map = np.zeros(map_.shape)
        for idx in self.idx_to_object:
            self.cost_map[map_==idx] = self.object_to_cost[self.idx_to_object[idx]]

    def plan(self, pose, goal, debug = False):
        '''
          @param [2x1 array] pose & goal
        '''
        if not self.valid_goal(goal):
            return []
        self.reset()
        start_node = Node(pose, g_value=0, h_value=self.eucilidean_distance(pose, goal))
        heappush(self.open_list, start_node)
        while self.open_list:
            current_node = heappop(self.open_list)
            if debug:
                print("******************************")
                print("current node: ", current_node.pose)
            if self.goal_reached(current_node.pose, goal):
                self.path = self.reconstruct_path(current_node)
                self.reset()
                return self.path
            self.closed_map[current_node.y, current_node.x] = -1

            successor_list = self.get_successor(current_node, goal)
            if debug:
                print("successor: ", [node.pose for node in successor_list])
                print("f_value: ", [node.f_value for node in successor_list])
            if not successor_list:
                continue
            for node in successor_list:
                node.father = current_node
                heappush(self.open_list, node)
        #print("can not find a path")
        self.reset()
        return []

    #def estimate_heuristic(self, start, end):
    #    return max(abs(start[0]-end[0]), abs(start[1]-end[1]))

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
            print("Goal is occupied by static obstacle!")
            return False
        # If the goal is occupied by dynamic obstacle, set it free
        self.map[y,x]=self.object_to_idx["free"]
        #self.cost_map[y,x] = self.object_to_cost["free"]
        return True
        
    def reconstruct_path(self, node):
        traj = []
        traj.append(node.pose)
        current_node = node
        while current_node.father:
            current_node = current_node.father
            traj.append(current_node.pose)
        traj.reverse()
        return traj

    def get_successor(self, node, goal, mode = 8):
        # return a list of node successors
        x_ = node.x
        y_ = node.y
        heuristic = self.manhattan_distance
        pose_list = [(x_+1,y_), (x_-1,y_), (x_, y_+1), (x_,y_-1)]
        if mode==8:
        	pose_list = [ (x_+1,y_), (x_-1,y_), (x_, y_+1), (x_,y_-1) ,(x_+1,y_+1), (x_-1,y_-1), (x_-1, y_+1), (x_+1,y_-1) ]
        	heuristic = self.diagonal_distance
        successor_list = []
        for x, y in pose_list:
            if 0<=x<self.col and 0<=y< self.row and self.closed_map[y,x]==0:
                h_value = heuristic([x, y],goal)
                g_value = node.g_value+self.eucilidean_distance([x_,y_],[x,y])+self.cost_map[y,x]
                new_node = Node(np.array([x,y]), g_value, h_value)
                successor_list.append(new_node)
                self.closed_map[y,x]=-1
        return successor_list
    
    def render_path(self, cost_map, path):
        cmap = cost_map.copy()
        for pose in path:
            cmap[pose[1],pose[0]] = 1
        print(cmap)
        return cmap
