'''
@Author: Zuxin Liu
@Email: zuxinl@andrew.cmu.edu
@Date:   2020-02-26 18:25:26
@LastEditTime: 2020-03-25 22:40:08
@Description:
'''

import yaml
import matplotlib.pyplot as plt
import numpy as np
import argparse
import math
from envs import grid_env
import time
from envs.rendering import Window
from random import sample
import pickle
import copy
from envs.astar import A_star
from envs.dstar_lite import D_star
import torch
from torch import optim
import torch.nn.functional as F
from torch.distributions import Categorical
from model1 import ActorCritic


idx_to_act = {0:"N",1:"S",2:"E",3:"W", 4:"NW",5:"WS",6:"SE",7:"EN",8:"."}
act_to_idx = dict(zip(idx_to_act.values(),idx_to_act.keys()))

def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        # print('rewards,',rewards,'len',len(rewards))
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns

class Agent(object):
    def __init__(self, map, ID=0, vis = 7):
        self.map = map
        self.id = ID
        self.vis = vis
        self.row, self.col = 2*self.vis+1, 2*self.vis+1
        self.obs_map = np.ones((3, self.row,self.col))
        self.object_to_idx = {"obstacle":1,"agent":2, "dynamic obstacle":3, "free":0}
        self.idx_to_object = dict(zip(self.object_to_idx.values(),self.object_to_idx.keys()))
        self.object_to_color = {"obstacle":-1,"agent":0.5, "dynamic obstacle":0, "free":1}
        self.path_color = 0.2
        self.goal_color = 0.5
        
        self.dynamic_obs_pose = {0:[],1:[],2:[]} # last pose, last last pose, last last last pose
        self.dynamic_obs_decay = {0:-0.8,1:-0.7,2:-0.6}
        self.agent_obs_pose = {0:[],1:[],2:[]}
        self.agent_obs_decay = {0:-0.3,1:-0.2,2:-0.1}
        self.pose_normalizer = 20

        self.planner = A_star(self.map, self.idx_to_object)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device( "cpu")

        self.ac = ActorCritic().to(self.device) #
        self.optimizer = None

        # training variables
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.masks     = []
        self.current_ent = 0
        self.entropy = 0
        self.collision = 0
        self.steps = 0

        self.subgoal_length = 3
        self.max_step = 50
        self.off_route_scale_factor = 0.5 #0.5
        self.poses = []
        self.oscillation_penalty = -0.3

    def reset_memory(self):
        # reset training variables
        self.log_probs = []
        self.values = []
        self.rewards   = []
        self.masks     = []
        self.entropy = 0


    def plan(self, start, goal):
        path = self.planner.plan(start.copy(), goal.copy())
        if len(path):
            self.path = path

    def set_max_step(self, state, ratio = 4):
        pose = state['pose'][self.id].copy()
        goal = state['goal'][self.id].copy()
        self.ratio = ratio
        self.plan(pose, goal)
        self.max_step = int(len(self.path)*ratio)

    def off_route_reward(self, state):
        if not len(self.path):
            return 0
        pose = np.array(state['pose'][self.id])
        #print("pose: ",pose)
        path = np.array(self.path)
        #print("path shape: ", path.shape)
        diff = path-pose
        x = diff[:,0]
        y = diff[:,1]
        distance = np.sqrt(x**2+y**2)
        #print(distance)
        #print(-np.min(distance))
        reward = -np.min(distance)*self.off_route_scale_factor
        return reward

    def compute_reward(self, state):
        pose = np.array(state['pose'][self.id])
        r1 = self.off_route_reward(state)
        if len(self.poses)<2:
            self.poses.append(pose)
            return r1
        pose_last_last = self.poses[1]
        r2 = 0
        if np.all(pose==pose_last_last):
            r2 = self.oscillation_penalty
        self.poses[1] = self.poses[0]
        self.poses[0] = pose
        return r1+r2

    def update_map(self, obs, pose):
        #print("observation: ")
        #print(obs)
        # print("pose: ",pose)
        new_map = self.map.copy()
        offset = np.array([pose[1]-self.vis, pose[0]-self.vis])
        agent_idx = np.argwhere(obs==self.object_to_idx["agent"])
        #print("offset", offset)
        #print("agent in obs", agent_idx)
        if np.size(agent_idx):
            agent_idx = agent_idx + offset
            #print("agent in obs", agent_idx)
            for idx in agent_idx:
                new_map[idx[0],idx[1]] = self.object_to_idx["agent"]
        
        agent_idx = np.argwhere(obs==self.object_to_idx["dynamic obstacle"])
        #print("dynamic obs in obs", agent_idx)
        if np.size(agent_idx):
            agent_idx = agent_idx + offset
            #print("dynamic obs in obs", agent_idx)
            for idx in agent_idx:
                new_map[idx[0],idx[1]] = self.object_to_idx["dynamic obstacle"]
        return new_map

    def return_need_to_attention_index_for_every_agent(self,all_pose,pose,sub_num_agent):
        index=[]
        diction={}
        for one_pose in all_pose:
            dis=np.power(pose[0]-one_pose[0],2)+np.power(pose[1]-one_pose[1],2)
            if dis == 0:
                dis=9999
            index.append(dis)
        for k, v in enumerate(index):
             diction[k] = v
        x=sorted(diction.items(),key=lambda x : x[1])
        y=[]
        for k, v in x[0:sub_num_agent]:
            y.append(k)

        return y #return mix distance's agent's index


    def preprocess(self, state,sub_num_agent, replan = False, debug = False):
        self.obs_map = np.ones((3, self.row,self.col))
        obs = state['obs'][self.id]
        pose = state['pose'][self.id]
        index=self.return_need_to_attention_index_for_every_agent(state['pose'],pose,sub_num_agent)
        # print(state['pose'])
        # print('self.id,pose',self.id,pose)
        goal = state['goal'][self.id]
        # first channel is the obs
        obs_color = copy.deepcopy(obs)
        for key in self.object_to_idx.keys():
            obs_color[obs_color==self.object_to_idx[key]] = self.object_to_color[key]
        self.obs_map[0,:,:] = obs_color
        
        # second channel is the trajectory of dynamic obs and agents
        dynamic_map = self.obs_map[1,:,:]
        mask = (obs==self.object_to_idx["dynamic obstacle"])
        dynamic_map[mask] = self.object_to_color["dynamic obstacle"]
        dynamic_obs_pose_now = np.argwhere(mask)[:,::-1] + pose - np.array([self.vis, self.vis])
        mask = (obs==self.object_to_idx["agent"])
        mask[self.vis, self.vis]=False
        dynamic_map[mask] = self.object_to_color["agent"]
        agent_obs_pose_now = np.argwhere(mask)[:,::-1] + pose - np.array([self.vis, self.vis])
        
        for key in self.dynamic_obs_pose.keys():
            obs_poses = self.dynamic_obs_pose[key]
            agent_poses = self.agent_obs_pose[key]
            if len(obs_poses):
                obs_poses_array = np.array(obs_poses)
                obs_pose_in_local_coord = obs_poses_array - pose
                idx = np.logical_and(np.abs(obs_pose_in_local_coord[:,0])<=self.vis, 
                             np.abs(obs_pose_in_local_coord[:,1])<=self.vis)
                obs_pose_in_local_coord = obs_pose_in_local_coord[idx] + np.array([self.vis, self.vis])
                tmp = np.zeros(dynamic_map.shape)
                tmp[obs_pose_in_local_coord[:,1],obs_pose_in_local_coord[:,0]] = self.dynamic_obs_decay[key]
                dynamic_map += tmp
            if len(agent_poses):
                obs_poses_array = np.array(agent_poses)
                obs_pose_in_local_coord = obs_poses_array - pose
                idx = np.logical_and(np.abs(obs_pose_in_local_coord[:,0])<=self.vis, 
                             np.abs(obs_pose_in_local_coord[:,1])<=self.vis)
                obs_pose_in_local_coord = obs_pose_in_local_coord[idx] + np.array([self.vis, self.vis])
                tmp = np.zeros(dynamic_map.shape)
                tmp[obs_pose_in_local_coord[:,1],obs_pose_in_local_coord[:,0]] = self.agent_obs_decay[key]
                dynamic_map += tmp          
        self.dynamic_obs_pose[2]=self.dynamic_obs_pose[1]
        self.dynamic_obs_pose[1]=self.dynamic_obs_pose[0]
        self.dynamic_obs_pose[0]=dynamic_obs_pose_now    
        self.agent_obs_pose[2]=self.agent_obs_pose[1]
        self.agent_obs_pose[1]=self.agent_obs_pose[0]
        self.agent_obs_pose[0]=agent_obs_pose_now
                
        # third channel is the reference path
        subgoal = goal
        if replan:
            new_map = self.update_map(obs, pose)
            #print(new_map)
            self.planner.update_map(new_map)
            self.plan(pose, goal)

            path_array = np.array(self.path)
            if len(self.path):
                # transform the path to the agent local coordinate
                path_in_local_coord = path_array - pose
                # filter out the path out of view
                idx = np.logical_and(np.abs(path_in_local_coord[:,0])<=self.vis, 
                                     np.abs(path_in_local_coord[:,1])<=self.vis)
                path_in_local_coord = path_in_local_coord[idx] + np.array([self.vis, self.vis])
                # set the path to 1

                self.obs_map[2,:,:][path_in_local_coord[:,1],path_in_local_coord[:,0]] = self.path_color

            # select the subgoal and draw it on the map
            if len(self.path)>self.subgoal_length:
                subgoal = self.path[self.subgoal_length]
            else:
                subgoal = goal

        goal_in_local_coord = subgoal-pose
        if np.abs(goal_in_local_coord[0])<=self.vis and np.abs(goal_in_local_coord[1])<=self.vis:
            goal_in_local_coord=goal_in_local_coord+np.array([self.vis, self.vis])
            self.obs_map[2,:,:][goal_in_local_coord[1],goal_in_local_coord[0]] = self.goal_color

        # normalized relative goal

        
        relative_goal = (subgoal-pose)/self.pose_normalizer
        input_val = list(relative_goal)
        theta = math.atan2(relative_goal[1],relative_goal[0])
        input_val.append(theta)
        #plt.imshow(self.obs_map)
        if debug:
            return self.obs_map.copy(), input_val, path_array
        else:
            return self.obs_map.copy(), input_val,index

    def act(self, input_img, input_val):
        state_img = torch.tensor([input_img])
        state_val = torch.tensor([input_val])
        state_img = state_img.float().to(self.device)
        state_val = state_val.float().to(self.device)

        img, probs = self.ac.forward(state_img, state_val)

#         print(probs)
        probs = torch.exp(probs)
        m = Categorical(probs)
        _, greedy_action = torch.max(probs.data, 1)
        action = m.sample()


        return greedy_action.item(), img, action.item(), m.log_prob(
                   action), m.entropy().mean()
