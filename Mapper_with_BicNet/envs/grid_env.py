'''
@Author: Zuxin Liu
@Email: zuxinl@andrew.cmu.edu
@Date:   2020-03-01 21:20:54
@LastEditTime: 2020-07-27 14:32:02
@Description:
'''

import matplotlib
import math
import cv2
import numpy as np
import sys
from PIL import Image
from envs.rendering import Renderer
from envs.rendering import Window
from envs.robot2 import RobotManager
import time
import copy
import os


class Agent(object):
    def __init__(self, pose, goal, ID):
        self.pose = pose
        self.goal = goal
        self.ID = ID
        self.step_cost = -0.01
        self.stay_cost = -0.02
        '''
        self.object_to_cost = {
          "free"     : 0,
          "obstacle" : -0.2,
          "dynamic obstacle" : -0.2,
          "agent" : -0.2,
          "goal"     : 20
        }
        '''
        self.object_to_cost = {
          "free"     : 0,
          "obstacle" : -5,
          "dynamic obstacle" : -5,
          "agent" : -5,
          "goal"     : 30
        }

        self.collision_count = 0
        self.steps = 0
        self.reset()
    def reset(self):
        self.reward = 0
        self.reward_static_obs = 0
        self.reward_agent = 0
        self.reward_dynamic_obs = 0
        self.reward_goal = 0
        self.reward_step = 0
    def set_goal(self, goal):
        self.goal = goal
    def reach_goal(self):
        self.reward_goal += self.object_to_cost["goal"]
    def collide_with_static_obs(self):
        self.reward_static_obs += self.object_to_cost["obstacle"]
    def collide_with_dynamic_obs(self):
        self.reward_dynamic_obs += self.object_to_cost["dynamic obstacle"]*0.05
        self.collision_count += 1
    def collide_with_agent(self):
        self.reward_dynamic_obs += self.object_to_cost["agent"]*0.05
        self.collision_count += 1
    def step(self, stay = False):
        self.steps += 1
        if stay:
            self.reward_step += self.stay_cost
        else:
            self.reward_step += self.step_cost
    def get_reward(self):
        self.reward = self.reward_static_obs + self.reward_dynamic_obs + self.reward_goal + self.reward_step
        return self.reward

class GridEnv:
    """Custom Environment that follows gym interface"""
    def __init__(self, map, agent_num, window = None, obs_num = 0, goal_range = 20):
        self.window = window
        self.map = map
        self.obstacles = map["map"]["obstacles"]
        self.agent_num = agent_num
        self.dynamic_obs_num = obs_num
        #Env constant
        self.tilesize = 48
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.visibility = 7
        self.goal_sample_range = goal_range
        self.detect_agent = 0.4; # probability of dynamic obstacle being able to detect agent
        self.traj_color = np.random.randint(256, size=(self.agent_num,3))
        #Map the grid index to object type.
        self.idx_to_object = {
            0   :   "free",
            1   :   "obstacle",
            2   :   "agent",
            3   :   "dynamic obstacle",
            4   :   "unseen",
            5   :   "goal",
        }
        self.object_to_idx = dict(zip(self.idx_to_object.values(), self.idx_to_object.keys()))
        self.row = map["map"]["dimensions"][1]
        self.col = map["map"]["dimensions"][0]
        print("row: ", self.row, " col: ", self.col)

        #self.img_save_path = "saved_img/"
        self.time = -1

        self.renderer = Renderer(self.row, self.col, self.tilesize, self.traj_color)



        #self.reset()

    def reset(self):
        #simulation variable
        self.time += 1
        self.descrip = str(self.time)
        self.traj = []
        self.agent_on_goal = [-1]*self.agent_num

        self.background_grid = np.zeros(shape=(self.row, self.col), dtype = int)
        self.background_grid = self.update_background_grid(self.background_grid)
        self.background_img = np.ones(shape=(self.row*self.tilesize, self.col * self.tilesize, 3), dtype=np.uint8)*255
        # Initialize a map to store all the agents' positions. -1 represents current grid is not occupied, otherwise represents the agent index.
        self.agent_map = -np.ones(self.background_grid.shape)

        current_grid = self.background_grid.copy()

        #initialize agents
        self.init_agents()
        for i in range(self.agent_num):
            # print("self.agent_pose[%d]:".format(i),self.agent_pose[i])
            self.traj.append([self.agent_pose[i]])
            self.set_grid(current_grid, self.agent_pose[i], "agent")

        #initialize dynamic obstacles
        self.robot_manager = RobotManager(self.background_grid.copy(), self.idx_to_object, self.dynamic_obs_num, self.detect_agent)
        self.robot_manager.init_robots(current_grid)
        self.robot_pose = self.robot_manager.robot_pose
        self.robot_future_traj = self.robot_manager.robot_future_traj
        self.robot_obs = self.get_obs(current_grid, name="robot")

        # Draw the goal and static obstacles on the background image
        self.background_img = self.renderer.draw_background(self.background_img, self.agent_goal, self.obstacles)

        observations = self.get_obs(current_grid, name="agent")

        pose = copy.deepcopy(self.agent_pose)
        obs = copy.deepcopy(observations)
        goal = copy.deepcopy(self.agent_goal)
        state = {"obs": obs, "pose": pose, "goal": goal}

        return state
    def step(self, action):
        '''
        @param [list] action : Agents action list for the current step. ['^','v','<','>','.']

        Return the list of agents pose (array), list of observations (array) and static map (array)
        '''
        assert (len(action)==self.agent_num), "The length of action list should be the same as the agents number"

        self.time += 1

        current_grid = self.background_grid.copy()

        current_agent_pose = []
        current_agent_map = -np.ones(current_grid.shape)
        agent_map = self.agent_map
        # get the current agent pose if no conflicts occured
        for i in range(self.agent_num):
            act = action[i]
            if not self.agent_done[i]:
                stay = (act=='.') # punish more if the robot stay at the original position
                self.agents[i].step(stay) # add step cost
            pose = self.agent_pose[i]
            result = self.move_agents(pose, act)
            if (result == np.array([-1,-1])).all():
                # collide with obstacles, do not move
                self.agents[i].collide_with_static_obs()
                current_agent_pose.append(pose)
                self.set_grid(current_grid, pose, "agent")
                continue
            else:
                current_agent_pose.append(result)

        # get the current robot pose, old robot map and init current robot map
        current_robot_map = -np.ones(current_grid.shape)
        current_robot_pose = self.robot_manager.query_next_pose(self.robot_obs)
        robot_map = self.robot_manager.robot_map

        self.collision_map = np.zeros(current_grid.shape)

        if self.dynamic_obs_num!=0:
            self.resolve_robot_conflict(current_grid, robot_map, current_robot_map, agent_map, current_agent_map, current_robot_pose, current_agent_pose)

        self.resolve_agent_conflict(current_grid, robot_map, current_robot_map, agent_map, current_agent_map, current_robot_pose, current_agent_pose)

        self.agent_map = current_agent_map
        observations = self.get_obs(current_grid, name="agent")
        self.robot_obs = self.get_obs(current_grid, name="robot")

        agent_reward = copy.deepcopy(self.get_reward())
        agent_collision_count = self.get_collision_count()
        agent_step = self.get_step()

        pose = copy.deepcopy(self.agent_pose)
        obs = copy.deepcopy(observations)
        goal = copy.deepcopy(self.agent_goal)
        state = {"obs":obs, "pose":pose, "goal":goal, "collision":agent_collision_count, "steps":agent_step}

        done = copy.deepcopy(self.agent_done)

        return state, agent_reward, done, None

    def render(self, show_traj = False, dynamic_obs_traj = False, traj = [], save_img = False, save_img_dir = "storage"):
        '''
        @param [boolean] show_traj : Determine if we render the agents' trajactories.
        @param [boolean] dynamic_obs : TODO, remove this param
        Return the rendered image.
        '''
        self.update_done_info()
        img = self.background_img.copy()
        self.renderer.draw_dynamic_obs(img, self.robot_pose)
        if dynamic_obs_traj:
            self.renderer.draw_trajectory(img, self.robot_future_traj)
        if show_traj:
            self.renderer.draw_trajectory(img, self.traj)
        if len(traj):
            self.renderer.draw_trajectory(img, traj)

        self.renderer.draw_agents(img, self.agent_pose, self.agent_on_goal)
        #self.renderer.draw_obs(img, self.agent_pose, self.visibility)

        if save_img:
            new_img = Image.fromarray(img)
            img_save_path = os.path.join(save_img_dir, str(self.time)+".png")
            new_img.save(img_save_path)

        self.window.show_img(img)

    def update_done_info(self):
        agent_on_goal = [count+done for count, done in zip(self.agent_on_goal,self.agent_done)]
        self.agent_on_goal = agent_on_goal

    def init_agents(self):
        current_grid = self.background_grid.copy()
        self.agent_pose = []
        self.agent_goal = []
        self.agent_done = []
        self.agents = []
        #self.pose2agent = {}
        for i in range(self.agent_num):
            goal = self.sample_free_space(current_grid)
            while (len(self.agent_goal) > 0 and any(np.array_equal(goal, x) for x in self.agent_goal)):
                goal = self.sample_free_space(current_grid)
            self.agent_goal.append(goal)
            #print("goal: ", goal)
            y,x = goal
            xmin = max(int(x-self.goal_sample_range), 0)
            ymin = max(int(y-self.goal_sample_range), 0)
            xmax = min(int(x+1+self.goal_sample_range), self.row)
            ymax = min(int(y+1+self.goal_sample_range), self.col)

            start_pose = self.sample_free_space(current_grid[xmin:xmax, ymin:ymax])+np.array((ymin, xmin))

            self.agent_pose.append(start_pose)
            self.set_grid(current_grid, start_pose, "agent")
            self.agent_map[start_pose[1],start_pose[0]] = i
            agent = Agent(start_pose, goal, i)
            self.agents.append(agent)
            #self.pose2agent[tuple(start_pose)] = agent
            self.agent_done.append(False)

    def move_agents(self, pose, act):
        '''
        Move the agent and check if the agent will collide with obstacles,

        @param [numpy_array] pose : The agent's position.
        @param [string] act : The agent's action.

        Return False if collision exists, otherwise return new pose, [numpy_array].
        '''
        if act=='N':
            pose_new = pose + np.array([0,-1])
        elif act=='S':
            pose_new = pose + np.array([0,1])
        elif act=='E':
            pose_new = pose + np.array([1,0])
        elif act=='W':
            pose_new = pose + np.array([-1,0])
        elif act=='NW':
            pose_new = pose + np.array([-1,-1])
        elif act=='WS':
            pose_new = pose + np.array([-1,1])
        elif act=='SE':
            pose_new = pose + np.array([1,1])
        elif act=='EN':
            pose_new = pose + np.array([1,-1])
        elif act=='.':
            pose_new = pose + np.array([0,0])
        else:
            sys.exit("No such a action '"+ act +"' in the action space")

        # Make sure the agent is not out of the map
        if not (0<=pose_new[0]<self.col and 0<=pose_new[1]<self.row):
            return np.array([-1,-1])

        pose_new_type = self.background_grid[pose_new[1], pose_new[0]]

        # Hit the obstacles in the map
        if pose_new_type == self.object_to_idx["obstacle"]:
            return np.array([-1,-1])
        else:
            return pose_new

    def get_obs(self, grid_map, name="agent"):
        # Get the observation of each agent on the current grid map, fixed size
        if name=="agent":
            observation = list()
            for i in range(self.agent_num):
                obs_size = self.visibility*2+1
                obs = np.ones((obs_size,obs_size))
                pos = self.agent_pose[i]
                y,x = pos
                xmin = max(int(x-self.visibility), 0)
                ymin = max(int(y-self.visibility), 0)
                xmax = min(int(x+1+self.visibility), self.row)
                ymax = min(int(y+1+self.visibility), self.col)
                height = xmax-xmin
                width = ymax-ymin
                obs_x_min, obs_x_max, obs_y_max, obs_y_min = (0, obs_size, obs_size, 0)
                if x-self.visibility<0:
                    obs_x_min = self.visibility - x
                if x+1+self.visibility>self.row:
                    obs_x_max = self.visibility + self.row - x
                if y-self.visibility<0:
                    obs_y_min = self.visibility - y
                if y+1+self.visibility>self.col:
                    obs_y_max = self.visibility + self.col - y
                obs[obs_x_min:obs_x_max,obs_y_min:obs_y_max] = grid_map[xmin:xmax, ymin:ymax]
                observation.append(obs)
        elif name=="robot":
            observation = list()
            for i in range(self.dynamic_obs_num):
                obs_size = self.visibility*2+1
                obs = np.ones((obs_size,obs_size))
                pos = self.robot_pose[i]
                y,x = pos
                xmin = max(int(x-self.visibility), 0)
                ymin = max(int(y-self.visibility), 0)
                xmax = min(int(x+1+self.visibility), self.row)
                ymax = min(int(y+1+self.visibility), self.col)
                height = xmax-xmin
                width = ymax-ymin
                obs_x_min, obs_x_max, obs_y_max, obs_y_min = (0, obs_size, obs_size, 0)
                if x-self.visibility<0:
                    obs_x_min = self.visibility - x
                if x+1+self.visibility>self.row:
                    obs_x_max = self.visibility + self.row - x
                if y-self.visibility<0:
                    obs_y_min = self.visibility - y
                if y+1+self.visibility>self.col:
                    obs_y_max = self.visibility + self.col - y
                obs[obs_x_min:obs_x_max,obs_y_min:obs_y_max] = grid_map[xmin:xmax, ymin:ymax]
                observation.append(obs)
        else:
            print("can not get local observations for type: ", name)
            return None
        return observation

    def update_background_grid(self, grid_map):
        for o in self.obstacles:
            self.set_grid(grid_map, o, "obstacle")
            #grid_map[int(o[1])][int(o[0])] = self.object_to_idx["obstacle"]
        return grid_map

    def sample_free_space(self, grid):
        '''
        @param [array] grid : grid map with obstacle information.
        '''
        idx_free = np.argwhere(grid==self.object_to_idx["free"])
        sample = np.random.randint(idx_free.shape[0])
        return idx_free[sample][::-1]

    def set_grid(self, grid, pos, name):
        grid[int(pos[1])][int(pos[0])] = self.object_to_idx[name]

    def query_nearby_robot(self, robot_map, pos, i):
        r = 2
        xmin = max(int(pos[1]-r), 0)
        ymin = max(int(pos[0]-r), 0)
        xmax = min(int(pos[1]+1+r), self.row)
        ymax = min(int(pos[0]+1+r), self.col)
        obs = robot_map[xmin:xmax, ymin:ymax]
        idx = obs[obs!=-1]
        idx = idx[idx!=i]
        return idx

    def query_nearby_agent(self, robot_map, pos, i):
        r = 2
        xmin = max(int(pos[1]-r), 0)
        ymin = max(int(pos[0]-r), 0)
        xmax = min(int(pos[1]+1+r), self.row)
        ymax = min(int(pos[0]+1+r), self.col)
        obs = robot_map[xmin:xmax, ymin:ymax]
        idx = obs[obs!=-1]
        idx = idx[idx!=i]
        return idx

    def get_reward(self):
        agent_reward = []
        for agent in self.agents:
            agent_reward.append(agent.get_reward())
            agent.reset()
        return agent_reward

    def get_collision_count(self):
        agent_collision_count = []
        for agent in self.agents:
            agent_collision_count.append(agent.collision_count)
        return agent_collision_count

    def get_step(self):
        agent_step = []
        for agent in self.agents:
            agent_step.append(agent.steps)
        return agent_step

    def resolve_robot_conflict(self, current_grid, robot_map, current_robot_map, agent_map, current_agent_map, current_robot_pose, current_agent_pose):
        # resolve robots conflict first
        for i in range(self.dynamic_obs_num):
            robot_pose_i_last = self.robot_pose[i]
            robot_pose_i_now = current_robot_pose[i]
            collision_flag = False
            idx1 = robot_map[robot_pose_i_now[1],robot_pose_i_now[0]]
            idx2 = agent_map[robot_pose_i_now[1],robot_pose_i_now[0]]
            # The position is occupied by other agents or robots
            if (idx1!=-1 and idx1!=i) or (idx2!=-1):
                collision_flag = True
            # for i-th robot, check all the surrounding robots and get the index
            if not collision_flag:
                idx_robot = self.query_nearby_robot(robot_map, robot_pose_i_now, i)
                if idx_robot.size>0:
                    for j in idx_robot:
                        robot_pose_j_last = self.robot_pose[int(j)]
                        robot_pose_j_now = current_robot_pose[int(j)]
                        # If swap position occured, stay at original position
                        if (robot_pose_i_now == robot_pose_j_last).all() and (robot_pose_i_last==robot_pose_j_now).all():
                            collision_flag = True
                            break
                        # if go to the same position
                        elif (robot_pose_i_now == robot_pose_j_now).all():
                            collision_flag = True
                            break
                        # cross occured, stay at original position
                        elif ((robot_pose_i_now+robot_pose_i_last)==(robot_pose_j_now+robot_pose_j_last)).all():
                            collision_flag = True
                            break
            if not collision_flag:
                # check all the surrounding agents and get the index
                idx_agent = self.query_nearby_agent(agent_map, robot_pose_i_now, -2)
                if idx_agent.size>0:
                    for j in idx_agent:
                        robot_pose_j_last = self.agent_pose[int(j)]
                        robot_pose_j_now = current_agent_pose[int(j)]
                        # If swap position occured, stay at original position
                        if (robot_pose_i_now == robot_pose_j_last).all() and (robot_pose_i_last==robot_pose_j_now).all():
                            collision_flag = True
                            break
                        # if go to the same position
                        elif (robot_pose_i_now == robot_pose_j_now).all():
                            collision_flag = True
                            break
                        # cross occured, stay at original position
                        elif ((robot_pose_i_now+robot_pose_i_last)==(robot_pose_j_now+robot_pose_j_last)).all():
                            collision_flag = True
                            break
            if collision_flag:
                current_robot_map[robot_pose_i_last[1],robot_pose_i_last[0]]=i
                self.set_grid(current_grid, robot_pose_i_last, "dynamic obstacle")
                continue
            # No collision, update all the states
            self.robot_manager.step_robot(i, robot_pose_i_now)
            self.robot_pose[i] = robot_pose_i_now
            current_robot_map[robot_pose_i_now[1],robot_pose_i_now[0]]=i
            self.set_grid(current_grid, robot_pose_i_now, "dynamic obstacle")

        #self.robot_manager.move_robots(current_grid)
        assert (self.robot_pose == self.robot_manager.robot_pose), "what the fuck?"
        self.robot_manager.robot_map = current_robot_map
        self.robot_future_traj = self.robot_manager.robot_future_traj
        return current_grid, robot_map, current_robot_map, agent_map, current_agent_map, current_robot_pose, current_agent_pose

    def resolve_agent_conflict(self, current_grid, robot_map, current_robot_map, agent_map, current_agent_map, current_robot_pose, current_agent_pose):
        # resolve agents conflict
        for i in range(self.agent_num):
            if self.agent_done[i]:
                self.set_grid(current_grid, self.agent_pose[i], "agent")
                self.traj[i].append(self.agent_pose[i])
                current_agent_map[self.agent_pose[i][1],self.agent_pose[i][0]]=-1
                continue
            agent_pose_i_last = self.agent_pose[i]
            agent_pose_i_now = current_agent_pose[i]
            collision_flag = False
            collision_agent_flag = False
            first_arrive_flag = False
            idx1 = robot_map[agent_pose_i_now[1],agent_pose_i_now[0]]
            idx2 = agent_map[agent_pose_i_now[1],agent_pose_i_now[0]]
            # The position is occupied by other agents or robots
            if (idx2!=-1 and idx2!=i) or (idx1!=-1):
                collision_flag = True
            if not collision_flag:
                # for i-th agent, check all the surrounding robots and get the index
                idx_robot = self.query_nearby_robot(robot_map, agent_pose_i_now, -2)
                if idx_robot.size>0:
                    for j in idx_robot:
                        robot_pose_j_last = self.robot_pose[int(j)]
                        robot_pose_j_now = current_robot_pose[int(j)]
                        # If swap position occured, stay at original position
                        if (agent_pose_i_now == robot_pose_j_last).all() and (agent_pose_i_last==robot_pose_j_now).all():
                            collision_flag = True
                            break
                        # if go to the same position
                        elif (agent_pose_i_now == robot_pose_j_now).all():
                            collision_flag = True
                            if not self.collision_map[agent_pose_i_now[1],agent_pose_i_now[0]]:
                                first_arrive_flag = True
                                self.collision_map[agent_pose_i_now[1],agent_pose_i_now[0]]=1
                            break
                        # cross occured, stay at original position
                        elif ((agent_pose_i_now+agent_pose_i_last)==(robot_pose_j_now+robot_pose_j_last)).all():
                            collision_flag = True
                            if not self.collision_map[agent_pose_i_now[1],agent_pose_i_now[0]]:
                                first_arrive_flag = True
                                self.collision_map[agent_pose_i_now[1],agent_pose_i_now[0]]=1
                            break
            if not collision_flag:
                # check all the surrounding agents and get the index
                idx_agent = self.query_nearby_agent(agent_map, agent_pose_i_now, i)
                if idx_agent.size>0:
                    for j in idx_agent:
                        agent_pose_j_last = self.agent_pose[int(j)]
                        agent_pose_j_now = current_agent_pose[int(j)]
                        # If swap position occured, stay at original position
                        if (agent_pose_i_now == agent_pose_j_last).all() and (agent_pose_i_last == agent_pose_j_now).all():
                            collision_flag = True
                            collision_agent_flag = True
                            break
                        # if go to the same position
                        elif (agent_pose_i_now == agent_pose_j_now).all():
                            collision_flag = True
                            collision_agent_flag = True
                            if not self.collision_map[agent_pose_i_now[1],agent_pose_i_now[0]]:
                                first_arrive_flag = True
                                self.collision_map[agent_pose_i_now[1],agent_pose_i_now[0]]=1
                            break
                        # cross occured, stay at original position
                        elif ((agent_pose_i_now+agent_pose_i_last)==(agent_pose_j_now+agent_pose_j_last)).all():
                            collision_flag = True
                            collision_agent_flag = True
                            if not self.collision_map[agent_pose_i_now[1],agent_pose_i_now[0]]:
                                first_arrive_flag = True
                                self.collision_map[agent_pose_i_now[1],agent_pose_i_now[0]]=1
                            break
            if collision_flag and not first_arrive_flag:
                current_agent_map[agent_pose_i_last[1],agent_pose_i_last[0]]=i
                if collision_agent_flag:
                    self.agents[i].collide_with_agent()
                else:
                    self.agents[i].collide_with_dynamic_obs()
                self.set_grid(current_grid, agent_pose_i_last, "agent")
                # print("agent ",i," collide with dynamic obs")
                continue
            else:
                if first_arrive_flag:  # collision occured but can move
                    if collision_agent_flag:
                        self.agents[i].collide_with_agent()
                    else:
                        self.agents[i].collide_with_dynamic_obs()
                # No collision, update all the states
                if (self.agent_goal[i]==agent_pose_i_now).all():
                    # reach the goal
                    self.agents[i].reach_goal()
                    self.agent_done[i] = True
                self.agent_pose[i] = agent_pose_i_now
                current_agent_map[agent_pose_i_now[1],agent_pose_i_now[0]]=i
                self.set_grid(current_grid, agent_pose_i_now, "agent")
                self.traj[i].append(agent_pose_i_now)
        return current_grid, robot_map, current_robot_map, agent_map, current_agent_map, current_robot_pose, current_agent_pose



