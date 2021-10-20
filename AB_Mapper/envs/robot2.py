from envs.astar import A_star
from envs.dstar_lite import D_star
import numpy as np
import copy
from random import sample
import sys
import random
class Robot(object):
	"""Robot agent"""
	def __init__(self, map, idx_to_object, blind = False):
		self.map = map
		self.idx_to_object = idx_to_object
		self.object_to_idx = dict(zip(self.idx_to_object.values(), self.idx_to_object.keys()))
		self.planner = D_star(map.copy(), idx_to_object)
		self.blind = blind
		#if (blind): print("I am blind")

		self.step_remain = 0
		self.goal = None
		self.pose = None

	def set_pose(self, pose):
		self.pose = pose

	def set_goal(self, goal):
		self.goal = goal

	def get_pose(self):
		return self.pose

	def get_goal(self):
		return self.goal

	def sample_free_space(self, grid):
		'''
		  @param [array] grid : grid map with obstacle information.
		'''
		idx_free = np.argwhere(grid==self.object_to_idx["free"])
		sample = np.random.randint(idx_free.shape[0])
		return idx_free[sample][::-1]

	def plan(self, observation = None):
		#print(observation)
		if self.blind:
			observation = self.mask_agent(observation)
		self.next_step = self.planner.plan(pose = self.pose, goal = self.goal, obs = observation, debug = False)
		self.future_path = self.planner.get_path()

	def get_future_path(self):
		return self.future_path

	def query_next_pose(self):
		#print(self.future_path)
		return self.next_step#self.future_path[0]

	def reset(self):
		self.planner.set_map(self.map)
		self.planner.reset()

	def mask_agent(self, observation):
		if observation is not None:
			observation[observation == 2] = 0
		return observation

class RobotManager(object):
	def __init__(self, map, idx_to_object, robot_num, detect_agent = 0):
		self.map = map
		self.row, self.col = map.shape
		self.idx_to_object = idx_to_object
		self.object_to_idx = dict(zip(self.idx_to_object.values(), self.idx_to_object.keys()))
		self.robot_num = robot_num
		# Initialize a map to store all the robots' positions. -1 represents current grid is not occupied by robots, otherwise represents the robot index.
		self.robot_map = -np.ones(map.shape)
		self.detect_agent = detect_agent; # probability of detecting agent

	def init_robots(self, current_map):
		self.robots = []
		self.robot_pose = []
		self.robot_future_traj = []

		#the index of robot that cannot detect agent
		r = [i for i in range(self.robot_num)]
		blind_robot = sample(r, int((1-self.detect_agent)*self.robot_num))
		#print("blind", blind_robot)

		for idx in range(self.robot_num):
			robot = Robot(self.map, self.idx_to_object, blind = (idx in blind_robot))
			start_pose = robot.sample_free_space(current_map)
			robot.set_pose(start_pose)
			self.robot_pose.append(start_pose)

			current_map[start_pose[1],start_pose[0]] = self.object_to_idx["dynamic obstacle"]
			self.robot_map[start_pose[1],start_pose[0]] = idx

			goal = robot.sample_free_space(current_map)
			robot.set_goal(goal)

			robot.plan()
			self.robots.append(robot)
			self.robot_future_traj.append(robot.get_future_path())

		#print(self.robot_pose)
	def query_next_pose(self, obs):
		current_robot_pose = []
		self.hot_area = {}
		cross = {}
		swap = {}
		still = []
		# store the poses if no collision occured
		for i in range(self.robot_num):
			#print("=="*15,i)

			robot = self.robots[i]
			pose_last = self.robot_pose[i]#last position

			#print("last :",self.robot_pose[i],"goal:", robot.get_goal())


			if np.array_equal(pose_last, robot.get_goal()): # arrive at goal, find new goal and plan, and do not move in this step
				# Assign a new goal for the robot
				goal = robot.sample_free_space(self.map)
				robot.set_goal(goal)
				robot.reset()
				robot.plan(observation = obs[i])
				pose = pose_last
				current_robot_pose.append(pose_last)

			else:# does not reach goal
				robot.plan(observation = obs[i])
				pose = robot.query_next_pose() if robot.query_next_pose() is not None else robot.get_pose()
				
				#check if the next pose is the same
				
				cur_pose = robot.get_pose()
				if tuple(pose) not in self.hot_area:
					self.hot_area[tuple(pose)] = i
				else:
					#print("not allowed to move next pose", i, self.hot_area[tuple(pose)])
					rand_move = self.random_move(cur_pose)
					pose = rand_move if rand_move is not None else robot.get_pose()
					#pose = robot.get_pose() # second robot keep still

				#check if crossover situtaion happens
				cur_pose = robot.get_pose()
				if abs(cur_pose[0] - pose[0]) + abs(cur_pose[1] - pose[1]) > 1: # cross next move
					#form hashable tuple
					if cur_pose[0] > pose[0]:
						hashtuple = [pose[0], cur_pose[0]]
					else: 
						hashtuple = [cur_pose[0], pose[0]]
					if cur_pose[1] > pose[1]:
						hashtuple += [pose[1], cur_pose[1]]
					else: 
						hashtuple += [cur_pose[1], pose[1]]

					if tuple(hashtuple) not in cross:
						cross[tuple(hashtuple)] = i
					else:
						pose = robot.get_pose() # second robot keep still
						#print("not allowed to move cross", i)
						rand_move = self.random_move(cur_pose)
						pose = rand_move if rand_move is not None else robot.get_pose()

				#check if two robots swap positions
				cur_pose = robot.get_pose()
				swap[tuple(cur_pose)] = tuple(pose)
				if tuple(pose) in swap and swap[tuple(pose)] == tuple(cur_pose): #if two robots swap position, second robot makes random move 
					#print("swap happens", i)
					rand_move = self.random_move(cur_pose)
					pose = rand_move if rand_move is not None else robot.get_pose()
			
				current_robot_pose.append(pose)
				if np.array_equal(pose, robot.get_pose()):
					still.append(i)

			#print("cur:",robot.get_pose()," next", pose, "future path:",robot.get_future_path())
		#print(current_robot_pose, hot_area, cross)
#		print(hot_area)
#		print(cross)
		#print("still robot:", still)
		return current_robot_pose

	def step_robot(self, i, pose):
		self.robots[i].set_pose(pose)
		self.robot_pose[i] = pose
		self.robot_future_traj[i] = self.robots[i].get_future_path()

	def random_move(self, pos):
		cur = tuple(pos)
		(x_, y_) = cur
		results = [(x_+1,y_), (x_-1,y_), (x_, y_+1), (x_,y_-1), (x_+1,y_+1), (x_-1,y_+1), (x_+1, y_-1), (x_-1,y_-1)]
		results = list(filter(self.empty_grid, results))
		if results:
			#print("inside random move: ",results)
			candidate = random.choice(results)
			next_move = np.array([candidate[0], candidate[1]])
			#print("random:", next_move)
			return next_move	
		else:
			return None

	def empty_grid(self, pos):
		(x, y) = pos
		return 0 <= x < self.col and 0 <= y < self.row and self.map[y][x] == 0 and pos not in self.hot_area

