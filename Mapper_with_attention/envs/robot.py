'''
@Author: Zuxin Liu
@Email: zuxinl@andrew.cmu.edu
@Date:   2020-02-16 16:43:10
@LastEditTime: 2020-03-25 22:42:30
@Description:
'''

from envs.astar import A_star
from envs.dstar_lite import D_star
import numpy as np

class Robot(object):
	"""Robot agent"""
	def __init__(self, map, idx_to_object):
		self.map = map
		self.idx_to_object = idx_to_object
		self.object_to_idx = dict(zip(self.idx_to_object.values(), self.idx_to_object.keys()))
		#self.planner = D_star(map,idx_to_object)
		self.planner = A_star(map,idx_to_object)
		self.step_remain = 0

	def sample_free_space(self, grid):
		'''
		  @param [array] grid : grid map with obstacle information.
		'''
		idx_free = np.argwhere(grid==self.object_to_idx["free"])
		sample = np.random.randint(idx_free.shape[0])
		return idx_free[sample][::-1]

	def set_pose(self, pose):
		self.pose = pose

	def plan(self, goal):
		self.path = self.planner.plan(self.pose, goal)
		self.future_path = self.path.copy()
		self.step_remain = len(self.path)
		self.step_max = len(self.path)

	def tick(self):
		self.step_remain = self.step_remain-1

	def query_next_pose(self):
		idx = self.step_max - self.step_remain
		pose = self.path[idx]
		self.future_path = self.path[idx:]
		return pose


class RobotManager(object):
	def __init__(self, map, idx_to_object, robot_num):
		self.map = map
		self.row, self.col = map.shape
		self.idx_to_object = idx_to_object
		self.object_to_idx = dict(zip(self.idx_to_object.values(), self.idx_to_object.keys()))
		self.robot_num = robot_num
		# Initialize a map to store all the robots' positions. -1 represents current grid is not occupied by robots, otherwise represents the robot index.
		self.robot_map = -np.ones(map.shape)

	def init_robots(self, current_map):
		self.robots = []
		self.robot_pose = []
		self.robot_future_traj = []

		for idx in range(self.robot_num):
			robot = Robot(self.map, self.idx_to_object)
			start_pose = robot.sample_free_space(current_map)
			robot.set_pose(start_pose)
			self.robot_pose.append(start_pose)
			current_map[start_pose[1],start_pose[0]] = self.object_to_idx["dynamic obstacle"]
			self.robot_map[start_pose[1],start_pose[0]] = idx
			goal = robot.sample_free_space(current_map)
			robot.plan(goal)
			self.robots.append(robot)
			self.robot_future_traj.append(robot.future_path)

	def query_next_pose(self, obs):
		current_robot_pose = []
		# store the poses if no collision occured
		for robot in self.robots:
			if robot.step_remain == 0:
				# Assign a new goal for the robot
				goal = robot.sample_free_space(self.map)
				robot.plan(goal)
				current_robot_pose.append(robot.pose)
			else:
				pose = robot.query_next_pose()
				current_robot_pose.append(pose)
		return current_robot_pose

	def step_robot(self, i, pose):
		self.robots[i].tick()
		self.robots[i].set_pose(pose)
		self.robot_pose[i] = pose
		self.robot_future_traj[i] = self.robots[i].future_path
