'''
@Author: Zuxin Liu
@Email: zuxinl@andrew.cmu.edu
@Date:   2020-03-01 21:53:06
@LastEditTime: 2020-03-25 22:42:13
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
import os
import torch
from torch import optim
import torch.nn.functional as F
from torch.distributions import Categorical
import datetime
import tensorboardX
import utils
import sys
import time


actions = ['N','S','E','W','NW','WS','SE','EN','.']
idx_to_act = {0:"N",1:"S",2:"E",3:"W", 4:"NW",5:"WS",6:"SE",7:"EN",8:"."}
act_to_idx = dict(zip(idx_to_act.values(),idx_to_act.keys()))

parser = argparse.ArgumentParser()

parser.add_argument("--exp", help="experiment number", type=int, default=0)
# 0 is 1500 seed=997,
# 1 is 2000 seed=1191
# 2 is 2000 seed=123
# 3 is 2000 seed =9527


parser.add_argument("--map-dir", help="static map dir", default="map")

parser.add_argument("--model-dir", help="model dir", default="weights")
parser.add_argument("--load", help="load from the given path", default=False)
parser.add_argument("--name", help="model name", default=None)
parser.add_argument("--lr", type=float, default=0.0003,
                    help="learning rate (default: 0.0003)")
parser.add_argument("--seed", type=int, default=997,
                    help="random seed (default: 1)")
parser.add_argument("--map", help="static map path", default="60_65_hard")
parser.add_argument("--obs", type=int, default=30,
                    help="dynamic obstacle number (default: 4)")
parser.add_argument("--agents", type=int, default=175,
                    help="agents number (default: 4)")
parser.add_argument("--interval", type=int, default=100,
                    help="episode interval to update all the other agents' model to the best one's model parameter (default: 100)")
parser.add_argument("--entropy-weight", type=float, default=0.01,
                    help="entropy weight in the loss term (default: 0.01)")
parser.add_argument("--render", type=bool, default=True,
                    help="render the env to visualize (default: True)")
parser.add_argument("--goal-range", type=int, default=6,
                    help="goal sample range (default: 6)")
parser.add_argument("--agent-type", type=int, default=0,
                    help="0: full feature; 1: without global planner A_star guidance; 2: without dynamic obstacle trajectory; 3: without sub-goal guidance (default: 0)")

args = parser.parse_args()

agent_type = args.agent_type
if agent_type==0:
    from agent import Agent, compute_returns
    print(" Import agent with full features...")
elif agent_type==1:
    from agent_no_A_star import Agent, compute_returns
    print(" Import agent without A_star planner...")
elif agent_type==2:
    from agent_no_trajectory import Agent, compute_returns
    print(" Import agent without dyanmic obstacle traj...")
elif agent_type==3:
    from agent_no_guidance import Agent, compute_returns
    print(" Import agent without sub goal guidance...")
else:
    sys.exit("without such type of agent!! check the --agent-type for more detail.")

RENDER=args.render
MAX_STEP_RATIO = 4
SUCCESS_RATE_THRES = 0.99

model_dir = args.model_dir
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

if args.load:
    load_model_path = os.path.join(model_dir, args.load)

date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
default_model_name = f"exp{args.exp}_{args.map}_lr_{args.lr}_seed{args.seed}_{date}"
save_model_path = os.path.join(model_dir, default_model_name+".pth")
if args.name:
    save_model_path = os.path.join(model_dir, args.name)

save_log_path = os.path.join(model_dir,"exp"+str(args.exp))
if not os.path.exists(save_log_path):
    os.mkdir(save_log_path)
    print("Exp log directory " , save_log_path ,  " Created ")
else:    
    print("Exp log directory ", save_log_path ,  " already exists")

# Load loggers and Tensorboard writer

txt_logger = utils.get_txt_logger(save_log_path, default_model_name)
csv_file, csv_logger = utils.get_csv_logger(save_log_path, default_model_name)
tb_writer = tensorboardX.SummaryWriter(save_log_path)

# Log command and all script arguments

txt_logger.info("{}\n".format(" ".join(sys.argv)))
txt_logger.info("{}\n".format(args))
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# device = torch.device( "cpu")
txt_logger.info(f"Device: {device}\n")

window = Window(default_model_name)

# environment initilization
with open( os.path.join(args.map_dir,args.map+".yaml")) as map_file:
    map = yaml.load(map_file, Loader=yaml.FullLoader)

utils.seed(args.seed)
env = grid_env.GridEnv(map, agent_num=args.agents, window=window, obs_num=args.obs, goal_range = args.goal_range)
state = env.reset()
txt_logger.info("Environments loaded\n")
if RENDER:
    env.render(show_traj=True)


agent_list = []
for i in range(args.agents):
    agent = Agent(env.background_grid.copy(), ID=i)
    agent.set_max_step(state, ratio = MAX_STEP_RATIO)

    optimizer = optim.Adam(agent.ac.parameters(),lr=args.lr)
    agent.optimizer = optimizer
    if args.load:
        print("Agent %d is loading model from "%(i), load_model_path)
        #print(agent.device)
        agent.ac.load_state_dict(torch.load(load_model_path,map_location=torch.device(agent.device)))
    agent_list.append(agent)

max_episode = 1500
step_count = 0
save_model_count = 0
save_model_frequency = args.interval
sum_rewards_array = np.zeros(args.agents)
entropy_weight = args.entropy_weight
succ_array = []
reward_array = []
for epi in range(max_episode):
    for agent in agent_list:
        agent.reset_memory()
    
    save_model_count += 1
    max_step, max_step_list = utils.get_max_step(agent_list)
    success_rate = 0
    first_done = np.zeros(args.agents)
    for step_num in range(max_step):
        action_list = []
        for i in range(args.agents):
            agent = agent_list[i]
            input_img, input_val = agent.preprocess(state, replan = True)
    #         print(input_val)
            _, action, log_prob, value, ent = agent.act(input_img, input_val)
    #         print(value)
            action_list.append(idx_to_act[action])
            agent.log_probs.append(log_prob)
            agent.values.append(value)
            # print('value',value)
            agent.current_ent = ent
        # print(action_list)
        next_state, reward, done, _ = env.step(action_list)
        #print(reward)
        
        for i in range(args.agents):
            agent = agent_list[i]
            #print("agent %d done %d first_done %d "%(i, done[i], first_done[i]))

            if done[i] and first_done[i]:
                agent.log_probs.pop()
                agent.values.pop()
                first_done[i] = 1
                continue
            if done[i] and not first_done[i]:
                first_done[i]=1
            additional_reward = agent.compute_reward(next_state)
            #print("agent %d reward: %3f, additional_reward: %3f"%(i, reward[i], additional_reward))
            reward[i] += additional_reward
            agent.entropy += agent.current_ent
            agent.collision = next_state["collision"]
            agent.steps = next_state["steps"]
            agent.rewards.append(torch.FloatTensor([reward[i]]).unsqueeze(1).to(device))
            agent.masks.append(torch.FloatTensor([1 - done[i]]).unsqueeze(1).to(device))
        state = next_state
        if RENDER:
            env.render(show_traj = True, dynamic_obs_traj = False)
        #time.sleep(1)
        success_rate = np.sum(done)/args.agents
        if success_rate>SUCCESS_RATE_THRES:
            # if more than SUCCESS_RATE_THRES% agents reached the goal
            break
    
    loss_min, entropy_min, actor_loss_min, critic_loss_min = 1e7,1e7,1e7,1e7
    sum_reward_max = -9999
    total_reward = []
    extra_time = []



    for i in range(args.agents):
        agent = agent_list[i]
        #print("agent %d reward list "%(i), agent.rewards)
        sum_reward = sum(agent.rewards).item()
        sum_rewards_array[i] += sum_reward

        total_reward.append(sum_reward)
        collsions=np.array(state["collision"])
        collsions[collsions>=1]=1
        total_steps=state["steps"]
        extra_time.append( (state["steps"][i]-max_step_list[i])/max_step_list[i] )
        
        input_img, input_val = agent.preprocess(next_state, replan = True)
        _, _, _, next_value, _ = agent.act(input_img, input_val)
        returns = compute_returns(next_value, agent.rewards, agent.masks)

        # print('next_value',next_value,'len',len(next_value))

        log_probs = torch.cat(agent.log_probs)
        returns  = torch.cat(returns).detach()
        # print('agent.values',agent.values,'\n',
        #       len(agent.values))
        values  = torch.cat(agent.values)
        # print('agent>values',agent.values)
        # print('values 0 ==',agent_list[0].values,
        #       '\n values 1 ==', agent_list[1].values,
        #       '\n values 2 ==',agent_list[2].values,
        #       '\n values 3 ==',agent_list[3].values,)

        advantage = returns - values
        actor_loss  = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        loss = actor_loss + 0.001 * critic_loss - entropy_weight * agent.entropy
        # print("agent.entropy", agent.entropy)

        if sum_reward>sum_reward_max:
            loss_min = loss.item()
            entropy_min = agent.entropy.item()
            actor_loss_min = actor_loss.item()
            critic_loss_min = critic_loss.item()
            sum_reward_max = sum_reward

        agent.optimizer.zero_grad()
        loss.backward()
        agent.optimizer.step()
    #time.sleep(20)

    state = env.reset()
    for agent in agent_list:
        agent.set_max_step(state, ratio = MAX_STEP_RATIO)

    if RENDER:
        env.render(show_traj=True)

    #print(total_reward)
    #print(total_steps)
    average_reward = np.mean(total_reward)
    collision_rate = np.mean(collsions)
    extra_step = np.mean(extra_time)

    # print and save training info
    header = ["exp", "type", "update","success_rate", "avg_reward", "collision","extra",  "max_reward", "loss", "entropy", "actor_loss", "critic_loss"]
    data = [args.exp, agent_type , epi, success_rate, average_reward,collision_rate, extra_step, sum_reward_max, loss_min, entropy_min,  actor_loss_min, critic_loss_min]

    succ_array.append(success_rate)
    reward_array.append(average_reward)

    txt_logger.info(
        "Exp {} | Agent {} | Epi {} | succ rate {:.2f}| avg reward {:.2f}| collision {:.2f}| extra {:.2f}| max reward {:.2f}| L {:.3f} | eL {:.3f} | aL {:.3f} | cL {:.3f}".format(*data))
    if epi == 0:
        csv_logger.writerow(header)
    csv_logger.writerow(data)
    csv_file.flush()
    for field, value in zip(header, data):
            tb_writer.add_scalar(field, value, epi)

    if save_model_count>=save_model_frequency and save_model_path:
        best_model_idx = np.argmax(sum_rewards_array)
        diff = np.max(sum_rewards_array) - np.min(sum_rewards_array)
        agent = agent_list[best_model_idx]
        torch.save(agent.ac.state_dict(), save_model_path)
        print("Best agent %d's model is saved to "%(best_model_idx), save_model_path)
        lamda = 2
        sum_rewards_norm = sum_rewards_array/diff+0.5
        sum_reward_exp = np.exp(sum_rewards_norm*lamda)
        max_reward = sum_reward_exp[best_model_idx]
        # update all the agent's parameters to the best one with probability
        for i in range(args.agents):
            agent = agent_list[i]
            p = np.random.uniform(0,1)
            if p> sum_reward_exp[i]/max_reward:
                agent.ac.load_state_dict(torch.load(save_model_path))
        save_model_count = 0
        sum_rewards_array = np.zeros(args.agents)
        

window.close()
data=[reward_array,succ_array]
file_name = ['./train_data/_reward/', './train_data/succ/']
i=0
for name in file_name:
    values = data[i]
    with open(name+'exp'+str(args.exp)+'.txt', "w") as output:
        output.write(str(values))
        output.close()
    i+=1

