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
from envs import grid_env
from envs.rendering import Window
import os
import torch
from torch import optim

import datetime
import tensorboardX
import utils
import sys

## new add

from utils.attention_sac import AttentionSAC
from torch.autograd import Variable
from torch import Tensor
from read_plot import R_P
from utils.misc import soft_update



actions = ['N','S','E','W','NW','WS','SE','EN','.']
idx_to_act = {0:"N",1:"S",2:"E",3:"W", 4:"NW",5:"WS",6:"SE",7:"EN",8:"."}
act_to_idx = dict(zip(idx_to_act.values(),idx_to_act.keys()))
avg_reward = []
succ_rate = []
parser = argparse.ArgumentParser()

parser.add_argument("--exp", help="experiment number",type=int, default=2)
# 0 is True V sub_num_agent=3 seed =997 target_actor_and_critic q_lr==0.0001 head =20
# 1 is True V sub_num_agent=3 seed =9527 target_actor_and_critic q_lr==0.0001 head =4
# 2 is True V sub_num_agent=3 seed =123 target_actor_and_critic q_lr==0.0001 head =4
parser.add_argument("--map-dir", help="static map dir", default="map")
parser.add_argument("--model-dir", help="model dir", default="weights")

parser.add_argument("--load", help="load from the given path", default=False)
parser.add_argument("--name", help="model name", default=None)
parser.add_argument("--lr", type=float, default=0.0003,
                    help="learning rate (default: 0.0003)")
parser.add_argument("--seed", type=int, default=123,
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
parser.add_argument("--render", type=bool, default=False,
                    help="render the env to visualize (default: True)")
parser.add_argument("--goal-range", type=int, default=6,
                    help="goal sample range (default: 6)")
parser.add_argument("--agent-type", type=int, default=0,
                    help="0: full feature; 1: without global planner A_star guidance; 2: without dynamic obstacle trajectory; 3: without sub-goal guidance (default: 0)")
"""
new add
"""

parser.add_argument("--num_updates",default=4,type=int,
                    help="Number of updates per update cycle")
parser.add_argument("--pol_hidden_dim", default=128, type=int)
parser.add_argument("--critic_hidden_dim", default=40*8, type=int)
parser.add_argument("--attend_heads", default=4, type=int)
parser.add_argument("--pi_lr", default=0.001, type=float)
parser.add_argument("--q_lr", default=0.0001, type=float)
parser.add_argument("--tau", default=0.001, type=float)
parser.add_argument("--gamma", default=0.99, type=float)
parser.add_argument("--reward_scale", default=20., type=float)
parser.add_argument("--critic_model-dir", help="model dir", default="critic_weights")
parser.add_argument("--sub_num_agent", default=15, type=int)
# 3 is 20_20_15_10 | 6 is 20_20_35_30
parser.add_argument("--train",default=True,type=bool)

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

def get_key1(dct, value):
   return list(filter(lambda k:dct[k] == value, dct))

RENDER=args.render
MAX_STEP_RATIO = 4
SUCCESS_RATE_THRES = 0.99

model_dir, critic_model_dir = args.model_dir, args.critic_model_dir
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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

txt_logger.info(f"Device: {device}\n")

cast = lambda x: Variable(Tensor(x).to(device), requires_grad=False)



# environment initilization
with open( os.path.join(args.map_dir,args.map+".yaml")) as map_file:
    map = yaml.load(map_file, Loader=yaml.FullLoader)


def insert_action(x):
    action_array = [0 for i in range(len(actions))]
    action_array[x]=1
    return action_array


file_name = ['./train_data/_reward/', './train_data/succ/','./train_data/_time/']
import time
def train():
    t1=time.time()
    time_list=[]
    window = Window(default_model_name)
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
    target_actor = agent_list



    max_episode =1500 #8000000
    step_count = 0
    save_model_count = 0
    save_model_frequency = args.interval
    sum_rewards_array = np.zeros(args.agents)
    entropy_weight = args.entropy_weight


    model = AttentionSAC.init_from_env(env,
                                       agents=args.agents,
                                       tau=args.tau,
                                       pi_lr=args.pi_lr,
                                       q_lr=args.q_lr,
                                       gamma=args.gamma,
                                       pol_hidden_dim=args.pol_hidden_dim,
                                       critic_hidden_dim=args.critic_hidden_dim,
                                       attend_heads=args.attend_heads,
                                       reward_scale=args.reward_scale)


    for epi in range(max_episode):
            for agent in agent_list:
                agent.reset_memory()

            save_model_count += 1
            max_step, max_step_list = utils.get_max_step(agent_list)
            success_rate = 0
            first_done = np.zeros(args.agents)
            # new add
            agent_values = [[] for i in range(len(agent_list))]
            img_list = []
            for step_num in range(max_step):

                action_list = []

                img_list = []
                MyAAC_list = []
                next_input_img_list =[]
                all_probs = []
                attention_index = []
                for i in range(args.agents):
                    agent = agent_list[i]

                    input_img, input_val,index = agent.preprocess(state,sub_num_agent=args.sub_num_agent, replan = True)

                    _, img, action, log_prob, ent, probs = agent.act(input_img, input_val)

                    MyAAC_list.append(cast([insert_action(action)]))

                    attention_index.append(index)
                    img_list.append(img)

                    # 'img' is the state after CNN(input_img)
                    action_list.append(idx_to_act[action])
                    agent.log_probs.append(log_prob)
                    agent.current_ent = ent
                    all_probs.append(probs)
                next_state, reward, done, _ = env.step(action_list)

                critic_in = list(zip(img_list, MyAAC_list))
                # print("attention_index",attention_index)
                values = model.critic(inps=critic_in, agent_index=attention_index,return_q=False, return_all_q=True)# this is all Q(s,a) ,for every agent's every action

                for i in range(len(agent_list)):
                    agent = agent_list[i]
                    agent.values.append((values[i] * all_probs[i]).sum(dim=1, keepdim=True)) # this is V(s) for every agent

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
                # time.sleep(1)
                success_rate = np.sum(done)/args.agents
                if success_rate>SUCCESS_RATE_THRES:
                    # if more than SUCCESS_RATE_THRES% agents reached the goal
                    break


            loss_min, entropy_min, actor_loss_min, critic_loss_min = 1e7,1e7,1e7,1e7
            sum_reward_max = -9999
            total_reward = []
            extra_time = []

            next_img_list = []
            next_action_list = []
            next_log_prob_list = []

            next_input_img_list = []
            next_input_val_list = []

            attention_index_tar= []
            for i in range(args.agents):
                agent = target_actor[i] # target_actor

                #print("agent %d reward list "%(i), agent.rewards)
                sum_reward = sum(agent.rewards).item()
                sum_rewards_array[i] += sum_reward

                total_reward.append(sum_reward)
                collsions = np.array(state["collision"])
                collsions[collsions>=1]=1
                total_steps = state["steps"]
                extra_time.append( (state["steps"][i]-max_step_list[i])/max_step_list[i] )

                input_img, input_val,index = agent.preprocess(next_state,sub_num_agent=args.sub_num_agent, replan = True)

                _, next_img, next_action, next_log_prob, _, nest_probs = agent.act(input_img, input_val)

                attention_index_tar.append(index)
                next_img_list.append(next_img)

                next_action_list.append(cast([insert_action(next_action)]))
                next_log_prob_list.append(next_log_prob)

                next_input_img_list.append(input_img)
                next_input_val_list.append(input_val)

            # sample = img_list, MyAAC_list, reward_list,next_img_list, done_list
        # calculate next_value
            target_critic_in = list(zip(next_img_list, next_action_list))

            next_values = model.target_critic(inps=target_critic_in, agent_index=attention_index_tar, return_q=True, return_all_q=False) # this is Q(s',a') for every agent

            model.critic_optimizer.zero_grad()
            for i in range(args.agents):
                agent = agent_list[i]
                # print('next_values[i]',next_values[i])
                returns = compute_returns(next_values[i], agent.rewards, agent.masks)

                log_probs = torch.cat(agent.log_probs)
                returns = torch.cat(returns).detach()
                # print('returns:', returns)

                values = torch.cat(agent.values).detach()

                advantage = returns - values
                actor_loss = -(log_probs * advantage.detach()).mean()

                critic_loss = advantage.pow(2).mean()
                loss = actor_loss - entropy_weight * agent.entropy+0.01*critic_loss
                       # + 0.001 \
                       # * critic_loss - entropy_weight * agent.entropy
                #print("agent", i," sum_reward: ", sum_reward)
                if sum_reward > sum_reward_max: #
                    loss_min = loss.item()
                    entropy_min = agent.entropy.item()
                    actor_loss_min = actor_loss.item()
                    # critic_loss_min = critic_loss.item()
                    sum_reward_max = sum_reward

                agent.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                agent.optimizer.step()

            model.critic_optimizer.step()
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

            succ_rate.append(success_rate)
            avg_reward.append(average_reward)

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
                torch.cuda.empty_cache()

                print("Best agent %d's model is saved to "%(best_model_idx), save_model_path)
                lamda = 2
                sum_rewards_norm = sum_rewards_array/diff+0.5
                sum_reward_exp = np.exp(sum_rewards_norm*lamda)
                max_reward = sum_reward_exp[best_model_idx]
                # update all the agent's parameters to the best one with probability
                for i in range(args.agents):
                    agent = agent_list[i]
                    p = np.random.uniform(0,1)

                    soft_update(target=target_actor[i].ac, source=agent_list[i].ac, tau=args.tau)

                    if p > sum_reward_exp[i]/max_reward:
                        agent.ac.load_state_dict(torch.load(save_model_path))

                model.update_critic_targets() #update target_critic
                save_model_count = 0
                sum_rewards_array = np.zeros(args.agents)

    window.close()

    t2=time.time()
    time_list.append(t2-t1)
    # torch.save(model.critic.state_dict(),)
    data=[avg_reward, succ_rate, time_list]

    i=0
    for name in file_name:
        values = data[i]
        with open(name+'exp'+str(args.exp)+'.txt', "w") as output:
            output.write(str(values))
            output.close()
        i+=1

def read_show_data(path, smooth):
    rp = R_P(smooth=smooth)
    data = []
    for i in range(len(path)):
        data.append(rp.read(path[i]))
    rp.show(data)

if __name__ == "__main__":
    if args.train == True:

        train()









