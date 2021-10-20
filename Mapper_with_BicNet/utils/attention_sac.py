import torch
import torch.nn.functional as F
from torch.optim import Adam


from misc import soft_update, hard_update, enable_gradients, disable_gradients
from agents import AttentionAgent
from critics import AttentionCritic
import torch.nn.utils as torch_utils

from model1 import ActorCritic
from torch.autograd import Variable
from torch import Tensor

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cast = lambda x: Variable(Tensor(x).to(device), requires_grad=False)
MSELoss = torch.nn.MSELoss()

class AttentionSAC(object):
    """
    Wrapper class for SAC agents with central attention critic in multi-agent
    task
    """
    def __init__(self, agent_init_params, sa_size,
                 gamma=0.95, tau=0.01, pi_lr=0.01, q_lr=0.01,
                 reward_scale=10.,
                 pol_hidden_dim=128,
                 critic_hidden_dim=512, attend_heads=4,
                 args = False,
                 **kwargs):
        """
        Inputs:
            agent_init_params (list of dict): List of dicts with parameters to
                                              initialize each agent
                num_in_pol (int): Input dimensions to policy
                num_out_pol (int): Output dimensions to policy
            sa_size (list of (int, int)): Size of state and action space for
                                          each agent
            gamma (float): Discount factor
            tau (float): Target update rate
            pi_lr (float): Learning rate for policy
            q_lr (float): Learning rate for critic
            reward_scale (float): Scaling for reward (has effect of optimal
                                  policy entropy)
            hidden_dim (int): Number of hidden dimensions for networks
        """
        self.nagents = len(sa_size)

        self.agents = [ActorCritic()
                         for i in range(self.nagents)]

        self.critic = AttentionCritic(sa_size, hidden_dim=critic_hidden_dim,
                                      attend_heads=attend_heads).to(device)
        self.critic_best_model_path = './critic_weights'
        self.target_critic = AttentionCritic(sa_size, hidden_dim=critic_hidden_dim,
                                             attend_heads=attend_heads)
        hard_update(self.target_critic, self.critic)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=q_lr,
                                     weight_decay=5e-4)
        self.target_actor = ActorCritic()

        self.agent_init_params = agent_init_params
        self.gamma = gamma
        self.tau = tau
        self.pi_lr = pi_lr
        self.q_lr = q_lr
        self.reward_scale = reward_scale
        self.pol_dev = 'cpu'  # device for policies
        self.critic_dev = 'cpu'  # device for critics
        self.trgt_pol_dev = 'cpu'  # device for target policies
        self.trgt_critic_dev = 'cpu'  # device for target critics
        self.niter = 0


    @property
    def target_policies(self):
        return [a for a in self.agents]

    def insert_action(self, x):
        action_array = [0 for i in range(9)]
        action_array[x] = 1
        return action_array

    def update_my_critic(self, sample, next_input_img_list,next_input_val_list, agent_list,
                         attention_index_tar, attention_index,
                         soft=True, logger=None, **kwargs):

        obs, acs, rews, next_obs, dones = sample
        # print('acs',[acs])
        # Q loss
        # print(
        #     '\n obs=',obs,
        #       '\n acs',acs,
        #       '\n rews', rews,
        #       '\n next_obs',next_obs,
        #       '\n dones',dones
        #       )
        next_acs = []
        next_log_pis = []
        for pi, ob, iv in zip(agent_list, next_input_img_list, next_input_val_list):

            _, _, curr_next_ac, curr_next_log_pi, _ = pi.act(ob,iv)
            next_acs.append(cast([self.insert_action(curr_next_ac)]))

            next_log_pis.append(curr_next_log_pi)
        # print('\n acs',acs,'\n next_acs',cast(next_acs))
        trgt_critic_in = list(zip(next_obs, next_acs))
        critic_in = list(zip(obs, acs))
        # print('critic_in',critic_in)


        next_qs = self.target_critic(trgt_critic_in,attention_index_tar)
        critic_rets = self.critic(critic_in, attention_index, regularize=True,
                                  logger=logger, niter=self.niter)

        q_loss = 0
        for a_i, nq, log_pi, (pq, regs) in zip(range(self.nagents), next_qs,
                                               next_log_pis, critic_rets):
            target_q = (rews[a_i] +
                        self.gamma * nq *
                        (1 - dones[a_i]))
            # print('target_q',target_q)
            if soft:
                target_q -= log_pi / self.reward_scale
            q_loss += MSELoss(pq, target_q.detach())
            for reg in regs:
                q_loss += reg  # regularizing attention

        q_loss.backward()

        torch_utils.clip_grad_norm_(self.critic.parameters(), 10)# 既然在BP过程中会产生梯度消失/爆炸（就是偏导无限接近0，导致长时记忆无法更新），
                                                                # 那么最简单粗暴的方法，设定阈值，当梯度小于/大于阈值时，更新的梯度为阈值，如下图所示：
        self.critic_optimizer.step()
        self.critic_optimizer.zero_grad()


    def update_critic_targets(self):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        soft_update(self.target_critic, self.critic, self.tau)
        # for a in self.agents:
        #     soft_update(a.target_policy, a.policy, self.tau)

    @classmethod
    def init_from_env(cls,env, agents=4, gamma=0.95, tau=0.01,
                      pi_lr=0.01, q_lr=0.01,
                      reward_scale=10.,
                      pol_hidden_dim=128, critic_hidden_dim=512, attend_heads=4,
                      **kwargs):
        """
        Instantiate instance of this class from multi-agent environment

        env: Multi-agent Gym environment
        gamma: discount factor
        tau: rate of update for target networks
        lr: learning rate for networks
        hidden_dim: number of hidden dimensions for networks
        """
        agent_init_params = []
        sa_size = []
        # for acsp, obsp in zip(env.action_space,
        #                       env.observation_space):
        #     agent_init_params.append({'num_in_pol': obsp.shape[0],
        #                               'num_out_pol': acsp.n})
        #     sa_size.append((obsp.shape[0], acsp.n))
        for i in range(agents):
            sa_size.append((128, 9))
        init_dict = {
            'agents': agents,
                    'gamma': gamma, 'tau': tau,
                     'pi_lr': pi_lr, 'q_lr': q_lr,
                     'reward_scale': reward_scale,
                     'pol_hidden_dim': pol_hidden_dim,
                     'critic_hidden_dim': critic_hidden_dim,
                     'attend_heads': attend_heads,
                     'agent_init_params': agent_init_params,
                     'sa_size': sa_size}
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance

    @classmethod
    def init_from_save(cls, filename, load_critic=False):
        """
        Instantiate instance of this class from file created by 'save' method
        """
        save_dict = torch.load(filename)
        instance = cls(**save_dict['init_dict'])
        instance.init_dict = save_dict['init_dict']
        for a, params in zip(instance.agents, save_dict['agent_params']):
            a.load_params(params)

        if load_critic:
            critic_params = save_dict['critic_params']
            instance.critic.load_state_dict(critic_params['critic'])
            instance.target_critic.load_state_dict(critic_params['target_critic'])
            instance.critic_optimizer.load_state_dict(critic_params['critic_optimizer'])
        return instance

    def save_critic_param(self, best_critic_id):
        torch.save(self.critic.critics[best_critic_id].state_dict(), self.critic_best_model_path+'critic.pth' )

    def update_each_critic_param(self,update_flag,best_critic_id):
        for a, params in zip(self.critic.critics, self.critic.critics.parameters()):

            a.load_params(self.critic.critics[best_critic_id].parameters())