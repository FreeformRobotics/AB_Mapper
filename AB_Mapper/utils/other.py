'''
@Author: Zuxin Liu
@Email: zuxinl@andrew.cmu.edu
@Date:   2020-02-24 10:30:30
@LastEditTime: 2020-03-25 22:43:22
@Description:
'''

import random
import numpy
import torch
import collections


def seed(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synthesize(array):
    d = collections.OrderedDict()
    d["mean"] = numpy.mean(array)
    d["std"] = numpy.std(array)
    d["min"] = numpy.amin(array)
    d["max"] = numpy.amax(array)
    return d

def get_max_step(agent_list):
    max_step = 0
    max_step_list = []
    for agent in agent_list:
        max_step_list.append(agent.max_step/agent.ratio)
        if agent.max_step>max_step:
            max_step = agent.max_step
    return max_step, max_step_list