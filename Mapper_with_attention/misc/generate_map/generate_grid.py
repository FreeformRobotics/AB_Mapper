#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 07:57:35 2020

@author: lance
"""

from PIL import Image
import numpy as np
import yaml
import matplotlib.pyplot as plt


img = Image.open('random_map.jpg').convert('L')

np_img = np.array(img)
np_img = ~np_img  # invert B&W
np_img = np_img/255
np_img[np_img > 0.75] = 1
np_img[np_img <= 0.75] = 0 #obstacle

obs = np.where(np_img == 1)

ret = dict()
info = dict()
info['dimensions'] = [np_img.shape[1], np_img.shape[0]]
info['obstacles'] = [(int(obs[1][i]), int(obs[0][i])) for i in range(len(obs[0]))]
ret['map'] = info
#ret['agents'] = [
 #       {'start': [0, 0], 'goal': [2, 0], 'name': 'agent0'}, 
 #       {'start': [2, 0], 'goal': [0, 0], 'name': 'agent1'}
 #       ]

with open('../../random_map.yaml', 'w') as yaml_file:
    yaml.dump(ret, yaml_file, default_flow_style = False)
