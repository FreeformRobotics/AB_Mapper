#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 17:57:35 2020

@author: Zuxin, lance
"""

import numpy as np
import cv2
import yaml

def read_img(img_file, mode=1, scale_ratio = 1):
    img = cv2.imread(img_file, mode)
    cols = int(img.shape[1]*scale_ratio)
    rows = int(img.shape[0]*scale_ratio)
    img = cv2.resize(img, (cols,rows), interpolation = cv2.INTER_NEAREST)
    return img

img_file = "map2.jpg"
scale_ratio = 0.2

start_pos = []
goal_pos = []

img = read_img(img_file, mode=0, scale_ratio = scale_ratio)
map = img.copy()

img = read_img(img_file, mode=1, scale_ratio = scale_ratio)
cols = img.shape[1]
rows = img.shape[0]
print(" map shape: ", img.shape)

def on_mouse(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print('Start Mouse Position: '+str(x)+', '+str(y))
        sbox = [x, y]
        start_pos.append(sbox)
        cv2.circle(img, (x,y), 1, (255,120,12),1)

    elif event == cv2.EVENT_MBUTTONDOWN:
        print('End Mouse Position: '+str(x)+', '+str(y))
        ebox = [x, y]
        goal_pos.append(ebox)
        cv2.circle(img, (x,y), 1, (0,255,120),1)

    elif event == cv2.EVENT_MOUSEMOVE:
        print('Mouse Position: '+str(x)+', '+str(y))

def save_map(map):
    np_img = np.array(map)
    np_img = ~np_img  # invert B&W
    np_img = np_img/255
    np_img[np_img > 0.5] = 1
    np_img[np_img <= 0.5] = 0 #obstacle
    obs = np.where(np_img == 1)
    ret = dict()
    info = dict()
    info['dimensions'] = [cols, rows]
    info['obstacles'] = [(int(obs[1][i]), int(obs[0][i])) for i in range(len(obs[0]))]
    ret['map'] = info
    start_len  = len(start_pos)
    goal_len = len(goal_pos)
    if start_len != goal_len:
        print("start list len: ", start_len, ". goal list len: ", goal_len,". They are not match!")
        return 0
    ret['agents'] = [
            {'start': s, 'goal': g, 'name': 'agent'+str(i) }for s,g,i in zip(start_pos, goal_pos, range(len(start_pos)))
            ]
    print(ret['agents'])
    with open('result.yaml', 'w') as yaml_file:
        yaml.dump(ret, yaml_file, default_flow_style = False)

while(1):
    
    cv2.namedWindow('map',cv2.WINDOW_NORMAL)
    render_resize_ratio = int(4/scale_ratio)
    cv2.resizeWindow('map', cols*render_resize_ratio,rows*render_resize_ratio)
    cv2.setMouseCallback('map', on_mouse, 0)
    cv2.imshow('map', img)
    key = cv2.waitKey(33)
    if  key == 27: # press "escape"
        cv2.destroyAllWindows()
        break
    elif key == ord('s'): #ASCII number of 's'
        print("save")
        save_map(map)