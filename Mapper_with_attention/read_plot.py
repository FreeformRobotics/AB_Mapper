import matplotlib.pyplot as plt
import os
import numpy as np


class R_P(object):
    def __init__(self, smooth=False):

        self.smooth = smooth

    def read(self, path):
        y = []
        with open(path, 'r') as f:
            data = f.read()
            y = eval(data)

        if self.smooth:
            window = 50
            smoothed_rewards = [np.mean(y[i - window:i + 1]) if i > window
                                else np.mean(y[:i + 1]) for i in range(len(y))]
            return smoothed_rewards
        else:
            return y

    def show(self, data):

        plt.figure(figsize=(12, 8))
        plt.ylabel('Total Rewards')
        plt.xlabel('Episodes')
        color = ['red','blue']
        text= ['baseline','our method']
        for i in range(len(data)):
            plt.plot(data[i], color=color[i],label=text[i])
        plt.legend()
        plt.show()

