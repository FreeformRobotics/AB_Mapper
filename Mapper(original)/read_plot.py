import matplotlib.pyplot as plt
import numpy as np


class R_P(object):
    def __init__(self, smooth=False):

        self.smooth = smooth

    def read(self, path):

        with open(path, 'r') as f:
            data = f.read()
            # print('data',data)
            y = eval(data)
            # print('y',y)
        if self.smooth:
            window = 20
            smoothed_rewards = [np.mean(y[i - window:i + 1]) if i > window
                                else np.mean(y[:i + 1]) for i in range(1500)]
            return smoothed_rewards
        else:
            return y

    def show(self, data):

        plt.figure(figsize=(12, 8))
        plt.ylabel('success rate')
        plt.xlabel('Episodes')
        color = ['red','blue','green','purple','black']
        text= ['notcritic_instead_9_action','our_path','baseline_path','notcritic_instead_9_action_entropy_weight_entropy','baseline_path_8_action']
        for i in range(len(data)):
            plt.plot(data[i], color=color[i],label=text[i])
        plt.legend()
        plt.show()

