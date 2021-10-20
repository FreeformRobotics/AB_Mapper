import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
from read_plot import R_P

def getdata():
    baselines_method_path = [
    '/media/ubuntu/D/ghf_project/20_20_15_10/baseline_mapper20_20_15_10/train_data/succ/exp2.txt',
    '/media/ubuntu/D/ghf_project/20_20_15_10/baseline_mapper20_20_15_10_2/train_data/succ/exp2.txt',
    '/media/ubuntu/D/ghf_project/20_20_15_10/baseline_mapper20_20_15_10 _1/train_data/succ/exp2.txt',
    '/media/ubuntu/D/ghf_project/20_20_15_10/baseline_mapper20_20_15_10_3/train_data/succ/exp2.txt',
    ]

    our_optimal = [
        '/media/ubuntu/D/GHF_pycode/20_20_15_10/mapper20_20_15_10/train_data/succ/exp3.txt',
        '/media/ubuntu/D/GHF_pycode/20_20_15_10/mapper20_20_15_10_1/train_data/succ/exp6.txt',
        '/media/ubuntu/D/GHF_pycode/20_20_15_10/mapper20_20_15_10/train_data/succ/exp4.txt',
        '/media/ubuntu/D/GHF_pycode/20_20_15_10/mapper20_20_15_10_3/train_data/succ/exp6.txt',
        '/media/ubuntu/D/GHF_pycode/20_20_15_10/mapper20_20_15_10_4/train_data/succ/exp5.txt'
    ] # 100_update tau=0.001 seed=9527 or 996 or 997 grad_norm=10 reward_scale=20 (new)

    our_attention_head = [
        '/media/ubuntu/D/GHF_pycode/20_20_15_10/mapper20_20_15_10/train_data/succ/exp7.txt',
        '/media/ubuntu/D/GHF_pycode/20_20_15_10/mapper20_20_15_10_1/train_data/succ/exp9.txt',
        '/media/ubuntu/D/GHF_pycode/20_20_15_10/mapper20_20_15_10_3/train_data/succ/exp10.txt',
    ]# 效果不好

    mapper3=[
        '/media/ubuntu/D/GHF_pycode/20_20_15_10/mapper20_20_15_10_2/train_data/succ/exp6.txt',
        '/media/ubuntu/D/GHF_pycode/20_20_15_10/mapper20_20_15_10_2/train_data/succ/exp7.txt',
        '/media/ubuntu/D/GHF_pycode/20_20_15_10/mapper20_20_15_10_4/train_data/succ/exp7.txt'
    ]#  10_update critic_t grad_norm=10 tau=0.01 _997 reward_scale=20. 4/30


    newmapper1=[
        '/media/ubuntu/D/GHF_pycode/20_20_15_10/mapper20_20_15_10_3/train_data/succ/exp8.txt',
        '/media/ubuntu/D/GHF_pycode/20_20_15_10/mapper20_20_15_10_4/train_data/succ/exp8.txt',
        '/media/ubuntu/D/GHF_pycode/20_20_15_10/mapper20_20_15_10_1/train_data/succ/exp7.txt'
    ]# 100_update grad_norm=10 tau=0.001 seed=996 target_q -= log_pi 5/2 [x]

    newmapper2=[
            '/media/ubuntu/D/ghf_project/35_30_hard/0/train_data/succ/exp1.txt',
        '/media/ubuntu/D/ghf_project/35_30_hard/0/train_data/succ/exp2.txt',
        '/media/ubuntu/D/ghf_project/35_30_hard/1/train_data/succ/exp1.txt',
    ]#baseline

    newmapper3 =[
        '/media/ubuntu/D/GHF_pycode/35_30_hard/35_attention_head/0/train_data/succ/exp2.txt',
    ]#our


    rp = R_P(smooth=True)

    # mapper1_data = []
    # for i in range(len(newmapper3)):
    #     mapper1_data.append(rp.read(newmapper3[i]))

    mapper2_data=[]
    for i in range(len(our_attention_head)):
        mapper2_data.append(rp.read(our_attention_head[i]))

    mapper3_data=[]
    for i in range(len(newmapper1)):
        mapper3_data.append(rp.read(newmapper1[i]))

    mapper4_data=[]
    for i in range(len(mapper3)):
        mapper4_data.append(rp.read(mapper3[i]))

    our_optimal_data=[]
    for i in range(len(our_optimal)):
        our_optimal_data.append(rp.read(our_optimal[i]))

    origin_baseline=[]
    for i in range(len(baselines_method_path)):
        origin_baseline.append(rp.read(baselines_method_path[i]))

    our_data = []
    for i in range(len(newmapper3)):
        our_data.append(rp.read(newmapper3[i]))
    baseline_data = []
    for i in range(len(newmapper2)):
        baseline_data.append(rp.read(newmapper2[i]))

    return our_data

data = getdata()
fig = plt.figure()
xdata = np.array([i for i  in range(2500)])
linestyle = ['--', '-.',':','-.','-','-.']
color = ['r', 'g','b','y','purple','orange']
label = [
         # '100_update tau=0.001 seed=9107 reward_scale=20 grad_norm=10 attention_head =15',
         # '10_update critic_t grad_norm=10 tau=0.01 _997 reward_scale=20.',
         # '100_update grad_norm=10 tau=0.001 reward_scale=20',
         # 'baseline',
            'our_data',
         ]

for i in range(len(data)):
    sns.tsplot(time=xdata, data=data[i], color=color[i], linestyle=linestyle[i], condition=label[i])
plt.ylabel("Success Rate", fontsize=25)
plt.xlabel("Iteration Number", fontsize=25)
plt.title("Hard_35_30", fontsize=30)
plt.show()