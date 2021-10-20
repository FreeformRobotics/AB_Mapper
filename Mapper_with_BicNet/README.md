# Multi-Agent path planning with deep reinforcement learning


### Structure
The `map` folder store the maps generated for training and testing.
The `env` folder contains the simulation environment implementation.
The `weights`  folder stores the demo model weights `final_weight.pth`.
The `agent.py` contains the preprocess of the state, whose output can be directly sent to the model.



### Training 
To use RL(AB_Mapper) train the model, simply run:
```
python train.py 
```
where the arguments are:\
`lr` - learning rate \
`obs` - number of dynamic obstacles \
`agents` - number of agents \
`interval` - number of iterations after which model weights get updated to the best among each agent \
`exp` - experiment number \
`load` - load base weights to start training (if needed) \
`entropy-weight` - coefficient of cross entropy in the calculation of loss \
`goal-range` - maximum distance to the goal of each agent from its starting pose \
`agent-type` - 0 (Full MAPPER), 1 (MAPPER w/o global waypoints), 2 (MAPPER w/o dynamic obstacle trajectory), 3 (MAPPER w/o sub goal guidance) 
`tau (float)` - Target update rate \
`pi_lr (float)` - Learning rate for policy \
`q_lr (float)` - Learning rate for critic \
`reward_scale (float)`- Scaling for reward (has effect of optimal   policy entropy) \
`hidden_dim (int)`- Number of hidden dimensions for networks \
`attention_heads` - number of attention head \ 
`sub_num_agent` - Each agent selects the nearest sub_num_agent allocates attention to two agents \


Ps:if you want to run environment"20_20" experiment, you need to overwrite the "grid_env" file in the first level folder with the "grid_env" file in the "env" folder
More arguments can be found in the code.
