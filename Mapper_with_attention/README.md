# Multi-Agent path planning with reinforcement learning


### Structure
The `map` folder store the maps generated for training and testing.
The `env` folder contains the simulation environment implementation.
The `weights`  folder stores the demo model weights `final_weight.pth`.
The `agent.py` contains the preprocess of the state, whose output can be directly sent to the model.
The `model1.py` contains the A2C neural network model.




### Training 
simply run:

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

Begin by training the A2C model without a base model with the number of agents and number of dynamic obstacles less than 10. After getting a sufficiently converged agent, stop the training. Use the trained weights as base weights to train the agent in a more complex environment with higher number of agents and obstacles. Keep iterating this procedure until the agent performs well in your desired environment setting.


