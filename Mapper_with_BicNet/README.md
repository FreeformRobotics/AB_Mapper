# Multi-Agent path planning with reinforcement learning


### Structure
The `map` folder store the maps generated for training and testing.
The `env` folder contains the simulation environment implementation.
The `weights`  folder stores the demo model weights `final_weight.pth`.
The `agent.py` contains the preprocess of the state, whose output can be directly sent to the model.
The `model1.py` contains the A2C neural network model.

### Generate Custom Map
Create an image using a photo editing software with each pizel representing a grid cell where white pixels are free cells and black pixels represent obstacles. Copy the created image to the `misc/generate_map` folder and edit the name of the image in the `misc/generate_map/generate_grid.py` script and run the script to create the .yaml file of the custom map.


### Training 
To use RL(A2C) train the model, simply run:
```
python train.py --map="random_map" --seed=999 --lr=0.0003 --obs=5 --agents=8 \
--interval=100 --exp=8 --model-dir="weights" --load=final_weight.pth --entropy-weight=0.01 \
--goal-range=7 --agent-type=0
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

### Testing
To test the model in multi-agent case, use
```
python test.py --map=easymap --seed=999 --obs=70 --agents=30 \
--agent-type=0 --episode=30 --model-dir="weights" --load=final_weight.pth 
```

If you need to store the rendering image to a given folder, set `--save-img=1 --save-img-dir=exp1`. Then use script `utils/save_video.py` to convert the saved images to video: 
```
python utils/save_video.py --dir=exp1 --name=exp1 --fps=5
```
More arguments can be found in the code.
