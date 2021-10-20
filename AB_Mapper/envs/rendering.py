from utils import *
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import matplotlib.widgets
from random import randint

OBSTACLE_COLOR = np.array([102,102,102])
GOAL_COLOR = np.array([25,25,25])
AGENT_COLOR = np.array([255,102,102])
DYNAMIC_OBS_COLOR = np.array([102,153,255])
FREESPACE_COLOR = np.array([224,224,224]) 
TRAJ_COLOR = np.array([153,153,153]) 
Attention_COLOR = np.array([0,0,255])

OBSTACLE_ICON = 'icons/shelf'
GOAL_ICON =  'icons/shelf'
AGENT_ICON =  'icons/walle'
DYNAMIC_OBS_ICON =  'icons/cargo3'

class Renderer():
    def __init__(self, row, col, tilesize, traj_color):
        self.row = row
        self.col = col
        self.tilesize = tilesize
        self.traj_color = traj_color

    def draw_background(self, img, agents_goal, obstacles):
        img = self.draw_goal(img, agents_goal)
        img = self.draw_static_obstacle(img, obstacles)
        return img

    def draw(self, img, i, j, obj, length = 1):
        xmin = i*self.tilesize
        ymin = j*self.tilesize
        xmax = (i+length)*self.tilesize
        ymax = (j+length)*self.tilesize
        tile = img[xmin:xmax, ymin:ymax, :]
        tile = obj.render(tile)
        img[xmin:xmax, ymin:ymax, :] = tile
        return img

    def draw_static_obstacle(self, img, obstacles):
        for o in obstacles:
            img = self.draw(img, int(o[1]), int(o[0]), Obstacle())
        return img

    def draw_goal(self, img, agents_goal):
        for goal in agents_goal:
            img = self.draw(img, int(goal[1]), int(goal[0]), Goal())
        return img

    def draw_agents(self, img, agents_pose, agent_on_goal):
        agents_num = len(agents_pose)
        for i in range(agents_num):
            pos = agents_pose[i]
            name = str(i)
            ## TO DO: add state
            if (len(agent_on_goal) == agents_num): state = agent_on_goal[i] 
            else: state = -1
            
            img = self.draw(img, int(pos[1]), int(pos[0]), Agent(name, state))

        return img
          
    def draw_dynamic_obs_debug(self, img, agents_pose):
        agents_num = len(agents_pose)
        for i in range(agents_num):
            pos = agents_pose[i]
            name = str(i)
            img = self.draw(img, int(pos[1]), int(pos[0]), Agent(name))
        return img

    def draw_dynamic_obs(self, img, obs):
        for ob in obs:
            img = self.draw(img, int(ob[1]), int(ob[0]), Dynamic_obs())
        return img

    def draw_obs(self, img, agents_pose, visibility):
        agents_num = len(agents_pose)
        for i in range(agents_num):
            pos = agents_pose[i]

            xmin = max(int(pos[1]-visibility)*self.tilesize, 0)
            ymin = max(int(pos[0]-visibility)*self.tilesize, 0)
            xmax = min(int(pos[1]+1+visibility)*self.tilesize, self.tilesize*(self.row+1))
            ymax = min(int(pos[0]+1+visibility)*self.tilesize, self.tilesize*(self.col+1))

            tile = img[xmin:xmax, ymin:ymax, :]
            tile = highlight_img(tile)
            img[xmin:xmax, ymin:ymax, :] = tile
        return img  

    def draw_trajectory(self, img, traj):
      agents_num = len(traj)
      for idx in range(agents_num):
          # trajectory = traj[idx][-3:]
          trajectory = traj[idx][-1:]
          if not trajectory: continue
          for i in range(len(trajectory)-1):
              if not np.array_equal(trajectory[i], trajectory[i+1]):
                  p1 = trajectory[i]
                  p2 = trajectory[i+1]
                  print('\np1',p1,'p2',p2)
                  img = draw_traj(img, ((p1[0]+0.5)*self.tilesize, (p1[1]+0.5)*self.tilesize), 
                            ((p2[0]+0.5)*self.tilesize,(p2[1]+0.5)*self.tilesize), TRAJ_COLOR)

                  # img = draw_traj(img, ((p1[0]+0.5)*self.tilesize, (p1[1]+0.5)*self.tilesize),
                  #           ((p2[0]+0.5)*self.tilesize,(p2[1]+0.5)*self.tilesize), Attention_COLOR)

    def draw_attention(self,img,agent_pose,agent,subagentlist):
        """
        agent is index

        """
        color_list =[
            np.array([255,0,0]),np.array([0,0,0]),np.array([0,0,255])
        ]
        k=0
        for i in agent:
            x1=int(agent_pose[i][0])
            y1=int(agent_pose[i][1])
            # print('x1',x1,'y1',y1)
            j=0
            for p in subagentlist[k]:
                    # print('P',p)
                # for j in len(p):
                    if j <=3:w=2
                    elif j<=7:w=1
                    elif j>=8:w=0
                    x2 = int(agent_pose[p][0])
                    y2 = int(agent_pose[p][1])
                    img =draw_attention_line(img,((x1+0.5)*self.tilesize,(y1+0.5)*self.tilesize),((x2+0.5)*self.tilesize,(y2+0.5)*self.tilesize),color_list[k],w)
            k+=1

class Obstacle():
    def __init__(self):
        self.color = OBSTACLE_COLOR
        #self.icon = load_PNG(OBSTACLE_ICON+str(randint(1,2))+'.png')
        #self.type = 2
    def render(self, img):
      fill_coords(img, point_in_rect(0, 1, 0, 1), self.color)
      return img#self.icon

class Goal():
    def __init__(self):
        self.color = GOAL_COLOR
        #self.type = 3
    def render(self, img):
      fill_coords(img, point_in_rect(0.2, 0.8, 0.2, 0.8), self.color)
      return img

class Agent():
    def __init__(self, name = None, state = -1):
        self.color = AGENT_COLOR
        self.name = name
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        #self.icon = load_PNG(AGENT_ICON+'.png')
        self.state = state
        #self.type = 4
    def render(self, img):
        #if (self.state >= 3): return img
        fill_coords(img, point_in_circle(0.5, 0.5, 0.25), self.color, self.state)
        #img = self.icon
        scale = img.shape[0]/96
        self.add_text(img,self.name, fontScale = scale, thickness = 1)
        #fade(img, state)
        return img

    def add_text(self, img, text, fontScale = 1, thickness = 2):
        textsize = cv2.getTextSize(text, self.font, fontScale, thickness)[0]
        textX = int(img.shape[0]/2 - textsize[0]/2)
        textY = int(img.shape[1]/2 + textsize[1]/2)
        cv2.putText(img, text, (textX, textY), self.font, fontScale, (0, 0, 0), thickness)

class Dynamic_obs():
    def __init__(self):
        self.color = DYNAMIC_OBS_COLOR
        #self.icon = load_PNG(DYNAMIC_OBS_ICON+'.png')
        #self.type = 5
    def render(self, img):
        #print(img.shape)
        fill_coords(img, point_in_triangle((0.5, 0.15), (0.9, 0.85), (0.1, 0.85),), self.color)
        #fill_coords(img, point_on_side(0.1), self.color)
        return img
        #return self.icon
class Window:
    """
    Window to draw a gridworld instance using Matplotlib
    """

    def __init__(self, title):
        self.fig = None

        self.imshow_obj = None

        # Create the figure and axes
        self.fig, self.ax = plt.subplots()

        # Show the env name in the window title
        self.fig.canvas.set_window_title(title)

        # Turn off x/y axis numbering/ticks
        self.ax.set_xticks([], [])
        self.ax.set_yticks([], [])

        # Flag indicating the window was closed
        self.closed = False

        def close_handler(evt):
            self.closed = True

        self.fig.canvas.mpl_connect('close_event', close_handler)

    def show_img(self, img):
        """
        Show an image or update the image being shown
        """

        # Show the first image of the environment
        if self.imshow_obj is None:
            self.imshow_obj = self.ax.imshow(img, interpolation='bilinear')

        self.imshow_obj.set_data(img)
        self.fig.canvas.draw()

        # Let matplotlib process UI events
        # This is needed for interactive mode to work properly
        plt.pause(0.001)

    def set_caption(self, text):
        """
        Set/update the caption text below the image
        """

        plt.xlabel(text)

    def reg_key_handler(self, key_handler):
        """
        Register a keyboard event handler
        """

        # Keyboard handler
        self.fig.canvas.mpl_connect('key_press_event', key_handler)

    def show(self, block=True):
        """
        Show the window, and start an event loop
        """

        # If not blocking, trigger interactive mode
        if not block:
            plt.ion()

        # Show the plot
        # In non-interative mode, this enters the matplotlib event loop
        # In interactive mode, this call does not block
        plt.show()

    def close(self):
        """
        Close the window
        """

        plt.close()
