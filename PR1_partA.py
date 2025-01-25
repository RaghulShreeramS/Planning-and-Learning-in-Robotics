#!/usr/bin/env python
# coding: utf-8

# In[82]:


from utils import *
from example import example_use_of_gym_env


# In[83]:


MF = 0  # Move Forward
TL = 1  # Turn Left
TR = 2  # Turn Right
PK = 3  # Pickup Keya
UD = 4  # Unlock Door


# In[84]:


def move_forward(x,y,orient): # Function to calculate the next if the action is Move Forward
    if orient == 0:  # North
        return x, y-1, orient
    elif orient == 1:  # South
        return x, y+1, orient
    elif orient == 2:  # East
        return x+1, y, orient
    elif orient == 3:  # West
        return x-1, y, orient


# In[85]:


def turn_left(orient):
    return {0: 3, 3: 1, 1: 2, 2: 0}[orient]

def turn_right(orient):
    return {0: 2, 2: 1, 1: 3, 3: 0}[orient]

def get_next_state(x, y, orient, action, has_key, KEY_POSITION, DOOR_POSITION,door_open, env):
    
    x_new,y_new,orient_new = move_forward(x,y,orient)

    if action == MF:
        
        if x_new <1 or y_new<1 or x_new >env.grid.width-1 or y_new>env.grid.height-1:  # Checking if the next state lies outside the grid
            return x, y, orient, has_key, door_open
        elif (x_new,y_new) == DOOR_POSITION: # Checking if the next state contains a door
            if door_open == 0:    # If the door is locked, staty at the same state
                return x, y, orient, has_key, door_open
            else:  # If the door is opened, move forward
                return x_new, y_new, orient_new, has_key, door_open
        cell = env.grid.get(x_new,y_new)
        if(cell!= None):
            if(cell.type == "wall"): # Check if the next state is wall and return the same state
                return x, y, orient, has_key, door_open
            else:
                return x_new, y_new, orient_new, has_key, door_open
        
        else:
            return x_new, y_new, orient_new, has_key, door_open
    elif action == TL:  # To turn Left
        return x, y, turn_left(orient), has_key, door_open
    elif action == TR:  # To turn Right
        return x, y, turn_right(orient), has_key, door_open
    elif action == PK:
        if (x_new==KEY_POSITION[0] and y_new == KEY_POSITION[1]): # Update the agent that it has picked the key if the next state has a key
            has_key = 1
            return x, y, orient, has_key, door_open
        else:
            return x, y, orient, has_key, door_open
        
    elif action == UD: # To toggle the door states if the agent has the key
        if (x_new,y_new) == DOOR_POSITION and has_key == 1:
            door_open = (door_open+1)%2
            
        return x, y, orient, has_key, door_open

    return x, y, orient, has_key, door_open


# In[86]:


def stage_cost(x,y,goal_pos):    
    if(x==goal_pos[0] and y == goal_pos[1]): # Stage cost is 0 for the goal position - lower cost
        return 0
    else:
        return 1  # Higher cost for the states except the goal state


# In[87]:


def dynamic_programming(env, info):
    
    ACTIONS = [MF, TL, TR, PK, UD]

    WIDTH = info["width"]
    HEIGHT = info["height"]
    DOOR_POSITION = tuple(info["door_pos"])
    KEY_POSITION = info["key_pos"]
    gamma = 1
    goal_pos = tuple(info["goal_pos"])

    pi = np.full((WIDTH, HEIGHT, 4, 2, 2), None)  # Define Policy
    V = np.full((WIDTH, HEIGHT, 4, 2, 2), np.inf) # Define Value function

    # Terminal conditions
    V[goal_pos[0], goal_pos[1], :, :, :] = 0  # Terminal cost for goal is zero

    # Precompute all state-action pairs
    state_action_pairs = []
    for x in range(WIDTH):
        for y in range(HEIGHT):
            for orient in range(4):
                for has_key in range(2):
                    for door_open in range(2):
                        for action in ACTIONS:
                            state_action_pairs.append((x, y, orient, has_key, door_open, action))
    
    
  
    next_states = np.zeros((WIDTH, HEIGHT, 4, 2, 2, len(ACTIONS), 5), dtype=int)
    for idx, (x, y, orient, has_key, door_open, action) in enumerate(state_action_pairs):
        next_states[x, y, orient, has_key, door_open, action] = get_next_state(x, y, orient, action, has_key, KEY_POSITION, DOOR_POSITION, door_open,env)
    
    # Dynamic Programming
    while True:
        Q = np.zeros((len(state_action_pairs)))
        for idx, (x, y, orient, has_key, door_open, action) in enumerate(state_action_pairs): # looping over all possible states
            x_prime, y_prime, orient_prime, has_key_prime, door_open_prime = next_states[x, y, orient, has_key, door_open, action]
            Q[idx] = stage_cost(x, y, goal_pos) + gamma * V[x_prime, y_prime, orient_prime, has_key_prime, door_open_prime] 
        
        Q_reshaped = Q.reshape((WIDTH, HEIGHT, 4, 2, 2, len(ACTIONS)))
        V_new = np.min(Q_reshaped, axis=-1)     # Find the minimum cost for an action  for each state
        pi_new = np.argmin(Q_reshaped, axis=-1) # Find the best possible action for each state
        
        if(np.allclose(V,V_new)): # Check if the new minimum value function is same as the previously calculated value function and break the loop. 
            break
        V = V_new    # Updating Value Function for all states
        pi = pi_new  # Updating Policy for all states

    return V, pi


# In[88]:


def partA(env,env_path,info,env_name):

    V,pi = dynamic_programming(env,info)
    directions = {
    (0, -1): 0,  # North
    (1, 0): 2,   # East
    (0, 1): 1,   # South
    (-1, 0): 3   # West
    }
    
    #Intializing the start position and the environment
    x = info['init_agent_pos'][0]
    y = info['init_agent_pos'][1]
    orient = directions.get(tuple(info["init_agent_dir"]))
    has_key = 0
    door_open = 0
    DOOR_POSITION = tuple(info["door_pos"])
    KEY_POSITION = info["key_pos"]
    goal_pos = ((info["goal_pos"][0], info["goal_pos"][1]))

    #Optimal Policy is retrieved for a specific environment
    optimal_policy = []
    while True:
        print(x,y,orient)
        optimal_policy.append(pi[x,y,orient,has_key,door_open])
        x,y,orient,has_key,door_open = get_next_state(x,y,orient,pi[x,y,orient,has_key,door_open],has_key,KEY_POSITION, DOOR_POSITION, door_open, env)
        
        if (x==goal_pos[0] and y==goal_pos[1]):
            optimal_policy.append(pi[x,y,orient,has_key,door_open])
            break
    
    draw_gif_from_seq(optimal_policy, load_env(env_path)[0], path=f"./gif/PartA/{env_name}.gif") 

    return V, optimal_policy 


# In[89]:


env_paths = {
    "doorkey-5x5-normal": "envs/known_envs/doorkey-5x5-normal.env",
    "doorkey-6x6-direct": "envs/known_envs/doorkey-6x6-direct.env",
    "doorkey-6x6-normal": "envs/known_envs/doorkey-6x6-normal.env",
    "doorkey-6x6-shortcut": "envs/known_envs/doorkey-6x6-shortcut.env",
    "doorkey-8x8-direct": "envs/known_envs/doorkey-8x8-direct.env",
    "doorkey-8x8-normal": "envs/known_envs/doorkey-8x8-normal.env",
    "doorkey-8x8-shortcut": "envs/known_envs/doorkey-8x8-shortcut.env",
    "example-8x8": "envs/known_envs/example-8x8.env"
}
env_name = "doorkey-8x8-normal" # Select the environment you want from env_paths defined above
env_path = env_paths[env_name]

env, info = load_env(env_path)  # load an environment
print("partA info: ",np.array(info))
    
Value_function, Control_Policy = partA(env,env_path,info,env_name) # Control Policy is returned here
print(Control_Policy)

