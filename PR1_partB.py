#!/usr/bin/env python
# coding: utf-8

# In[1]:


from utils import *
from example import example_use_of_gym_env
import os


# In[2]:


MF = 0  # Move Forward
TL = 1  # Turn Left
TR = 2  # Turn Right
PK = 3  # Pickup Key
UD = 4  # Unlock Door


# In[3]:


def move_forward(x,y,orient): # Function to calculate the next if the action is Move Forward
    if orient == 0:  # North
        return x, y-1, orient
    elif orient == 1:  # South
        return x, y+1, orient
    elif orient == 2:  # East
        return x+1, y, orient
    elif orient == 3:  # West
        return x-1, y, orient


# In[4]:


def turn_left(orient):
    return {0: 3, 3: 1, 1: 2, 2: 0}[orient]

def turn_right(orient):
    return {0: 2, 2: 1, 1: 3, 3: 0}[orient]

def get_next_state(x, y, orient, action, has_key, key_pos, DOOR_POSITIONS,door1_open,door2_open, goal_pos):
    
    
    door_states = [door1_open, door2_open]
    x_new,y_new,orient_new = move_forward(x,y,orient)

    if action == MF:
        
        
        
        if x_new <0 or y_new<0 or x_new >7 or y_new>7:  # Checking if the next state lies outside the grid
            return x, y, orient, has_key, door1_open,door2_open
        elif (x_new,y_new) in DOOR_POSITIONS: # Checking if the next state contains a door
            door_idx = DOOR_POSITIONS.index((x_new, y_new))
            if door_states[door_idx] == 0:    # If the door is locked, staty at the same state
                return x, y, orient, has_key, door1_open,door2_open
            else:  # If the door is opened, move forward
                return x_new, y_new, orient_new, has_key, door1_open,door2_open
    
        if(x_new==4 and (y_new!=2 and y_new!=5)): # Check if the next state is wall and return the same state
            return x, y, orient, has_key, door1_open,door2_open
        else:
            return x_new, y_new, orient_new, has_key, door1_open,door2_open
    
    elif action == TL:  # To turn Left
        return x, y, turn_left(orient), has_key, door1_open,door2_open
    elif action == TR:  # To turn Right
        return x, y, turn_right(orient), has_key, door1_open,door2_open
    elif action == PK:
        
        if (x_new==key_pos[0] and y_new == key_pos[1]): # Update the agent that it has picked the key if the next state has a key
            has_key = 1
        return x, y, orient, has_key, door1_open,door2_open
        
    elif action == UD: # To toggle the door states if the agent has the key
        if (x_new,y_new) == DOOR_POSITIONS[0] and has_key == 1:
            door1_open = (door1_open+1)%2
            
        elif (x_new,y_new) == DOOR_POSITIONS[1] and has_key == 1:
            door2_open = (door2_open+1)%2
        
        return x, y, orient, has_key, door1_open,door2_open

    return x, y, orient, has_key, door1_open,door2_open



# In[5]:


def stage_cost(x,y, goal_pos):      
    if(x==goal_pos[0] and y == goal_pos[1]): # Stage cost is 0 for the goal position - lower cost
        return 0
    else:
        return 1  # Higher cost for the states except the goal state


# In[6]:


def dynamic_programming_partB():
    
    ACTIONS = [MF, TL, TR, PK, UD]
    KEY_LOCATIONS = [(1, 1), (2, 3), (1, 6)]
    GOAL_LOCATIONS = [(5, 1), (6, 3), (5, 6)]
    DOOR_POSITIONS = [(4, 2), (4, 5)]
    WIDTH = 8
    HEIGHT = 8
    GAMMA = 1
    

    V = np.full((WIDTH, HEIGHT, 4, 2, 2, 2, len(KEY_LOCATIONS), len(GOAL_LOCATIONS)), np.inf) # Define Value function
    pi = np.full((WIDTH, HEIGHT, 4, 2, 2, 2, len(KEY_LOCATIONS), len(GOAL_LOCATIONS)), None)  # Define Policy

    # Terminal conditions
    for goal_pos in GOAL_LOCATIONS: # Terminal cost for goal is zero
        V[goal_pos[0], goal_pos[1], :, :, :, :, :, :] = 0


    
    state_action_pairs = []
    next_states = np.zeros((WIDTH, HEIGHT, 4, 2, 2, 2, len(KEY_LOCATIONS), len(GOAL_LOCATIONS), len(ACTIONS), 6), dtype=int)
    for x in range(WIDTH):
        for y in range(HEIGHT):
            for orient in range(4):
                for has_key in range(2):
                    for door1_open in range(2):
                        for door2_open in range(2):
                            for key_pos in range(len(KEY_LOCATIONS)):
                                for goal_pos in range(len(GOAL_LOCATIONS)):
                                    for action in ACTIONS:
                                        state_action_pairs.append((x, y, orient, has_key, door1_open, door2_open, key_pos, goal_pos, action))
                                        next_states[x, y, orient, has_key, door1_open, door2_open, key_pos, goal_pos, action] = get_next_state(x, y, orient, action, has_key, KEY_LOCATIONS[key_pos], DOOR_POSITIONS, door1_open, door2_open, GOAL_LOCATIONS[goal_pos])

    # Dynamic Programming
    while True:
        Q = np.zeros((len(state_action_pairs)))
        for idx, (x, y, orient, has_key, door1_open, door2_open, key_pos, goal_pos, action) in enumerate(state_action_pairs): # looping over all possible states
            x_prime, y_prime, orient_prime, has_key_prime, door1_open_prime, door2_open_prime = next_states[x, y, orient, has_key, door1_open, door2_open, key_pos, goal_pos, action]
            Q[idx] = stage_cost(x, y, GOAL_LOCATIONS[goal_pos]) + GAMMA * V[x_prime, y_prime, orient_prime, has_key_prime, door1_open_prime, door2_open_prime, key_pos, goal_pos]
        
        Q_reshaped = Q.reshape((WIDTH, HEIGHT, 4, 2, 2, 2, len(KEY_LOCATIONS), len(GOAL_LOCATIONS), len(ACTIONS)))
        
        V_new = np.min(Q_reshaped, axis=-1)     # Find the minimum cost for an action  for each state
        pi_new = np.argmin(Q_reshaped, axis=-1) # Find the best possible action for each state
        
        if(np.allclose(V,V_new)): # Check if the new minimum value function is same as the previously calculated value function and break the loop.
            break
        V = V_new    # Updating Value Function for all states
        pi = pi_new  # Updating Policy for all states

    return V, pi


# In[7]:


ACTIONS = [MF, TL, TR, PK, UD]
KEY_LOCATIONS = [(1, 1), (2, 3), (1, 6)]
GOAL_LOCATIONS = [(5, 1), (6, 3), (5, 6)]
DOOR_POSITIONS = [(4, 2), (4, 5)]


# ## Dynamic Programming Part(B)

# In[8]:


Value_function,Single_control_policy = dynamic_programming_partB()


# ## "Single_control_policy" - Variable to acccess
# - Access this variable to get the policy for any environment.
# - Example of using this single control policy for an environment is given below.
# - States = [ x, y, orientation, has_key_or_not, door_1_open, door_2_open, Key_position, Goal_posiiton] 
# 

# In[9]:


env_folder = "envs/random_envs"
env, info, env_path = load_random_env(env_folder)
env_name, _ = os.path.splitext(os.path.basename(env_path))  # load an environment



#Intializing the start position and the environment
x = info['init_agent_pos'][0]
y = info['init_agent_pos'][1]
orient = 0
has_key =0 


door1_open = 1 if info["door_open"][0] else 0
door2_open = 1 if info["door_open"][1] else 0
key_pos = info["key_pos"]
goal_pos = info["goal_pos"]

key_pos_tuple = tuple(key_pos)
goal_pos_tuple = tuple(goal_pos)
key_idx = KEY_LOCATIONS.index(key_pos_tuple)
goal_idx = GOAL_LOCATIONS.index(goal_pos_tuple)

#Optimal Policy is retrieved here
optimal_policy = []

while True:
    
    optimal_policy.append(Single_control_policy[x,y,orient,has_key,door1_open,door2_open,key_idx,goal_idx])
    x,y,orient,has_key,door1_open,door2_open= get_next_state(x,y,orient,Single_control_policy[x,y,orient,has_key,door1_open,door2_open,key_idx,goal_idx],has_key, key_pos, DOOR_POSITIONS,door1_open,door2_open, goal_pos)
    print(x,y,orient)
    if (x==info['goal_pos'][0] and y==info['goal_pos'][1]):
        optimal_policy.append(Single_control_policy[x,y,orient, has_key,door1_open,door2_open,key_idx,goal_idx])
        break
    
print(optimal_policy)
draw_gif_from_seq(optimal_policy, load_env(env_path)[0],path=f"./gif/PartB/{env_name}.gif")
    


# ## Environment changes after each state
# See if the agent is reaching the goal or not by adapting to the environment.
# - Each state is printed
# - Goal Reached will be printed if the goal is reached else the program will be running in infinte loop
# - Optimal policy will be printed

# In[11]:

print("When the environment changes after each state: ")
env_folder = "envs/random_envs"
env, info, env_path = load_random_env(env_folder)
env_name, _ = os.path.splitext(os.path.basename(env_path))  # load an environment



#Intializing the start position and the environment
x = info['init_agent_pos'][0]
y = info['init_agent_pos'][1]
orient = 0
has_key =0 


door1_open = 1 if info["door_open"][0] else 0
door2_open = 1 if info["door_open"][1] else 0
key_pos = info["key_pos"]
goal_pos = info["goal_pos"]

key_pos_tuple = tuple(key_pos)
goal_pos_tuple = tuple(goal_pos)
key_idx = KEY_LOCATIONS.index(key_pos_tuple)
goal_idx = GOAL_LOCATIONS.index(goal_pos_tuple)

#Optimal Policy is retrieved here
optimal_policy = []

while True:

    optimal_policy.append(Single_control_policy[x,y,orient,has_key,door1_open,door2_open,key_idx,goal_idx])
    x,y,orient,has_key,door1_open,door2_open= get_next_state(x,y,orient,Single_control_policy[x,y,orient,has_key,door1_open,door2_open,key_idx,goal_idx],has_key, key_pos, DOOR_POSITIONS,door1_open,door2_open, goal_pos)
    print(x,y,orient)
    env_folder = "envs/random_envs"
    env, info, env_path = load_random_env(env_folder)
    door1_open = 1 if info["door_open"][0] else 0
    door2_open = 1 if info["door_open"][1] else 0
    key_pos = info["key_pos"]
    goal_pos = info["goal_pos"]

    key_pos_tuple = tuple(key_pos)
    goal_pos_tuple = tuple(goal_pos)
    key_idx = KEY_LOCATIONS.index(key_pos_tuple)
    goal_idx = GOAL_LOCATIONS.index(goal_pos_tuple)
    if (x==info['goal_pos'][0] and y==info['goal_pos'][1]):
        optimal_policy.append(Single_control_policy[x,y,orient, has_key,door1_open,door2_open,key_idx,goal_idx])
        break
    
print(f"Goal Reached [{x}, {y}]")
print("Optimal_Policy",optimal_policy)
    

