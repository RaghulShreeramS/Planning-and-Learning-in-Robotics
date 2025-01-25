from utils import *
from example import example_use_of_gym_env

MF = 0  # Move Forward
TL = 1  # Turn Left
TR = 2  # Turn Right
PK = 3  # Pickup Key
UD = 4  # Unlock Door


def doorkey_problem(env):
    """
    You are required to find the optimal path in
        doorkey-5x5-normal.env
        doorkey-6x6-normal.env
        doorkey-8x8-normal.env

        doorkey-6x6-direct.env
        doorkey-8x8-direct.env

        doorkey-6x6-shortcut.env
        doorkey-8x8-shortcut.env

    Feel Free to modify this fuction
    """
    optim_act_seq = [TL, MF, PK, TL, UD, MF, MF, MF, MF, TR, MF]
    return optim_act_seq

def turn_left(orient):
    return {'N': 'W', 'W': 'S', 'S': 'E', 'E': 'N'}[orient]

def turn_right(orient):
    return {'N': 'E', 'E': 'S', 'S': 'W', 'W': 'N'}[orient]
def get_next_state(x, y, orient, action):
    """ Calculate the next state based on the current orientation and action taken """
    if action == 'MF':
        if orient == 'N':
            return x, y-1, orient
        elif orient == 'S':
            return x, y+1, orient
        elif orient == 'E':
            return x+1, y, orient
        elif orient == 'W':
            return x-1, y, orient
    elif action == 'TL':
        return x, y, turn_left(orient)
    elif action == 'TR':
        return x, y, turn_right(orient)
    return x, y, orient

def partA():
    env_path = "starter_code/envs/known_envs/example-8x8.env"
    env, info = load_env(env_path)  # load an environment
    print(f"Loaded environment from: {env_path}")
    print(f"Environment dimensions: {info['width']}x{info['height']}")
    print(f"Initial agent position: {info['init_agent_pos']} and direction: {info['init_agent_dir']}")
    print(info)
    seq = doorkey_problem(env)  # find the optimal action sequence
    draw_gif_from_seq(seq, load_env(env_path)[0])  # draw a GIF & save


def partB():
    env_folder = "starter_code/envs/random_envs"
    env, info, env_path = load_random_env(env_folder)
    print("partB info: ",info)


if __name__ == "__main__":
    # example_use_of_gym_env()
    partA()
    partB()

