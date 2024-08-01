



def get_objs(info, env_name):
    
    if env_name == 'Ant-v3':
        obj1, obj2 = get_ant_objs(info)

    if env_name == 'HalfCheetah-v3':
        obj1, obj2 = get_cheetah_objs(info)

    if env_name == 'MOwalker2d-v3':
        obj1, obj2 = get_walker_objs(info)

    if env_name == 'MOHopper-v3':
        obj1, obj2 = get_walker_objs(info)

    if env_name == 'Humanoid-v3':
        obj1, obj2 = get_humanoid_objs(info)

    return obj1, obj2



def get_ant_objs(info):
    
    healthy_reward = info['reward_survive']

    forward_reward = info['x_velocity']        
    y_reward = info['y_velocity']       

    rewards1 = forward_reward + healthy_reward
    rewards2 = y_reward + healthy_reward

    costs = info['reward_ctrl'] + info['reward_contact']

    obj1 = (rewards1 + costs - 2) * 2
    obj2 = (rewards2 + costs - 2) * 2

    return obj1, obj2


def get_cheetah_objs(info):
    
    obj1 = info['reward_run'] * 0.5
    obj2 = 4 + info['reward_ctrl'] * 10

    return obj1, obj2



def get_walker_objs(info):
    
    obj1 = info['rewards'] 
    obj2 = (2 - info['costs'] * 1000) * 3

    return obj1, obj2


def get_humanoid_objs(info):
    
    healthy_reward = info['reward_alive']

    forward_reward = info['x_velocity']        
    y_reward = info['y_velocity']       

    rewards1 = forward_reward + healthy_reward

    costs = info['reward_quadctrl'] + info['reward_impact']

    obj1 = (rewards1 - 5) * 7
    obj2 = 20 + costs * 100
    
    return obj1, obj2
