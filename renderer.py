import os, sys
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(base_dir)
import gym
import mo_envs



env = gym.make('MOHumanoid-v0')

for i_episode in range(1000):

    done = False
    
    obs = env.reset()

    while not done:
        action = env.action_space.sample()  # Sample random action

        next_obs, reward, done, _ = env.step(action) # Step
        env.render()