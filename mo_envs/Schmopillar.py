import os
import gym
from gym import utils
from gym.envs.robotics import fetch_env
import numpy as np
from gym.envs.robotics import rotations, robot_env
from gym.envs.robotics import utils as roboutils

import pickle

# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "assets/Schmopillar.xml")

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)

class SchmopillarEnv(robot_env.RobotEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
            # 'object0:joint': [1.25, 0.53, 0.7, 1., 0., 0., 0.],
            'joint1': 0, #np.deg2rad(np.random.randint(low=0, high=90)),
            'joint2': 0, #np.deg2rad(np.random.randint(low=35, high=45)),
            'joint3': 1.7, #np.deg2rad(np.random.randint(low=35, high=45)),
            'joint4': 0,
            'joint5': 2, #np.deg2rad(np.random.randint(low=80, high=90)),
            'joint6': 0,
        }

        self.joint_names = ['joint1', 'joint2', 'joint3', 'joint5', 'right_finger']
        self.rot = [1., 1., 1., 1.]

        self.n_actions = 2

        self.gripper_extra_height = 0.2
        self.block_gripper = False
        self.has_object = True
        self.target_in_the_air = True
        self.target_offset = 0.0
        self.obj_range = 0.15
        self.target_range = 0.15
        self.distance_threshold = 0.05
        self.reward_type = reward_type
        
        self.jogs = []
        
        super(SchmopillarEnv, self).__init__(
            model_path=MODEL_XML_PATH, n_substeps=20, n_actions=self.n_actions,
            initial_qpos=initial_qpos)

        utils.EzPickle.__init__(self)

    def compute_reward(self, achieved_goal, goal, info):

        # obj1 : moving blue boxes
        # obj2 : do not touch red boxes
        
        bluebox1_xy = self.sim.data.get_site_xpos('bluebox1')[:2]
        bluebox2_xy = self.sim.data.get_site_xpos('bluebox2')[:2]

        redbox1_xy = self.sim.data.get_site_xpos('redbox1')[:2]
        redbox2_xy = self.sim.data.get_site_xpos('redbox2')[:2]
        redbox3_xy = self.sim.data.get_site_xpos('redbox3')[:2]
        
        obj1 = (0.1 - np.linalg.norm(redbox1_xy - np.array([0.9, 0]), axis=-1)) * 10

        obj2 = (0.1 - np.linalg.norm(redbox1_xy - bluebox1_xy, axis=-1) - np.linalg.norm(redbox1_xy - redbox2_xy, axis=-1) - np.linalg.norm(redbox1_xy - bluebox2_xy, axis=-1) - np.linalg.norm(redbox1_xy - redbox3_xy, axis=-1)) * 6
        obj2 = obj2 * 0.5

        reward = {
            "reward_obj1": obj1,
            "reward_obj2": obj2,
        }

        return reward 

    def _set_action(self, action):
        assert action.shape == (self.n_actions,)
        action = action.copy()  

        pos_ctrl = np.concatenate([action[:2], [0]])   
        pos_ctrl = pos_ctrl * 0.025

        # grip_action = action[2]
            
        gripper_ctrl = np.array([-1, 1])

        # gripper_ctrl = gripper_ctrl * grip_action

        action = np.concatenate([pos_ctrl, self.rot, gripper_ctrl])

        # Apply action to simulation.
        roboutils.ctrl_set_action(self.sim, action)
        roboutils.mocap_set_action(self.sim, action)

    def _get_obs(self):

        grip_pos = self.sim.data.get_site_xpos('grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep

        gripper_state = [self.sim.data.get_joint_qpos('right_finger'), self.sim.data.get_joint_qpos('left_finger')]

        bluebox1_pos = self.sim.data.get_site_xpos('bluebox1')[:2]
        bluebox2_pos = self.sim.data.get_site_xpos('bluebox2')[:2]
        redbox1_pos = self.sim.data.get_site_xpos('redbox1')[:2]
        redbox2_pos = self.sim.data.get_site_xpos('redbox2')[:2]
        redbox3_pos = self.sim.data.get_site_xpos('redbox3')[:2]

        achieved_goal = np.squeeze(grip_pos.copy())

        obs = np.concatenate([
            grip_pos[:2], bluebox1_pos.ravel()[:2], bluebox2_pos.ravel()[:2], redbox1_pos.ravel()[:2], redbox2_pos.ravel()[:2], redbox3_pos.ravel()[:2]
        ])

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }

    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id('link')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 180.
        self.viewer.cam.elevation = -20.

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id('target0')
        self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]
        self.sim.forward()

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)        
        self.sim.forward()
        return True

    def _sample_goal(self):
        
        # initial red boxes positions
        
        self.initial_redbox1_xy = self.sim.data.get_site_xpos('redbox1')[:2]
        self.initial_redbox2_xy = self.sim.data.get_site_xpos('redbox2')[:2]
        self.initial_redbox3_xy = self.sim.data.get_site_xpos('redbox3')[:2]

        goal = self.initial_gripper_xpos[:3]
        goal[2] = 10.01
        
        return goal.copy()

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)


    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        roboutils.reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector into position.
        gripper_target = np.array([0.6, -0., 0.05]) 
        self.sim.data.set_mocap_pos('mocap', gripper_target + self.gripper_extra_height)
        self.sim.data.set_mocap_quat('mocap', self.rot)

        for _ in range(10):     
            pos_ctrl = gripper_target - self.sim.data.get_mocap_pos('mocap')  

            action = np.concatenate([pos_ctrl, self.rot, [0]], -1)
            roboutils.ctrl_set_action(self.sim, action)
            roboutils.mocap_set_action(self.sim, action)
            self.sim.step()

        # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_site_xpos('grip').copy()