import os
import gym
from gym import utils
#from gym.envs.mujoco import mujoco_env
from gym.envs.robotics import fetch_env
import numpy as np
from gym.envs.robotics import robot_env
from gym.envs.robotics import utils as roboutils


# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "assets/moRG6classify.xml")

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)

class moRG6classifyEnv(robot_env.RobotEnv, utils.EzPickle):
    def __init__(self, reward_type='spars'):
        initial_qpos = {
            'joint1': 0, #np.deg2rad(np.random.randint(low=0, high=90)),
            'joint2': 0, #np.deg2rad(np.random.randint(low=35, high=45)),
            'joint3': 1, #np.deg2rad(np.random.randint(low=35, high=45)),
            'joint4': 0,
            'joint5': 1.7, #np.deg2rad(np.random.randint(low=80, high=90)),
            'joint6': 0,
            # 'base_link_joint': 0,
            'finger_joint': np.deg2rad(0),
            'left_inner_finger_joint': 0,
            # 'left_inner_knuckle_joint': 0,
            'right_outer_knuckle_joint': 0,
            'right_inner_finger_joint': 0,
            # 'right_inner_knuckle_joint': 0,
        }

        self.joint_names = ['joint1', 'joint2', 'joint3', 'joint5', 'finger_joint']
        # self.rot = [0., 0., 0., 1.]
        self.rot = [0., 0., 1., 0.]

        self.n_actions = 3

        self.gripper_extra_height = 0.
        self.block_gripper = False
        self.has_object = True
        self.target_in_the_air = True
        self.target_offset = 0.0
        self.obj_range = 0.15
        self.target_range = 0.15
        self.distance_threshold = 0.05
        self.reward_type = reward_type

        robot_env.RobotEnv.__init__(self,
            model_path=MODEL_XML_PATH, n_substeps=20, n_actions=self.n_actions,
            initial_qpos=initial_qpos)
        utils.EzPickle.__init__(self)

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        redbox1_xy = self.sim.data.get_site_xpos('redbox1')[:2]
        purplebox1_xy = self.sim.data.get_site_xpos('purplebox1')[:2]
        
        obj1 = 0.2 - np.linalg.norm(redbox1_xy - np.array([0.6, 0.2]), axis=-1)

        obj2 = 0.2 - np.linalg.norm(purplebox1_xy - np.array([0.6, -0.2]), axis=-1)

        reward = {
            "reward_obj1": obj1,
            "reward_obj2": obj2,
        }

        return reward 

    def _set_action(self, action):
        assert action.shape == (self.n_actions,)
        action = action.copy()  

        gripper = action[2]

        if gripper > 0:
            gripper = 1

        else:
            gripper = -1

        # pos_ctrl = action[:2]
        grip = self.sim.data.get_site_xpos('grip').copy()
        z = (0.04 - grip[2]) * 100
        pos_ctrl = np.concatenate([[action[0]], [action[1]], [z]])  
        pos_ctrl = pos_ctrl * 0.01

        # print(action)

        pos_ctrl = np.concatenate([pos_ctrl, self.rot, [gripper]])

        # Apply action to simulation.
        roboutils.ctrl_set_action(self.sim, pos_ctrl)
        roboutils.mocap_set_action(self.sim, pos_ctrl)

    def _get_obs(self):

        dt = self.sim.nsubsteps * self.sim.model.opt.timestep

        grip_pos = self.sim.data.get_site_xpos('grip').copy()[:2]

        gripper = [self.sim.data.get_joint_qpos('finger_joint')]

        redbox1_xy = self.sim.data.get_site_xpos('redbox1').copy()[:2]

        purplebox1_xy = self.sim.data.get_site_xpos('purplebox1').copy()[:2]
        # tray_xyz = self.sim.data.get_site_xpos('tray')

        achieved_goal = np.squeeze(self.sim.data.get_site_xpos('grip').copy())

        obs = np.concatenate([
            grip_pos, redbox1_xy.ravel(), purplebox1_xy.ravel(), gripper
        ])

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }

    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id('base_link')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 1.75
        self.viewer.cam.azimuth = 180.
        self.viewer.cam.elevation = 0.

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
        gripper_target = np.array([0.6, 0.0, 0.3]) 
        # gripper_target = np.array([0., 0.6, 0.3]) 
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