import os
import gym
from gym import utils
from gym.envs.robotics import fetch_env
import numpy as np
from gym.envs.robotics import robot_env
from gym.envs.robotics import utils as roboutils

# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "assets/moRG6tray.xml")

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)

class moRG6trayEnv(robot_env.RobotEnv, utils.EzPickle):
    def __init__(self, reward_type='spars'):
        initial_qpos = {
            'joint1': -0.004, #np.deg2rad(np.random.randint(low=0, high=90)),
            'joint2': 1.225, #np.deg2rad(np.random.randint(low=35, high=45)),
            'joint3': 1.11, #np.deg2rad(np.random.randint(low=35, high=45)),
            'joint4': -0.01,
            'joint5': -0.765, #np.deg2rad(np.random.randint(low=80, high=90)),
            'joint6': 0.007,
            'base_link_joint': 0,
            'finger_joint': np.deg2rad(0),
            'left_inner_finger_joint': 0,
            # 'left_inner_knuckle_joint': 0,
            'right_outer_knuckle_joint': 0,
            'right_inner_finger_joint': 0,
            # 'right_inner_knuckle_joint': 0,
        }

        self.joint_names = ['joint1', 'joint2', 'joint3', 'joint5', 'finger_joint']
        self.rot = [1., 0., 1., 0.]
        # self.rot = [0., 0., 1., 0.]

        self.n_actions = 2

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

        grip_pos = self.sim.data.get_site_xpos('grip')
        
        ball_delta_y = self.sim.data.get_site_xpos('ball')[1] - grip_pos[1].copy()

        obj1 = (grip_pos[2].copy() - self.prev_grip_z) * 150

        ball_delta_y = (ball_delta_y * ball_delta_y) 
        prev_ball_delta_y = (self.prev_ball_delta_y * self.prev_ball_delta_y) 

        obj2 = 1 - (prev_ball_delta_y - ball_delta_y) 

        reward = {
            "reward_obj1": obj1,
            "reward_obj2": obj2,
        }

        return reward 

    def _set_action(self, action):
        assert action.shape == (self.n_actions,)
        action = action.copy()  

        mocap_y = self.sim.data.get_mocap_pos('mocap')[1]

        grip_pos = self.sim.data.get_site_xpos('grip').copy()
        self.prev_grip_z = grip_pos[2]
        self.prev_ball_delta_y = (self.sim.data.get_site_xpos('ball')[1] - grip_pos[1]) * 10

        # pos_ctrl = action[:2]
        pos_ctrl = np.concatenate([[-0, -grip_pos[1]], [action[0]]])  
        pos_ctrl = pos_ctrl * 0.01

        rot = self.sim.data.get_mocap_quat('mocap') 

        pos_ctrl = np.concatenate([pos_ctrl, self.rot, [-1], [action[1]]])

        # Apply action to simulation.
        roboutils.ctrl_set_action(self.sim, pos_ctrl)
        roboutils.mocap_set_action(self.sim, pos_ctrl)

    def _get_obs(self):

        dt = self.sim.nsubsteps * self.sim.model.opt.timestep

        grip_pos = self.sim.data.get_site_xpos('grip')

        obs1 = [(grip_pos[2].copy() - 0.19) * 6]

        ball_y = self.sim.data.get_site_xpos('ball')[1] 
        obs2 = [ball_y * 10]

        base_link_joint = self.sim.data.get_joint_qpos('base_link_joint') * 3
        # tray_xyz = self.sim.data.get_site_xpos('tray')

        achieved_goal = np.squeeze(grip_pos.copy())

        obs = np.concatenate([
            obs1, obs2, [base_link_joint]
        ])
        
        # print('------------------------------------')
        # jog = self.sim.data.get_joint_qpos('joint1')
        # print(jog)
        # jog = self.sim.data.get_joint_qpos('joint2')
        # print(jog)
        # jog = self.sim.data.get_joint_qpos('joint3')
        # print(jog)
        # jog = self.sim.data.get_joint_qpos('joint4')
        # print(jog)
        # jog = self.sim.data.get_joint_qpos('joint5')
        # print(jog)
        # jog = self.sim.data.get_joint_qpos('joint6')
        # print(jog)
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
        self.viewer.cam.distance = 0.75
        self.viewer.cam.azimuth = 180.
        self.viewer.cam.elevation = -30.

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
        gripper_target = np.array([0.8, 0.0, 0.05]) 
        self.sim.data.set_mocap_pos('mocap', gripper_target + self.gripper_extra_height)
        self.sim.data.set_mocap_quat('mocap', self.rot)

        self.prev_grip_z = self.sim.data.get_site_xpos('grip')[2].copy()
        self.prev_ball_delta_y = (self.sim.data.get_site_xpos('ball')[1] - self.sim.data.get_site_xpos('grip')[1].copy()) * 10

        # for _ in range(10):     
        #     pos_ctrl = gripper_target - self.sim.data.get_mocap_pos('mocap')  

        #     action = np.concatenate([pos_ctrl, self.rot, [0]], -1)
        #     roboutils.ctrl_set_action(self.sim, action)
        #     roboutils.mocap_set_action(self.sim, action)
        #     self.sim.step()

        # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_site_xpos('grip').copy()