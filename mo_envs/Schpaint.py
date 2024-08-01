import os
import gym
from gym import utils
#from gym.envs.mujoco import mujoco_env
from gym.envs.robotics import fetch_env
# from DSenv.gripper.DS_gripper_env import DSGripperEnv
import numpy as np
# import DSenv.ds_util as ds_util
from gym.envs.robotics import rotations, robot_env
from gym.envs.robotics import utils as roboutils


# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "assets/Schpaint.xml")

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)

class SchpaintEnv(robot_env.RobotEnv, utils.EzPickle):
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
        self.rot = [0., 0., 1., 0.]

        self.n_actions = 2        

        self.grid = np.zeros((8, 8))

        self.gripper_extra_height = 0.2
        self.block_gripper = False
        self.has_object = True
        self.target_in_the_air = True
        self.target_offset = 0.0
        self.obj_range = 0.15
        self.target_range = 0.15
        self.distance_threshold = 0.05
        self.reward_type = reward_type

        self.gripper_color = 1
        self._cur_step = 0
        self._rednum = np.random.randint(51)
        
        super(SchpaintEnv, self).__init__(
            model_path=MODEL_XML_PATH, n_substeps=20, n_actions=self.n_actions,
            initial_qpos=initial_qpos)

        utils.EzPickle.__init__(self)

    def compute_reward(self, achieved_goal, goal, info):

        prev_grid = self.grid.copy()

        # update painted grid
        # 1 : red / 0 : white / -1 : blue
        # 0.2 * 0.2 / 8 * 8
        # 0.025 squre

        xy = self.sim.data.get_site_xpos('grip')[:2]

        rel_xy = xy - np.array((0.4,0))

        grid_xy = rel_xy + np.array((0.2,0.2))
        grid_xy = grid_xy * 20

        grid_x = int(grid_xy[0])
        grid_y = int(grid_xy[1])
                     
        # paint grid
        
        prev_color = 0

        if grid_x >= 0 and grid_x <= 7 and grid_y >= 0 and grid_y <= 7:
            prev_color = self.grid[grid_x][grid_y].copy()

            if self.gripper_color == 1:
                self.grid[grid_x][grid_y] = 1
            if self.gripper_color == -1:
                self.grid[grid_x][grid_y] = -1

        # identify target grid color
        # obj1 : correctly painted red grid 
        # obj2 : correctly painted blue grid 

        coord_x = grid_x + 0.5
        coord_y = grid_y + 0.5

        is_grid_which_color = 0

        if np.sum(np.square([coord_x - 4, coord_y - 4])) < 16:
            if coord_y > 4 and np.sum(np.square([coord_x - 6, coord_y - 4])) > 4:
                is_grid_which_color = 1
            
            if np.sum(np.square([coord_x - 2, coord_y - 4])) < 4:
                is_grid_which_color = 1

        if np.sum(np.square([coord_x - 4, coord_y - 4])) < 16:
            if coord_y < 4 and np.sum(np.square([coord_x - 2, coord_y - 4])) > 4:
                is_grid_which_color = -1
            
            if np.sum(np.square([coord_x - 6, coord_y - 4])) < 4:
                is_grid_which_color = -1

        obj1 = 0
        obj2 = 0

        if self.gripper_color == 1:
            if not prev_color == 1:
                if is_grid_which_color == 1:
                    obj1 = 1
                if is_grid_which_color == -1:
                    obj2 = -1

        if self.gripper_color == -1:
            if not prev_color == -1:
                if is_grid_which_color == 1:
                    obj1 = -1
                if is_grid_which_color == -1:
                    obj2 = 1

        if is_grid_which_color == 0:
            obj1 = -1
            obj2 = -1

        reward = {
            "reward_obj1": obj1,
            "reward_obj2": obj2,
        }

        return reward 

    def _set_action(self, action):
        assert action.shape == (self.n_actions,)
        action = action.copy()  
        pos_ctrl = np.concatenate([action[:2], [0]])    
        pos_ctrl = pos_ctrl * 0.05 
    
        # gripper_ctrl = action[2]
        # if gripper_ctrl > 0:
        #     self.gripper_color = 1
        # else:
        #     self.gripper_color = -1

        self._cur_step = self._cur_step + 1

        if self._cur_step > self._rednum:
            self.gripper_color = -1
        

        # gripper_ctrl = np.array([gripper_ctrl[0], -gripper_ctrl[0]])
        gripper_ctrl = np.array([-1, 1])

        action = np.concatenate([pos_ctrl, self.rot, gripper_ctrl])

        # Apply action to simulation.
        roboutils.ctrl_set_action(self.sim, action)
        roboutils.mocap_set_action(self.sim, action)

    def _get_obs(self):

        grip_pos = self.sim.data.get_site_xpos('grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        # gripper state
        # gripper_state = [self.sim.data.get_joint_qpos('right_finger'), self.sim.data.get_joint_qpos('left_finger')]

        # achieved_goal : coordinates of grip
        achieved_goal = np.squeeze(grip_pos.copy())

        obs = np.concatenate([
            grip_pos[:2], [self.gripper_color], self.grid.copy().ravel()
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

        self.grid = np.zeros((8, 8))
        
        self.sim.forward()
        return True

    def _sample_goal(self):

        self._rednum = np.random.randint(51)
        self._cur_step = 0
        self.gripper_color = 1

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
        gripper_target = np.array([0.4, -0., 0.04]) 
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