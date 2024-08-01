import os
import numpy as np

from gym import utils
from gym.envs.robotics import robot_env
from gym.envs.robotics import hand_env
# import DSenv.ds_util as ds_util
import pickle
from gym.envs.robotics import utils as roboutils
# from gym.envs.robotics.utils import robot_get_obs

DEFAULT_INITIAL_QPOS = {
    'joint1': 0., #np.deg2rad(np.random.randint(low=0, high=90)),
    'joint2': 0.48, #np.deg2rad(np.random.randint(low=35, high=45)),
    'joint3': 2.12, #np.deg2rad(np.random.randint(low=35, high=45)),
    'joint4': 0.,
    'joint5': -1.04, #np.deg2rad(np.random.randint(low=80, high=90)),
    # 'joint6': 0.002,
    'joint_0.0': 0,
    'joint_1.0': 0.,
    'joint_2.0': 0.,
    'joint_3.0': 0.,
    'joint_4.0': 0,
    'joint_5.0': 0.,
    'joint_6.0': 0.,
    'joint_7.0': 0.,
    'joint_8.0': 0,
    'joint_9.0': 0.,
    'joint_10.0': 0.,
    'joint_11.0': 0.,
    'joint_12.0': 0,
    'joint_13.0': 0,
    'joint_14.0': 0,
    'joint_15.0': 0,
}


# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "assets/mo_alle_drone.xml")
# MODEL_XML_PATH = os.path.join('hand', 'reach.xml')

def robot_get_obs(sim):
    """Returns all joint positions and velocities associated with
    a robot.
    """
    if sim.data.qpos is not None and sim.model.joint_names:
        names = [n for n in sim.model.joint_names if n.startswith('joint')]
        return (
            np.array([sim.data.get_joint_qpos(name) for name in names]),
            np.array([sim.data.get_joint_qvel(name) for name in names]),
        )
    return np.zeros(0), np.zeros(0)

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class moAlleDroneEnv(robot_env.RobotEnv, utils.EzPickle):
    def __init__(
        self, distance_threshold=0.01, n_substeps=20, relative_control=False,
        initial_qpos=DEFAULT_INITIAL_QPOS, reward_type='spars',
    ):
        self.rot = [1., 0., 1., 0.]
        self.n_actions = 23
        self.hand_start_pos = np.array([0.5, 0., 0.2])

        self.temp = 0

        utils.EzPickle.__init__(**locals())
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type

        robot_env.RobotEnv.__init__(
            self, MODEL_XML_PATH, n_substeps=n_substeps, n_actions=self.n_actions, initial_qpos=initial_qpos)


    def _get_achieved_goal(self):
        goal = [0 for i in range(3)]
        return np.array(goal).flatten()

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info):
        
        # grip_pos = self.sim.data.get_site_xpos('grip')
        
        # ball_delta_y = self.sim.data.get_site_xpos('ball')[1] - grip_pos[1].copy()

        # obj1 = (grip_pos[2].copy() - self.prev_grip_z) * 150

        mocap_pos = self.sim.data.get_mocap_pos('mocap').copy()
        mocap_pos[2] = mocap_pos[2] - 0.1

        target_pos = self.sim.data.get_site_xpos('target').copy()

        # geom_id = self.sim.model.geom_name2id('G9_0')
        # string_pos = self.sim.data.geom_xpos[geom_id].copy()

        # print(mocap_pos, target_pos, self.sim.data.geom_xpos[geom_id])

        robot_qpos, robot_qvel = robot_get_obs(self.sim)

        target_to_start = -np.linalg.norm(target_pos - self.hand_start_pos, axis=-1)
        target_to_mocap = -np.linalg.norm(target_pos - mocap_pos, axis=-1)

        ftip1 = self.sim.data.get_site_xpos('ftip1')
        ftip2 = self.sim.data.get_site_xpos('ftip2')
        ftip3 = self.sim.data.get_site_xpos('ftip3')
        ftip4 = self.sim.data.get_site_xpos('ftip4')

        finger_grip = np.linalg.norm(ftip4 - ftip1, axis=-1) + np.linalg.norm(ftip4 - ftip2, axis=-1) + np.linalg.norm(ftip4 - ftip3, axis=-1)

        # mocap_to_string = -np.linalg.norm(string_pos[:2] - mocap_pos[:2], axis=-1)
        # print(target_to_start + target_to_mocap, mocap_to_string * 20)

        obj1 = target_to_start + target_to_mocap - finger_grip * 0.1
        # obj2 = target_to_start + target_to_mocap + mocap_to_string * 20
        obj2 = target_to_start + target_to_mocap - np.linalg.norm(robot_qpos[5:].copy(), axis=-1) * 0.1

        reward = {
            "reward_obj1": obj1,
            "reward_obj2": obj2,
        }

        return reward 

    # RobotEnv methods
    # ----------------------------

    def _set_action(self, action):
        assert action.shape == (self.n_actions,)

        action = action.copy()  

        mocap_pos = action[0:3]
        mocap_pos[0:2] = [0,0]
        mocap_pos = mocap_pos * 0.01


        self.prev_mocap_quat = action[3:7]
        # self.prev_mocap_quat = [-1 for i in range(4)]
        mocap_quat = action[3:7] * 0.01

        # body_id = self.sim.model.body_name2id('link6')
        # lookat = self.sim.data.body_xpos[body_id]
        # mocap_pos = self.hand_start_pos - lookat
        mocap_quat = [1, 0, 1, 0]

        mocap_ctrl = np.concatenate([mocap_pos, mocap_quat])

        # mocap_ctrl = [-0.0 for i in range(7)]

        self.prev_ctrl_action = action[7:23]
        # self.prev_ctrl_action = [0 for i in range(16)]
        ctrl_action = action[7:23] 
        # ctrl_action = [-0 for i in range(16)]
                
        # ctrl_action[0] = 0. 
        # ctrl_action[4] = 0. 
        # ctrl_action[8] = 0. 

        # ctrl_action[12] = 1. 
        # ctrl_action[13] = 1. 
        # ctrl_action[14] = 1. 
        # ctrl_action[15] = 1. 

        # rot = self.sim.data.get_mocap_quat('mocap') 
        # print(rot)

        # ctrlrange = self.sim.model.actuator_ctrlrange
        self.sim.data.ctrl[:] = ctrl_action 
        # self.sim.data.ctrl[:] = np.clip(self.sim.data.ctrl, ctrlrange[:, 0], ctrlrange[:, 1])

        # Apply action to simulation.
        # roboutils.ctrl_set_action(self.sim, pos_ctrl)
        roboutils.mocap_set_action(self.sim, mocap_ctrl)

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)

        roboutils.reset_mocap_welds(self.sim)

        # Move end effector into position.
        gripper_target = np.array([0.5, 0.0, 0.4]) 
        self.sim.data.set_mocap_pos('mocap', gripper_target)
        self.sim.data.set_mocap_quat('mocap', self.rot)


        # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_site_xpos('palm').copy()
        
        self.initial_goal = self._get_achieved_goal().copy()
        # self.palm_xpos = self.sim.data.body_xpos[self.sim.model.body_name2id('robot0:palm')].copy()
        self.palm_xpos = np.array([0.0, 0.0, 0.0])

        self.sim.forward()

    def _get_obs(self):

        dt = self.sim.nsubsteps * self.sim.model.opt.timestep

        # palm_pos = self.sim.data.get_site_xpos('palm')

        mocap_pos = self.sim.data.get_mocap_pos('mocap').copy()
        rot = self.sim.data.get_mocap_quat('mocap').copy()

        robot_qpos, robot_qvel = robot_get_obs(self.sim)

        achieved_goal = self._get_achieved_goal().ravel()

        target_pos = self.sim.data.get_site_xpos('target').copy()

        # geom_id = self.sim.model.geom_name2id('G9_0')
        # string_pos = self.sim.data.geom_xpos[geom_id].copy()

        # observation = np.concatenate([mocap_pos, rot, robot_qpos[5:].copy(), target_pos, string_pos[:2]])
        observation = np.concatenate([mocap_pos, rot, robot_qpos[5:].copy(), target_pos])

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
            'observation': observation.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }

    def _sample_goal(self):

        goal = self.initial_goal.copy().reshape(-1, 3)

        self.temp = 0

        return goal.flatten()

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id('target0')
        self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]
        self.sim.forward()

    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id('alle_base_body')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 1
        self.viewer.cam.azimuth = -90.
        self.viewer.cam.elevation = 0.
