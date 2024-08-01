import os
from gym import utils
from gym.envs.robotics import fetch_env
import numpy as np

from gym.envs.robotics import utils as robo_util

# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join("fetch", "pick_and_place.xml")


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)

class mo_FetchPickAndPlaceEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type="sparse"):
        initial_qpos = {
            "robot0:slide0": 0.405,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
            "object0:joint": [1.25, 0.53, 0.4, 1.0, 0.0, 0.0, 0.0],
        }
        fetch_env.FetchEnv.__init__(
            self,
            MODEL_XML_PATH,
            has_object=True,
            block_gripper=False,
            n_substeps=20,
            gripper_extra_height=0.2,
            target_in_the_air=True,
            target_offset=0.0,
            obj_range=0.15,
            target_range=0.15,
            distance_threshold=0.05,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
        )
        utils.EzPickle.__init__(self, reward_type=reward_type)

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal) * 5.0

        d2 = goal_distance(achieved_goal, self.sim.data.get_site_xpos("robot0:grip"))

        reward_energy = - 1.0 * np.square(self.action).sum()

        info = {
            "reward_obj1": np.array(- d - d2),
            "reward_obj2": np.array(reward_energy),
        }
        return info

    def _set_action(self, action):
        assert action.shape == (4,)
        action = (
            action.copy()
        )  # ensure that we don't change the action outside of this scope

        self.action = action.copy()

        pos_ctrl, gripper_ctrl = action[:3], action[3]

        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl = [
            1.0,
            0.0,
            1.0,
            0.0,
        ]  # fixed rotation of the end effector, expressed as a quaternion
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply action to simulation.
        robo_util.ctrl_set_action(self.sim, action)
        robo_util.mocap_set_action(self.sim, action)

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        self.sim.forward()
        return True

    def _sample_goal(self):

        goal = self.initial_gripper_xpos[:3] + self.target_offset
        goal[2] = self.height_offset + 0.45

        return goal.copy()