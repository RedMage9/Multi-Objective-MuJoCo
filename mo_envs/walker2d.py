import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from os import path


class Walker2dEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, model_path = path.join(path.abspath(path.dirname(__file__)), "assets/walker2d.xml"), frame_skip = 5)
        # mujoco_env.MujocoEnv.__init__(self, "walker2d.xml", 4)
        utils.EzPickle.__init__(self)

    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0

        reward_forward_speed = (posafter - posbefore) / self.dt + alive_bonus

        reward_energy_efficiency = 4 - np.square(a).sum() + alive_bonus

        #self.scalar = 0.3

        #reward = self.scalar * reward_forward_speed + (1 - self.scalar) * reward_energy_efficiency
        #reward = [reward_forward_speed, reward_energy_efficiency]

        info = {
            "reward_obj1": reward_forward_speed,
            "reward_obj2": reward_energy_efficiency,
        }

        #self._reward_value = [reward_forward_speed, reward_energy_efficiency]


        
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        

        done = not (height > 0.8 and height < 2.0 and ang > -1.0 and ang < 1.0)
        ob = self._get_obs()
        return ob, reward, done, info

    # def set_reward_scalar(self, scalar):        
    #     #print('scalar : ' + str(scalar))
    #     self.scalar = scalar * 0.01
    #     #self.scalar = alf.get_config_value(
    #     #        'TrainerConfig.scalar') * 0.01

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel()

    def reset_model(self):
        self.set_state(
            self.init_qpos
            + self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nq),
            self.init_qvel
            + self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nv),
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20
