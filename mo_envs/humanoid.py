import numpy as np
from gym.envs.mujoco import mujoco_env
from gym import utils


def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, 1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))[0]


class HumanoidEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        # mujoco_env.MujocoEnv.__init__(self, model_path = path.join(path.abspath(path.dirname(__file__)), "assets/humanoid.xml"), frame_skip = 5)
        mujoco_env.MujocoEnv.__init__(self, "humanoid.xml", 5)
        utils.EzPickle.__init__(self)

    def _get_obs(self):
        data = self.sim.data
        return np.concatenate(
            [
                data.qpos.flat[2:],
                data.qvel.flat,
                data.cinert.flat,
                data.cvel.flat,
                data.qfrc_actuator.flat,
                data.cfrc_ext.flat,
            ]
        )

    def step(self, a):
        pos_before = mass_center(self.model, self.sim)
        self.do_simulation(a, self.frame_skip)
        pos_after = mass_center(self.model, self.sim)

        alive_bonus = 3.0
        data = self.sim.data
        reward_run = 1.25 * (pos_after - pos_before) / self.dt + alive_bonus
        reward_energy = 3.0 - 4.0 * np.square(data.ctrl).sum() + alive_bonus
        #quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
        #quad_impact_cost = 0.5e-6 * np.square(data.cfrc_ext).sum()
        #quad_impact_cost = min(quad_impact_cost, 10)        
        #reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus

        qpos = self.sim.data.qpos
        done = bool((qpos[2] < 1.0) or (qpos[2] > 2.0))

        info = {
            "reward_obj1": np.array(reward_run),
            "reward_obj2": np.array(reward_energy),
        }


        return (
            self._get_obs(),
            0.,
            done,
            info,
        )

    def reset_model(self):
        c = 0.01
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel
            + self.np_random.uniform(
                low=-c,
                high=c,
                size=self.model.nv,
            ),
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.lookat[2] = 2.0
        self.viewer.cam.elevation = -20