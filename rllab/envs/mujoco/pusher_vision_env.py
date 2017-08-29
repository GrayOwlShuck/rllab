import numpy as np

from rllab.core.serializable import Serializable
from rllab.envs.base import Step
from rllab.envs.mujoco.mujoco_env import MujocoEnv
from rllab.misc import logger
from rllab.misc.overrides import overrides
from PIL import Image


def smooth_abs(x, param):
    return np.sqrt(np.square(x) + np.square(param)) - param


class PusherVisionEnv(MujocoEnv, Serializable):

    FILE = None #'pusher.xml'

    def __init__(self, *args, **kwargs):
        self.frame_skip = 5
        self.__class__.FILE = kwargs['xml_file']
        if 'distractors' in kwargs:
            self.include_distractors = kwargs['distractors']
        else:
            self.include_distractors = False
        super(PusherVisionEnv, self).__init__(*args, **kwargs)
        self.frame_skip = 5
        Serializable.__init__(self, *args, **kwargs)
        self.adjust_viewer()

    def adjust_viewer(self):
        viewer = self.get_viewer()
        viewer.autoscale()
        viewer.cam.trackbodyid = -1
        viewer.cam.lookat[0] = 0.3 # more positive moves the dot left
        viewer.cam.lookat[1] = -0.3 # more positive moves the dot down
        viewer.cam.lookat[2] = 0.0
        viewer.cam.distance = 1.3
        viewer.cam.elevation = -90
        viewer.cam.azimuth = 90

    def get_current_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat[:7],
            self.model.data.qvel.flat[:7],
            self.get_body_com("tips_arm"),
            self.get_body_com("distractor"),
            self.get_body_com("object"),
            self.get_body_com("goal"),
        ])

    def get_current_image_obs(self):
        image = self.viewer.get_image()
        pil_image = Image.frombytes('RGB', (image[1], image[2]), image[0])
        pil_image = pil_image.resize((125,125), Image.ANTIALIAS)
        image = np.flipud(np.array(pil_image))
        return image, np.concatenate([
            self.model.data.qpos.flat[:7],
            self.model.data.qvel.flat[:7],
            self.get_body_com('tips_arm'),
            self.get_body_com('goal'),
            ])

    #def get_body_xmat(self, body_name):
    #    idx = self.model.body_names.index(body_name)
    #    return self.model.data.xmat[idx].reshape((3, 3))

    def get_body_com(self, body_name):
        idx = self.model.body_names.index(body_name)
        return self.model.data.com_subtree[idx]

    def step(self, action):
        self.frame_skip = 5
        vec_1 = self.get_body_com("object") - self.get_body_com("tips_arm")
        vec_2 = self.get_body_com("object") - self.get_body_com("goal")
        reward_near = - np.linalg.norm(vec_1)
        reward_dist = - np.linalg.norm(vec_2)
        reward_ctrl = - np.square(action).sum()
        reward = reward_dist + 0.1 * reward_ctrl + 0.5 * reward_near

        self.forward_dynamics(action)
        next_obs = self.get_current_obs()

        done = False
        return Step(next_obs, reward, done)

    @overrides
    def reset(self, init_state=None, init_arm_only=False, left=None):
        self.frame_skip = 5
        qpos = self.init_qpos.copy()
        if left is not None:
            if left:
                ylow = -0.2
                yhigh = 0
            else:
                ylow = 0
                yhigh = 0.2
        else:
            ylow = -0.2
            yhigh = 0.2

        if not init_arm_only:
            self.goal_pos = np.asarray([0, 0])

            while True:
                self.obj_pos = np.concatenate([
                        np.random.uniform(low=-0.3, high=0, size=1),
                        np.random.uniform(low=ylow, high=yhigh, size=1)])
                if np.linalg.norm(self.obj_pos - self.goal_pos) > 0.17:
                    break

            if self.include_distractors:
                if self.obj_pos[1] < 0:
                    y_range = [0.0, 0.2]
                else:
                    y_range = [-0.2, 0.0]
                while True:
                    self.distractor_pos = np.concatenate([
                            np.random.uniform(low=-0.3, high=0, size=1),
                            np.random.uniform(low=y_range[0], high=y_range[1], size=1)])
                    if np.linalg.norm(self.distractor_pos - self.goal_pos) > 0.17 and np.linalg.norm(self.obj_pos - self.distractor_pos) > 0.1:
                        break
                qpos[-6:-4,0] = self.distractor_pos

            qpos[-4:-2,0] = self.obj_pos
            qpos[-2:,0] = self.goal_pos
        else:
            qpos[-6:,0] = self.model.data.qpos.copy()[-6:,0]
        #qvel = self.init_qvel + np.random.uniform(low=-0.005,
        #            high=0.005, size=(1,self.model.nv))
        qvel = self.init_qvel + np.random.uniform(low=-0.005,
                    high=0.005, size=(self.model.nv,1))
        setattr(self.model.data, 'qpos', qpos)
        setattr(self.model.data, 'qvel', qvel)
        self.model.data.qvel = qvel
        self.model._compute_subtree()
        self.model.forward()

        #self.reset_mujoco(init_state)
        #self.model.forward()
        self.current_com = self.model.data.com_subtree[0]
        self.dcom = np.zeros_like(self.current_com)
        return self.get_current_obs()

    @overrides
    def log_diagnostics(self, paths):
        pass
