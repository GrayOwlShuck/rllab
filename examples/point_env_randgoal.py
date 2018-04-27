import os.path as osp
import tempfile
import scipy.misc
from rllab.envs.base import Env
from rllab.spaces import Box
from rllab.envs.base import Step
from rllab.misc import logger
import numpy as np
import tensorflow as tf
from sandbox.rocky.tf.policies.base import Policy
from rllab.misc.overrides import overrides
from rllab.core.serializable import Serializable
import matplotlib.pyplot as plt

# from pylab import *
import matplotlib.colorbar as cbar
import matplotlib.patches as patches


class PointEnvRandGoal(Env):
    def __init__(self, goal=None, noise=0.1, tolerance=0.1):  # Can set goal to test adaptation.
        self._goal = goal
        self._state = (0, 0)
        self._noise = noise
        self._tolerance = tolerance

    @property  # this should have the same name for all: ant/cheetah direction,... and be a list of params
    def objective_params(self):
        return self._goal

    def set_objective_params(self, goal):
        self._goal = goal

    @property
    def observation_space(self):
        return Box(low=-8, high=8, shape=(2,))

    @property
    def action_space(self):
        return Box(low=-0.1, high=0.1, shape=(2,))

    def reset(self, objective_params=None, clean_reset=False, *reset_args, **reset_kwargs):
        if clean_reset:
            # print("cleaning goal")
            self._goal = None
        if objective_params is not None:
            # print("using given goal")
            self._goal = objective_params
        elif self._goal is None:
            # print("the goal was None so I resample a random")
            # Only set a new goal if this env hasn't had one defined before or if it has been cleaned
            self._goal = np.random.uniform(-5, 5, size=(2,))
        # else:
            # print("there was already a goal! ", self._goal)
        # print("After reset, new goal is: ", self._goal)
        self._state = (0, 0)
        observation = np.copy(self._state)
        return observation

    def step(self, action):
        # print("inside env step: _state, goal, action:", self._state, self._goal, self. action)
        self._state = np.clip(self._state + action + self._noise * np.random.randn(*np.shape(self._state)),
                              *self.observation_space.bounds)
        x, y = self._state
        x -= self._goal[0]
        y -= self._goal[1]
        reward = - (x ** 2 + y ** 2) ** 0.5
        done = abs(x) < 0.1 and abs(y) < 0.1
        next_observation = np.copy(self._state)
        return Step(observation=next_observation, reward=reward, done=done, goal=self._goal, goal_reached=done)  # goal goes to env_infos

    def render(self):
        print('current state:', self._state)

    def log_diagnostics_2(self, postupdate_paths, demos=None, preupdate_paths=None, all_updates_paths=None, prefix='',
                        report=None, policy=None, goals=None):
        """
        :param postupdate_paths: list of dicts, one per gradient update. The keys of the dict are the env numb and the val the UNprocessed samples
        :param demos: same
        """
        logger.log("Saving visualization of paths")
        if type(postupdate_paths) == dict:
            postupdate_paths = postupdate_paths
            preupdate_paths = preupdate_paths
        elif (type(postupdate_paths) == list and len(postupdate_paths) > 1 and type(postupdate_paths[0]) == dict):
            preupdate_paths = postupdate_paths[0]  # paths obtained before any update
            postupdate_paths = postupdate_paths[-1]  # paths obtained after all updates
        else:
            raise ValueError("log_diagnostics only supports paths from vectorized sampler!")

        num_envs = len(postupdate_paths)
        avg_postupdate_rewards = [None] * num_envs
        avg_postupdate_success = [None] * num_envs
        avg_preupdate_rewards = [None] * num_envs
        avg_preupdate_success = [None] * num_envs
        avg_demo_rewards = [None] * num_envs
        avg_demo_success = [None] * num_envs
        for ind in range(num_envs):
            avg_postupdate_rewards[ind] = np.mean([np.sum(p['rewards']) for p in postupdate_paths[ind]])
            avg_postupdate_success[ind] = np.mean([np.any(p['env_infos']['goal_reached']) for p in postupdate_paths[ind]])
            if demos is not None:
                avg_demo_rewards[ind] = np.mean([np.sum(p['rewards']) for p in demos[ind]])
                avg_demo_success[ind] = np.mean([np.any(p['env_infos']['goal_reached']) for p in demos[ind]])
            if preupdate_paths is not None:
                avg_preupdate_rewards[ind] = np.mean([np.sum(p['rewards']) for p in preupdate_paths[ind]])
                avg_preupdate_success[ind] = np.mean([np.any(p['env_infos']['goal_reached']) for p in preupdate_paths[ind]])

        for ind in range(min(5, num_envs)):
            plt.clf()
            plt.plot(*postupdate_paths[ind][0]['env_infos']['goal'][0], 'r*',
                     markersize=10)
            post_points = postupdate_paths[ind][0]['observations']
            plt.plot(post_points[:, 0], post_points[:, 1], '-b', linewidth=1)

            # pre_points = preupdate_paths[ind][1]['observations']
            # post_points = postupdate_paths[ind][1]['observations']
            # plt.plot(pre_points[:, 0], pre_points[:, 1], '--r', linewidth=2)
            # plt.plot(post_points[:, 0], post_points[:, 1], '--b', linewidth=1)
            #

            if demos is not None:
                if type(demos) == dict:
                    demo_paths = demos
                else:  # todo: in what case is demos a list? what are the other elements?
                    demo_paths = demos[0]
                demo_points = demo_paths[ind][0]['observations']
                plt.plot(demo_points[:, 0], demo_points[:, 1], '-.g', linewidth=2)

            if preupdate_paths is not None:
                pre_points = preupdate_paths[ind][0]['observations']
                plt.plot(pre_points[:, 0], pre_points[:, 1], '-m', linewidth=1)
                plt.legend(['goal', 'post-update path', 'demos', 'pre-update path'])
            else:
                plt.legend(['goal', 'post-update path', 'demos'])

            plt.plot(0, 0, 'b.', markersize=5)
            plt.xlim([-8, 8])
            plt.ylim([-8, 8])

            log_dir = logger.get_snapshot_dir()
            if report is None:
                plt.savefig(osp.join(log_dir, 'post_update_plot' + str(ind) + '.png'))
            else:
                vec_img = save_image()
                summary_text = 'Env_{}:\nAvg Postupdate Rew: {:.2f}, Avg Post Success:{:.2f}\n'.format(ind,
                                                                                      avg_postupdate_rewards[ind],
                                                                                      avg_postupdate_success[ind])
                if avg_demo_rewards[ind] is not None:
                    summary_text += 'Avg Demo Rew: {:.2f}, Avg Demo Success:{:.2f}\n'.format(avg_demo_rewards[ind],
                                                                                     avg_demo_success[ind])
                if avg_preupdate_rewards[ind] is not None:
                    summary_text += 'Avg Preupdate Rew: {:.2f}, Avg Preupdate Success:{:.2f}\n'.format(avg_preupdate_rewards[ind],
                                                                                               avg_preupdate_success[ind])
                report.add_image(vec_img, summary_text)

        report.new_row()
        if policy is not None:
            plot_policy_means(policy, self, self.observation_space.bounds, report, goals=goals)

        summary_text = 'Avg Postupdate Rew: {:.2f}, Avg Post Success:{:.2f}\n'.format(np.mean(avg_postupdate_rewards),
                                                                              np.mean(avg_postupdate_success))
        logger.record_tabular('postUpdate_AverageSuccess', np.mean(avg_postupdate_success))
        if avg_demo_rewards[0] is not None:
            summary_text += 'Avg Demo Rew: {:.2f}, Avg Demo Success:{:.2f}\n'.format(np.mean(avg_demo_rewards),
                                                                             np.mean(avg_demo_success))
            logger.record_tabular('demo_AverageSuccess', np.mean(avg_demo_success))
        if avg_preupdate_rewards[0] is not None:
            summary_text += 'Avg Preupdate Rew: {:.2f}, Avg Preupdate Success:{:.2f}\n'.format(np.mean(avg_preupdate_rewards),
                                                                                       np.mean(avg_preupdate_success))
            logger.record_tabular('preUpdate_AverageSuccess', np.mean(avg_preupdate_success))
        report.add_text(summary_text)
        report.save()


class StraightDemo(Policy, Serializable):
    def __init__(self, *args, **kwargs):
        Serializable.quick_init(self, locals())
        super(StraightDemo, self).__init__(*args, **kwargs)

    @overrides
    def get_action(self, observation, objective_params, *args,
                   **kwargs):  # the same policy obj is applied to all envs!! so it needs to take in the goal!
        goal_vec = np.array(objective_params) - np.array(observation)
        return np.clip(goal_vec, *self.action_space.bounds), dict()

    @overrides
    def get_actions(self, observations, objective_params, *args, **kwargs):
        """
        needed for the vec env
        :param objective_params: this is an arg for the reset of the env! It specifies the task
        :param args/kwargs: these are just to throw away all other reset args for the env that don't define the task
        """
        actions = []
        for obs, goal in zip(observations, objective_params):
            goal_vec = np.array(goal) - np.array(obs)
            max_coef = max(np.max(goal_vec / self.action_space.bounds), 1)  # done to scale the action, not clip it
            actions.append(goal_vec / max_coef)  # so that direction is maintained.
        return actions, dict()

    # def get_params(self):
    #     return tf.Variable('dummy', [0])

    def get_param_values(self):
        return None  # tf.Variable('dummy', [0])

    def set_param_values(self, flattened_params, **tags):
        pass


def plot_policy_means(policy, env, bounds=None, report=None, num_samples=10, goals=None, idxs=np.arange(5)):
    # use idxs=None if not vectorized, and a list if specifying what envs to plot
    if len(idxs) > len(goals):
        idxs = np.arange(len(goals))
    if bounds is None:
        bounds = env.observation_space.bounds
    x = np.linspace(bounds[0][0], bounds[1][0], num_samples)
    y = np.linspace(bounds[0][1], bounds[1][1], num_samples)
    states = np.stack(np.meshgrid(x, y), axis=-1).reshape(-1, 2)
    # observations = [np.concatenate([state, [0, ] * (env.observation_space.flat_dim - len(state) - len(goal)), goal]) for
    #                 state in states]  # in case we need to append goal for the policy
    means = []
    log_stds = []
    # import pdb; pdb.set_trace()
    for state in states:
        actions, agent_info = policy.get_action([state] * len(goals), idx=idxs)
        means.append(agent_info['mean'])
        log_stds.append(agent_info['log_std'])
    for idx in idxs:
        vecs = np.array([mean[idx] for mean in means])
        vars = np.array([np.exp(log_std[idx]) * 0.25 for log_std in log_stds])
        ells = [patches.Ellipse(state, width=vars[i][0], height=vars[i][1], angle=0) for i, state in enumerate(states)]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        for e in ells:
            ax.add_artist(e)
            e.set_alpha(0.2)
        # plt.scatter(*np.array(goals)[idx], color='r', s=100)
        plt.plot(*np.array(goals)[idx], 'r*', markersize=10)
        plt.plot(0, 0, 'b.', markersize=5)
        Q = plt.quiver(states[:, 0], states[:, 1], vecs[:, 0], vecs[:, 1], units='xy', angles='xy', scale_units='xy',
                       scale=1)  # , np.linalg.norm(vars * 4)
        qk = plt.quiverkey(Q, 0.8, 0.85, 1, r'1 Nkg', labelpos='E', coordinates='figure')
        # cb = plt.colorbar(Q)
        vec_img = save_image()
        if report is not None:
            report.add_image(vec_img, 'policy means Env_' + str(idx))


def save_image(fig=None, fname=None):
    if fname is None:
        fname = tempfile.TemporaryFile()
    if fig is not None:
        fig.savefig(fname)
    else:
        plt.savefig(fname, format='png')
    plt.close('all')
    fname.seek(0)
    img = scipy.misc.imread(fname)
    fname.close()
    return img
