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
# from rllab.core.serializable import Serializable
import matplotlib.pyplot as plt

# from pylab import *
import matplotlib.colorbar as cbar
import matplotlib.patches as patches

class PointEnvRandGoal(Env):  #, Serializable):
    def __init__(self, goal=None, noise=0.1, tolerance=0.1, fix_goal=False,
                 obs_append_objective=False, sparse_reward=False):

        # Serializable.__init__(self, locals())  # todo: is this safe?
        # self.quick_init(locals())  # todo: is this safe?
        # Serializable.quick_init(self, locals())
        self._goal = goal
        self._state = (0, 0)
        self._noise = noise
        self._tolerance = tolerance
        self._fix_goal = fix_goal
        self._obs_append_objective = obs_append_objective
        self._sparse_reward = sparse_reward

    @property  # this should have the same name for all: ant/cheetah direction,... and be a list of params
    def objective_params(self):
        return self._goal

    @property
    def test_objective_params(self):
        return [(-6, -6), (-6, 6), (6, -6), (6, 6), (0, 6),  (6, 0), (-6, 0), (0, -6)]

    def set_objective_params(self, goal):
        self._goal = goal

    @property
    def observation_space(self):
        obs_dim = 4 if self._obs_append_objective else 2
        return Box(low=-8, high=8, shape=(obs_dim,))

    @property
    def action_space(self):
        return Box(low=-0.1, high=0.1, shape=(2,))

    def get_current_obs(self):
        observation = np.copy(self._state)
        if self._obs_append_objective:
            observation = np.concatenate([observation, self.objective_params])
        return observation

    def reset(self, objective_params=None, clean_reset=False, *reset_args, **reset_kwargs):
        if clean_reset and not self._fix_goal:
            # print("cleaning goal!")
            self._goal = None
        if objective_params is not None and not self._fix_goal:
            # print("using given goal: ", objective_params)
            self._goal = objective_params
        elif self._goal is None:  # I will allow to rest if there is no goal
            # print("the goal was None so I resample a random")
            # Only set a new goal if this env hasn't had one defined before or if it has been cleaned
            self._goal = np.random.uniform(-5, 5, size=(2,))
        # else:
            # print("there was already a goal! ", self._goal)
        # print("After reset, new goal is: ", self._goal)
        self._state = (0, 0)
        observation = self.get_current_obs()
        return observation

    def step(self, action):
        # print("inside env step: _state, goal, action:", self._state, self._goal, self. action)
        self._state = np.clip(self._state + action + self._noise * np.random.randn(*np.shape(self._state)),
                              *(b[:2] for b in self.observation_space.bounds))  # remove the goal bounds
        x, y = self._state
        x -= self._goal[0]
        y -= self._goal[1]
        dist = (x ** 2 + y ** 2) ** 0.5
        reward = int(dist < self._tolerance) if self._sparse_reward else - dist
        done =  dist < self._tolerance
        next_observation = self.get_current_obs()
        return Step(observation=next_observation, reward=reward, done=done, goal=self._goal, goal_reached=done)  # goal goes to env_infos

    def render(self):
        print('current state:', self._state)

    def log_diagnostics(self, paths, prefix='', plot=False, *args, **kwargs):
        if type(paths) is dict:
            avg_success = np.mean([np.any(roll['env_infos']['goal_reached']) for p in paths.values() for roll in p])
            logger.record_tabular('NumPathsLog', len([roll for p in paths.values() for roll in p]))
        else:
            avg_success = np.mean([np.any(p['env_infos']['goal_reached']) for p in paths])
            logger.record_tabular('NumPathsLog', len(paths))
        logger.record_tabular('AverageSuccess', avg_success)
        if plot:
            self.log_diagnostics_2(postupdate_paths=paths, *args, **kwargs)

    def log_diagnostics_2(self, postupdate_paths, demos=None, preupdate_paths=None, all_updates_paths=None, prefix='',
                        report=None, policy=None, goals=None):
        """
        :param postupdate_paths: list of dicts, one per gradient update. The keys of the dict are the env numb and the val the UNprocessed samples
        :param demos: same
        """
        logger.log("Saving visualization of paths")
        if type(postupdate_paths) == dict:  # a dic with integer keys (the envs). We have to give unprocessed paths (ie, non-concat)
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
        avg_postupdate_len = [None] * num_envs
        avg_preupdate_rewards = [None] * num_envs
        avg_preupdate_success = [None] * num_envs
        avg_preupdate_len = [None] * num_envs
        avg_improvement_rewards = [None] * num_envs
        avg_improvement_success = [None] * num_envs
        avg_demo_rewards = [None] * num_envs
        avg_demo_success = [None] * num_envs
        avg_demo_len = [None] * num_envs
        for ind in range(num_envs):
            avg_postupdate_rewards[ind] = np.mean([np.sum(p['rewards']) for p in postupdate_paths[ind]])
            avg_postupdate_success[ind] = np.mean([np.any(p['env_infos']['goal_reached']) for p in postupdate_paths[ind]])
            avg_postupdate_len[ind] = np.mean([len(p['rewards']) for p in postupdate_paths[ind]])
            if demos is not None:
                avg_demo_rewards[ind] = np.mean([np.sum(p['rewards']) for p in demos[ind]])
                avg_demo_success[ind] = np.mean([np.any(p['env_infos']['goal_reached']) for p in demos[ind]])
                avg_demo_len[ind] = np.mean([len(p['rewards']) for p in demos[ind]])
            if preupdate_paths is not None:
                avg_preupdate_rewards[ind] = np.mean([np.sum(p['rewards']) for p in preupdate_paths[ind]])
                avg_preupdate_success[ind] = np.mean([np.any(p['env_infos']['goal_reached']) for p in preupdate_paths[ind]])
                avg_preupdate_len[ind] = np.mean([len(p['rewards']) for p in preupdate_paths[ind]])
                avg_improvement_rewards[ind] = avg_postupdate_rewards[ind] - avg_preupdate_rewards[ind]
                avg_improvement_success[ind] = avg_postupdate_success[ind] - avg_preupdate_success[ind]

        # logging for viskit plotting
        logger.record_tabular('postUpdate_AverageSuccess', np.mean(avg_postupdate_success))
        logger.record_tabular('postUpdate_AverageReward', np.mean(avg_postupdate_rewards))
        if preupdate_paths is not None:
            logger.record_tabular('preUpdate_AverageSuccess', np.mean(avg_preupdate_success))
            logger.record_tabular('preUpdate_AverageReward', np.mean(avg_preupdate_rewards))
            logger.record_tabular('Improvement_AverageSuccess', np.mean(avg_improvement_success))
            logger.record_tabular('Improvement_AverageReward', np.mean(avg_improvement_rewards))
        if demos is not None:
            logger.record_tabular('demo_AverageSuccess', np.mean(avg_demo_success))

        for ind in range(min(5, num_envs)):
            summary_text = ''
            if avg_preupdate_rewards[0] is not None:
                summary_text += 'AvgPreRew: {:.2f}, AvgPreSucc: {:.2f}, AvgLen: {:.2f}\n'.format(np.mean(avg_preupdate_rewards),
                                                                                                   np.mean(avg_preupdate_success),
                                                                                                  np.mean(avg_preupdate_len))
            summary_text += 'AvgPostRew: {:.2f}, AvgPostSucc: {:.2f}, AvgLen: {:.2f}\n'.format(np.mean(avg_postupdate_rewards),
                                                                                          np.mean(avg_postupdate_success),
                                                                                           np.mean(avg_postupdate_len))
            if avg_demo_rewards[0] is not None:
                summary_text += 'AvgDemRew: {:.2f}, AvgDemSucc: {:.2f}, AvgLen: {:.2f}\n'.format(np.mean(avg_demo_rewards),
                                                                                         np.mean(avg_demo_success),
                                                                                         np.mean(avg_demo_len))
            goal = postupdate_paths[ind][0]['env_infos']['goal'][0]
            post_obs = postupdate_paths[ind][0]['observations']
            demo_obs = None
            pre_obs = None
            if demos is not None:
                if type(demos) == dict:
                    demo_paths = demos
                else:  # todo: in what case is demos a list? what are the other elements?
                    demo_paths = demos[0]
                demo_obs = demo_paths[ind][0]['observations']
            if preupdate_paths is not None:
                pre_obs = preupdate_paths[ind][0]['observations']

            self.plot_paths(preupdate_obs=pre_obs, postupdate_obs=post_obs, demo_obs=demo_obs, goal=goal,
                            summary_text=summary_text, report=report, ind=ind)

        if report is not None:
            report.new_row()
        if policy is not None:
            plot_policy_means(policy, self, self.observation_space.bounds, report, goals=goals)

    def plot_paths(self, preupdate_obs=None, postupdate_obs=None, demo_obs=None, goal=None, summary_text="",
                   report=None, ind=0):
        plt.clf()
        if goal is not None:
            plt.plot(*goal, 'r*', markersize=10, label='goal')
        if preupdate_obs is not None:
            plt.plot(preupdate_obs[:, 0], preupdate_obs[:, 1], '-m', linewidth=1, label='preupdate')
        if postupdate_obs is not None:
            plt.plot(postupdate_obs[:, 0], postupdate_obs[:, 1], '-b', linewidth=1, label='post-update')
        if demo_obs is not None:
            plt.plot(demo_obs[:, 0], demo_obs[:, 1], '-.g', linewidth=2, label='demo')

        plt.legend()
        plt.plot(0, 0, 'b.', markersize=5)
        plt.xlim([-8, 8])
        plt.ylim([-8, 8])

        log_dir = logger.get_snapshot_dir()
        if report is None:
            plt.figtext(0.02, 0.02, summary_text)  # todo: pull image up to fit text without overlap
            plt.savefig(osp.join(log_dir, 'post_update_plot' + str(ind) + '.png'))
        else:
            vec_img = save_image()
            report.add_image(vec_img, summary_text)
        plt.close('all')


class StraightDemo(Policy):  # , Serializable):
    def __init__(self, *args, **kwargs):
        # Serializable.quick_init(self, locals())
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
    plt.close('all')


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
