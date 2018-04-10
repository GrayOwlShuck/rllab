import pickle
from rllab.misc import special
from rllab.misc import tensor_utils
from rllab.algos import util
import rllab.misc.logger as logger
import tensorflow as tf
from rllab.sampler.base import BaseSampler
from sandbox.rocky.tf.envs.parallel_vec_env_executor import ParallelVecEnvExecutor
from sandbox.rocky.tf.envs.vec_env_executor import VecEnvExecutor
from rllab.misc import tensor_utils
import numpy as np
from rllab.sampler.stateful_pool import ProgBarCounter
import rllab.misc.logger as logger
import itertools


class VectorizedDemoSampler(BaseSampler):

    def __init__(self, algo, policy=None, n_envs=None, batch_size=None):  # allows to define another policy (for demos)!
        """
        :param policy: allows to define a sampling policy. uses 0 baseline!
        :param n_envs: 
        """
        super(VectorizedDemoSampler, self).__init__(algo, policy=policy)
        self.n_envs = n_envs
        self.batch_size = batch_size
        if batch_size is None:
            self.batch_size = algo.batch_size

    def process_samples(self, itr, paths, prefix='demo_', log=True):
        # if self.policy is not self.algo.policy:
        #     raise NotImplementedError("The ran policy was different: you will be fitting the wrong baseline!")
        # else:
        #     return super(VectorizedDemoSampler, self).process_samples(*args, **kwargs)

        baselines = []
        returns = []

        for idx, path in enumerate(paths):
            path["returns"] = special.discount_cumsum(path["rewards"], self.algo.discount)

        if log:
            logger.log("using zero baseline")
        all_path_baselines = [np.zeros_like(path['rewards']) for path in paths]

        for idx, path in enumerate(paths):
            path_baselines = np.append(all_path_baselines[idx], 0)
            deltas = path["rewards"] + \
                     self.algo.discount * path_baselines[1:] - \
                     path_baselines[:-1]
            path["advantages"] = special.discount_cumsum(
                deltas, self.algo.discount * self.algo.gae_lambda)
            baselines.append(path_baselines[:-1])
            returns.append(path["returns"])

        avg_path_length = np.mean([len(path["rewards"]) for path in paths])

        observations = tensor_utils.concat_tensor_list([path["observations"] for path in paths])
        actions = tensor_utils.concat_tensor_list([path["actions"] for path in paths])
        rewards = tensor_utils.concat_tensor_list([path["rewards"] for path in paths])
        returns = tensor_utils.concat_tensor_list([path["returns"] for path in paths])
        advantages = tensor_utils.concat_tensor_list([path["advantages"] for path in paths])
        env_infos = tensor_utils.concat_tensor_dict_list([path["env_infos"] for path in paths])
        agent_infos = tensor_utils.concat_tensor_dict_list([path["agent_infos"] for path in paths])

        if self.algo.center_adv:
            advantages = util.center_advantages(advantages)

        if self.algo.positive_adv:
            advantages = util.shift_advantages_to_positive(advantages)

        average_discounted_return = \
            np.mean([path["returns"][0] for path in paths])

        undiscounted_returns = [sum(path["rewards"]) for path in paths]

        samples_data = dict(
            observations=observations,
            actions=actions,
            rewards=rewards,
            returns=returns,
            advantages=advantages,
            env_infos=env_infos,
            agent_infos=agent_infos,
            paths=paths,
        )

        if log:
            logger.record_tabular(prefix+'AverageReturn', np.mean(undiscounted_returns))
            logger.record_tabular(prefix+'AveragePathLen', np.mean(avg_path_length))
            logger.record_tabular(prefix+'NumTrajs', len(paths))
            logger.record_tabular(prefix+'StdReturn', np.std(undiscounted_returns))
            logger.record_tabular(prefix+'MaxReturn', np.max(undiscounted_returns))
            logger.record_tabular(prefix+'MinReturn', np.min(undiscounted_returns))

        return samples_data

    def start_worker(self):
        print('starting worker for n_env=', self.n_envs)
        n_envs = self.n_envs
        if n_envs is None:
            n_envs = int(self.batch_size / self.algo.max_path_length)
            n_envs = max(1, min(n_envs, 100))
            print('now n_env=', n_envs)

        if getattr(self.algo.env, 'vectorized', False):
            print("the env is already vectorized, so we just take the vec_env_executor as self.vec_env")
            self.vec_env = self.algo.env.vec_env_executor(n_envs=n_envs, max_path_length=self.algo.max_path_length)
        else:
            print('the env is NOT vect so we pickle and load it n times!')
            envs = [pickle.loads(pickle.dumps(self.algo.env)) for _ in range(n_envs)]
            self.vec_env = VecEnvExecutor(
                envs=envs,
                #env=pickle.loads(pickle.dumps(self.algo.env)),
                #n = n_envs,
                max_path_length=self.algo.max_path_length
            )
        self.env_spec = self.algo.env.spec

    def shutdown_worker(self):
        self.vec_env.terminate()

    def obtain_samples(self, itr, return_dict=False, num_rollouts_per_env=1, *reset_args, **reset_kwargs):
        # return_dict: whether or not to return a dictionary or list form of paths
        # *reset_args: every arg can be a list of length num_envs or a single arg that will be copied equally
        # **reset_kwargs: every kwargs can be a list of length num_envs or a single kwarg that will be copied eq.

        logger.log("Obtaining samples for iteration %d..." % itr)

        paths = {}
        for i in range(self.vec_env.num_envs):
            paths[i] = []

        n_samples = 0

        obses = self.vec_env.reset(*reset_args, **reset_kwargs)
        dones = np.asarray([True] * self.vec_env.num_envs)
        running_paths = [None] * self.vec_env.num_envs

        # in case no specific goal was given, use the ones sampled when reseting the envs!
        if not reset_args and not reset_kwargs:
            reset_kwargs['objective_params'] = [env.wrapped_env.wrapped_env.objective_params for env in self.vec_env.envs]  # todo: hack! don't hand-unwrap!

        pbar = ProgBarCounter(self.batch_size)
        policy_time = 0
        env_time = 0
        process_time = 0

        import time

        while n_samples < self.batch_size:
            t = time.time()
            self.policy.reset(dones)
            actions, agent_infos = self.policy.get_actions(obses, *reset_args, **reset_kwargs)  # uses the same reset_args than the env!

            policy_time += time.time() - t
            t = time.time()
            next_obses, rewards, dones, env_infos = self.vec_env.step(actions)

            if np.any(dones):
                new_obses = self.vec_env.reset(dones, *reset_args, **reset_kwargs)
                reset_idx = 0
                for idx, done in enumerate(dones):
                    if done:
                        next_obses[idx] = new_obses[reset_idx]
                        reset_idx += 1

            env_time += time.time() - t
            t = time.time()

            agent_infos = tensor_utils.split_tensor_dict_list(agent_infos)
            env_infos = tensor_utils.split_tensor_dict_list(env_infos)
            if env_infos is None:
                env_infos = [dict() for _ in range(self.vec_env.num_envs)]
            if agent_infos is None:
                agent_infos = [dict() for _ in range(self.vec_env.num_envs)]
            for idx, observation, action, reward, env_info, agent_info, done in zip(itertools.count(), obses, actions,
                                                                                    rewards, env_infos, agent_infos,
                                                                                    dones):
                if running_paths[idx] is None:
                    running_paths[idx] = dict(
                        observations=[],
                        actions=[],
                        rewards=[],
                        env_infos=[],
                        agent_infos=[],
                    )
                running_paths[idx]["observations"].append(observation)
                running_paths[idx]["actions"].append(action)
                running_paths[idx]["rewards"].append(reward)
                running_paths[idx]["env_infos"].append(env_info)
                running_paths[idx]["agent_infos"].append(agent_info)
                if done:
                    paths[idx].append(dict(
                        observations=self.env_spec.observation_space.flatten_n(running_paths[idx]["observations"]),
                        actions=self.env_spec.action_space.flatten_n(running_paths[idx]["actions"]),
                        rewards=tensor_utils.stack_tensor_list(running_paths[idx]["rewards"]),
                        env_infos=tensor_utils.stack_tensor_dict_list(running_paths[idx]["env_infos"]),
                        agent_infos=tensor_utils.stack_tensor_dict_list(running_paths[idx]["agent_infos"]),
                    ))
                    n_samples += len(running_paths[idx]["rewards"])
                    running_paths[idx] = None
            process_time += time.time() - t
            pbar.inc(len(obses))
            obses = next_obses

        pbar.stop()

        logger.record_tabular("PolicyExecTime", policy_time)
        logger.record_tabular("EnvExecTime", env_time)
        logger.record_tabular("ProcessExecTime", process_time)

        if not return_dict:
            flatten_list = lambda l: [item for sublist in l for item in sublist]
            paths = flatten_list(paths.values())
            #path_keys = flatten_list([[key]*len(paths[key]) for key in paths.keys()])

        return paths

"""
    def new_obtain_samples(self, itr, reset_args=None, return_dict=False):
        # reset_args: arguments to pass to the environments to reset
        # return_dict: whether or not to return a dictionary or list form of paths

        logger.log("Obtaining samples for iteration %d..." % itr)

        #paths = []
        paths = {}
        for i in range(self.algo.meta_batch_size):
            paths[i] = []

        n_samples = 0
        pbar = ProgBarCounter(self.batch_size)
        policy_time = 0
        env_time = 0
        process_time = 0
        policy = self.algo.policy
        import time

        num_tasks = self.algo.meta_batch_size
        task_batch = self.vec_env.num_envs / num_tasks
        # inds 0 through task_batch are task 0, the next task_batch are task 1, etc.
        task_idxs = np.reshape(np.tile(np.arange(num_tasks), [task_batch,1]).T, [-1])
        obses = self.vec_env.reset([reset_args[idx] for idx in task_idxs])
        dones = np.asarray([True] * self.vec_env.num_envs)
        running_paths = [None] * self.vec_env.num_envs

        while n_samples <= self.batch_size:

            t = time.time()
            policy.reset(dones)
            actions, agent_infos = policy.get_actions(obses)

            policy_time += time.time() - t
            t = time.time()
            next_obses, rewards, dones, env_infos = self.vec_env.step(actions)
            env_time += time.time() - t

            t = time.time()

            agent_infos = tensor_utils.split_tensor_dict_list(agent_infos)
            env_infos = tensor_utils.split_tensor_dict_list(env_infos)

            if env_infos is None:
                env_infos = [dict() for _ in range(self.vec_env.num_envs)]
            if agent_infos is None:
                agent_infos = [dict() for _ in range(self.vec_env.num_envs)]

            for idx, observation, action, reward, env_info, agent_info, done in zip(itertools.count(), obses, actions,
                                                                                    rewards, env_infos, agent_infos,
                                                                                    dones):
                if running_paths[idx] is None:
                    running_paths[idx] = dict(
                        observations=[],
                        actions=[],
                        rewards=[],
                        env_infos=[],
                        agent_infos=[],
                        )
                running_paths[idx]["observations"].append(observation)
                running_paths[idx]["actions"].append(action)
                running_paths[idx]["rewards"].append(reward)
                running_paths[idx]["env_infos"].append(env_info)
                running_paths[idx]["agent_infos"].append(agent_info)
                if done:
                    paths[task_idxs[idx]].append(dict(
                        observations=self.env_spec.observation_space.flatten_n(running_paths[idx]["observations"]),
                        actions=self.env_spec.action_space.flatten_n(running_paths[idx]["actions"]),
                        rewards=tensor_utils.stack_tensor_list(running_paths[idx]["rewards"]),
                        env_infos=tensor_utils.stack_tensor_dict_list(running_paths[idx]["env_infos"]),
                        agent_infos=tensor_utils.stack_tensor_dict_list(running_paths[idx]["agent_infos"]),
                    ))
                    n_samples += len(running_paths[idx]["rewards"])
                    running_paths[idx] = None
            process_time += time.time() - t
            pbar.inc(len(obses))
            obses = next_obses

        pbar.stop()

        logger.record_tabular("PolicyExecTime", policy_time)
        logger.record_tabular("EnvExecTime", env_time)
        logger.record_tabular("ProcessExecTime", process_time)

        if not return_dict:
            flatten_list = lambda l: [item for sublist in l for item in sublist]
            paths = flatten_list(paths.values())
            #path_keys = flatten_list([[key]*len(paths[key]) for key in paths.keys()])

        return paths
"""

