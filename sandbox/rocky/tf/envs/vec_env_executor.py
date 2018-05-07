import numpy as np
import pickle as pickle
from sandbox.rocky.tf.misc import tensor_utils


class VecEnvExecutor(object):
    def __init__(self, envs, max_path_length):
        self.envs = envs
        self._action_space = envs[0].action_space
        self._observation_space = envs[0].observation_space
        self.ts = np.zeros(len(self.envs), dtype='int')
        self.max_path_length = max_path_length

    def step(self, action_n):
        # import pdb; pdb.set_trace()
        all_results = [env.step(a) for (a, env) in zip(action_n, self.envs)]
        obs, rewards, dones, env_infos = list(map(list, list(zip(*all_results))))
        dones = np.asarray(dones)
        rewards = np.asarray(rewards)
        self.ts += 1
        if self.max_path_length is not None:
            dones[self.ts >= self.max_path_length] = True
        # for (i, done) in enumerate(dones):
        #     if done:
        #         obs[i] = self.envs[i].reset(reset_args[i])
        #         self.ts[i] = 0
        return obs, rewards, dones, tensor_utils.stack_tensor_dict_list(env_infos)

    def reset(self, dones=None, force_expand=False, *reset_args, **reset_kwargs):
        # if not specify the dones, reset all the envs
        if dones is None:
            dones = [True] * self.num_envs

        # if the elements of reset_args are not lists of num_envs, we set the same arg for each env
        expanded_reset_args = []
        for arg in reset_args:
            if not hasattr(arg, '__len__') or len(arg) != self.num_envs:
                expanded_reset_args.append([arg] * self.num_envs)
            else:
                expanded_reset_args.append(arg)
        if len(expanded_reset_args) == 0:
            expanded_reset_args = [()] * self.num_envs

        # if the reset args are not list of dirs, we set the same args for each env
        expanded_reset_kwargs = [{} for _ in range(self.num_envs)]
        for kw, val in reset_kwargs.items():
            if not hasattr(val, '__len__') or len(val) != self.num_envs \
                    or force_expand:  # if len num_envs use force_expand!
                for env_idx in range(self.num_envs):
                    expanded_reset_kwargs[env_idx][kw] = val
            else:
                for env_idx, kwarg in enumerate(val):
                    expanded_reset_kwargs[env_idx][kw] = kwarg

        # print("inside VectEnvEx: expanded_reset_arg=", expanded_reset_args, ", expanded_reset_kwargs=", expanded_reset_kwargs)
        dones = np.cast['bool'](dones)
        results = [env.reset(*arg, **kwargs) for idx, env, arg, kwargs in
                   zip(range(self.num_envs), self.envs, expanded_reset_args, expanded_reset_kwargs) if dones[idx]]
        self.ts[dones] = 0
        return results

    @property
    def num_envs(self):
        return len(self.envs)

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    def terminate(self):
        pass
