import argparse

import joblib
import numpy as np
import tensorflow as tf

from rllab.misc.console import query_yes_no
from rllab.sampler.utils import rollout

def eval_success(path):
      obs = path['observations']
      target = obs[:, -3:-1]
      obj = obs[:, -6:-4]
      dists = np.sum((target-obj)**2, 1)  # distances at each timestep
      return np.sum(dists < 0.017) >= 10


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--max_path_length', type=int, default=100,
                        help='Max length of rollout')
    parser.add_argument('--speedup', type=float, default=1,
                        help='Speedup')
    args = parser.parse_args()

    # If the snapshot file use tensorflow, do:
    # import tensorflow as tf
    # with tf.Session():
    #     [rest of the code]
    import pdb; pdb.set_trace()
    with tf.Session() as sess:
        data = joblib.load(args.file)
        policy = data['policy']
        env = data['env']
        returns = []
        while True:
            path = rollout(env, policy, max_path_length=args.max_path_length,
                           animated=True, speedup=args.speedup, always_return_paths=True, save_video=True)
            succ = eval_success(path)
            print('Success: ' + str(succ))
            returns.append(path['rewards'].sum())
            print('Return: '+str(path['rewards'].sum()))
            print('Action cost return: ' + str(np.sum(np.square(path['actions']))))
            print('Average Return so far: '+str(np.mean(returns)))
            import pdb; pdb.set_trace()
            #if not query_yes_no('Continue simulation?'):
            #    break
