import argparse

import joblib
import numpy as np
import tensorflow as tf

from rllab.misc.console import query_yes_no
from rllab.sampler.utils import rollout

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('out_file', type=str,
                        help='path to new data file')
    parser.add_argument('--max_path_length', type=int, default=1000,
                        help='Max length of rollout')
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of roll outs')
    args = parser.parse_args()

    # If the snapshot file use tensorflow, do:
    # import tensorflow as tf
    # with tf.Session():
    #     [rest of the code]
    with tf.Session() as sess:
        data = joblib.load(args.file)
        policy = data['policy']
        env = data['env']
        all_paths = []
        for _ in range(args.num_rollouts):
            path = rollout(env, policy, max_path_length=args.max_path_length,
                           animated=False)
            all_paths.append(path)

        returns = [np.sum(path['rewards']) for path in all_paths]
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        print 'Expert Policy Return', avg_return, '+-', std_return
        import pdb; pdb.set_trace()

        joblib.dump(all_paths, args.out_file, compress=3)


