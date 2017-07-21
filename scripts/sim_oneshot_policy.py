import argparse

import joblib
import pickle
import numpy as np
import tensorflow as tf

from rllab.misc.console import query_yes_no
from rllab.sampler.utils import rollout

from rllab.envs.mujoco.pusher_env import PusherEnv
from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.envs.base import TfEnv

class TFAgent(object):
    def __init__(self, tf_weights_file, scale_bias_file, sess):
        with open(scale_bias_file, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        self.scale = data['scale']
        self.bias = data['bias']
        self.sess = sess

        new_saver = tf.train.import_meta_graph(tf_weights_file)
        new_saver.restore(self.sess, tf_weights_file[:-5])

        self.statea = tf.get_default_graph().get_tensor_by_name('statea:0')
        self.actiona = tf.get_default_graph().get_tensor_by_name('actiona:0')
        self.stateb = tf.get_default_graph().get_tensor_by_name('stateb:0')
        self.actionb = tf.get_default_graph().get_tensor_by_name('output_action:0')

    def reset(self):
        pass

    def set_demo(self, demoX, demoU):
        self.demoX = demoX
        self.demoU = demoU

    def get_action(self, obs):
        obs = obs.reshape((1,1,23))
        action = self.sess.run(self.actionb, {self.statea: self.demoX.dot(self.scale) + self.bias,
                               self.actiona: self.demoU,
                               self.stateb: obs.dot(self.scale) + self.bias})
        return action, dict()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the meta graph')
    parser.add_argument('scale_file', type=str,
                        help='path to the scale and bias ')
    parser.add_argument('--id', type=int, default=1,
                        help='ID of pickle file')
    parser.add_argument('--animated', type=int, default=0,
                        help='whether or not to animate')
    parser.add_argument('--save_video', type=int, default=1,
                        help='whether or not to save video (only happens if animate=1)')
    parser.add_argument('--distractor', type=int, default=0,
                        help='include distractor in environment')
    args = parser.parse_args()

    demo_info = pickle.load(open('data/expert_demos_filter_rew32_jnt7e-1_tries12/'+str(args.id)+'.pkl', 'rb'))
    xml_filepath = demo_info['xml']
    demoX = demo_info['demoX'][0:1,:,:]
    demoU = demo_info['demoU'][0:1,:,:]

    suffix = xml_filepath[xml_filepath.index('pusher'):]
    prefix = '/home/cfinn/code/rllab/vendor/local_mujoco_models/'
    if args.distractor:
        prefix += 'distractor_'
    xml_filepath = str(prefix + suffix)

    pusher_env = PusherEnv(**{'xml_file':xml_filepath, 'distractors': bool(args.distractor)})
    env = TfEnv(normalize(pusher_env))
    import pdb; pdb.set_trace()

    with tf.Session() as sess:
        policy = TFAgent(args.file, args.scale_file, sess)
        policy.set_demo(demoX, demoU)
        returns = []
        while True:
            path = rollout(env, policy, max_path_length=100, #args.max_path_length,
                           animated=bool(args.animated), speedup=1, always_return_paths=True, save_video=bool(args.save_video))
            print('Return: '+str(path['rewards'].sum()))
            returns.append(path['rewards'].sum())
            print('Average Return so far: ' + str(np.mean(returns)))
            if not query_yes_no('Continue simulation?'):
                break

