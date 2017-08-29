import argparse

import glob
import imageio
import joblib
import numpy as np
import os
import tensorflow as tf

from rllab.misc.console import query_yes_no
from rllab.sampler.utils import rollout
import pickle

from rllab.envs.mujoco.pusher_vision_env import PusherVisionEnv
from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.envs.base import TfEnv

def eval_success(path):
      obs = path['observations']
      target = obs[:, -3:-1]
      obj = obs[:, -6:-4]
      dists = np.sum((target-obj)**2, 1)  # distances at each timestep
      return np.sum(dists < 0.017) >= 10


#files = glob.glob('data/s3/rllab-fixed-push-experts/*/*itr_300*')
#files = glob.glob('data/s3/init5-push-experts/*/*itr_300*')
files = glob.glob('data/s3/init5-push-experts-test/*/*itr_300*')
files.sort()
demos_per_expert = 24 #8
#output_dir = 'data/expert_demos/'

#use_filter = True
#filter_thresh = -34
#joint_thresh = 1.0
#max_num_tries=16
#output_dir = 'data/expert_demos_filter_joint0/'

use_filter = True
filter_thresh = -32
joint_thresh=0.7
max_num_tries=34 #30 #12

# use wood table
WOOD = True # note - never collected last 100 demos for WOOD=False
TWO_TEXTURES = False
ONE_OBJECT = False #True
ENSURE_DOUBLE_DEMO = True
"""
if WOOD:
    if TWO_TEXTURES:
        if ENSURE_DOUBLE_DEMO:
            output_dir = 'data/ensure_twotext_push_demos_filter_vision24/'
        else:
            output_dir = 'data/twotext_wood_fixed_push_demos_filter_vision24/'
    elif ENSURE_DOUBLE_DEMO:
        output_dir = 'data/ensure_push_demos_filter_vision24/'
    elif ONE_OBJECT:
        output_dir = 'data/oneobj_push_demos_filter_vision24/'
    else:
        output_dir = 'data/new_wood_fixed_push_demos_filter_vision/'
else:
    output_dir = 'data/fixed_push_demos_filter_vision/'
"""

output_dir = 'data/test_paired_consistent_push_demos/'

TEST2 = True  # if True, use held out textures.
if TEST2:
    output_dir = 'data/test2_paired_consistent_push_demos/'

#os.mkdir(output_dir)

# do for first 100 experts
#offset = -1000  # offset to get expert policy
#expert_inds = range(0,100)
offset = 0
expert_inds = range(0,20)

for expert_i in expert_inds:
    expert = files[expert_i+offset]
    if expert_i % 25 == 0:
        print('collecting #' + str(expert_i))
    if '2017_06_23_21_04_45_0091' in expert:
        continue
    with tf.Session() as sess:
        data = joblib.load(expert)
        policy = data['policy']
        env = data['env']
        xml_file = env._wrapped_env._wrapped_env.FILE
        print(xml_file)
        #if WOOD:
        #    if ENSURE_DOUBLE_DEMO and TWO_TEXTURES:
        #        prefix = '/home/cfinn/code/rllab/vendor/local_mujoco_models/ensure_texture2_woodtable_distractor_'
        #    elif TWO_TEXTURES:
        #        prefix = '/home/cfinn/code/rllab/vendor/local_mujoco_models/texture2_woodtable_distractor_'
        #    elif ENSURE_DOUBLE_DEMO:
        #        prefix = '/home/cfinn/code/rllab/vendor/local_mujoco_models/ensure_woodtable_distractor_'
        #    elif ONE_OBJECT:
        #        prefix = '/home/cfinn/code/rllab/vendor/local_mujoco_models/oneobj_'
        #    else:
        #        prefix = '/home/cfinn/code/rllab/vendor/local_mujoco_models/woodtable_distractor_'
        #else:
        #    prefix = '/home/cfinn/code/rllab/vendor/local_mujoco_models/distractor_'
        ##suffix = xml_file[xml_file.index('pusher'):]
        #suffix = 'pusher' + str(expert_i+1) + '.xml'
        #print(prefix+suffix)
        if TEST2:
            xml_file = xml_file.replace('test', 'test2')
        pusher_env = PusherVisionEnv(**{'xml_file':xml_file, 'distractors': True})
        env = TfEnv(normalize(pusher_env))
        returns = []
        demoX = []
        demoU = []
        videos = []
        if not use_filter:
            for _ in range(demos_per_expert):
                path = rollout(env, policy, max_path_length=100, speedup=1,
                               animated=True, always_return_paths=True, save_video=False)
                returns.append(path['rewards'].sum())
                demoX.append(path['nonimage_obs'])
                demoU.append(path['actions'])
                #print(path['rewards'].sum())
        else:
            num_tries = 0
            obj_left = True
            while (len(returns) < demos_per_expert and num_tries < max_num_tries):
                num_tries += 1
                if len(returns) >= demos_per_expert / 2.0:
                    obj_left = False
                path = rollout(env, policy, max_path_length=100, speedup=1, left=obj_left,
                         animated=True, always_return_paths=True, save_video=False, vision=True)
                if path['observations'][-1,0] > joint_thresh:
                    num_tries = max_num_tries
                #if path['rewards'].sum() > filter_thresh and path['observations'][-1,0] < joint_thresh:
                if eval_success(path) and path['observations'][-1,0] < joint_thresh:
                    returns.append(path['rewards'].sum())
                    demoX.append(path['nonimage_obs'])
                    demoU.append(path['actions'])
                    videos.append(path['image_obs'])
        if len(returns) >= demos_per_expert:
            demoX = np.array(demoX)
            demoU = np.array(demoU)
            with open(output_dir + str(expert_i) + '.pkl', 'wb') as f:
                #pickle.dump({'demoX': demoX, 'demoU': demoU, 'xml':prefix+suffix}, f, protocol=2)
                pickle.dump({'demoX': demoX, 'demoU': demoU, 'xml':xml_file}, f, protocol=2)
            video_dir = output_dir + 'object_' + str(expert_i) + '/'
            os.mkdir(video_dir)
            for demo_index in range(demos_per_expert):
                imageio.mimwrite(video_dir + 'cond' + str(demo_index) + '.samp0.gif', list(videos[demo_index]), format='gif')
    tf.reset_default_graph()


