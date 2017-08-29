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

files = glob.glob('data/s3/rllab-fixed-push-experts/*/*itr_300*')
files.sort()
demos_per_expert = 400
#output_dir = 'data/expert_demos/'

#use_filter = True
#filter_thresh = -34
#joint_thresh = 1.0
#max_num_tries=16
#output_dir = 'data/expert_demos_filter_joint0/'

use_filter = True
filter_thresh = -32
joint_thresh=0.7
max_num_tries=demos_per_expert*2

# use wood table
WOOD = True # note - never collected last 100 demos for WOOD=False
ONE_OBJECT = True
if WOOD:
    if ONE_OBJECT:
        output_dir = 'data/onetask_oneobj_push_demos_vision/'
    else:
        output_dir = 'data/onetask_wood_push_demos_vision/'
else:
    output_dir = 'data/onetask_push_demos_filter_vision/'

expert_inds = [201]

for expert_i in expert_inds:
    expert = files[expert_i]
    if expert_i % 25 == 0:
        print('collecting #' + str(expert_i))
    if '2017_06_23_21_04_45_0091' in expert:
        continue
    with tf.Session() as sess:
        data = joblib.load(expert)
        policy = data['policy']
        env = data['env']
        xml_file = env._wrapped_env._wrapped_env.FILE
        if WOOD:
            if ONE_OBJECT:
                prefix = '/home/cfinn/code/rllab/vendor/local_mujoco_models/oneobj_'
            else:
                prefix = '/home/cfinn/code/rllab/vendor/local_mujoco_models/woodtable_distractor_'
        else:
            prefix = '/home/cfinn/code/rllab/vendor/local_mujoco_models/distractor_'
        suffix = xml_file[xml_file.index('pusher'):]
        print(prefix+suffix)
        pusher_env = PusherVisionEnv(**{'xml_file':prefix+suffix, 'distractors': True})
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
            while (len(returns) < demos_per_expert and num_tries < max_num_tries):
                if num_tries % 25 == 0:
                    print(num_tries)
                num_tries += 1
                path = rollout(env, policy, max_path_length=100, speedup=1,
                         animated=True, always_return_paths=True, save_video=False, vision=True)
                if path['observations'][-1,0] > joint_thresh:
                    num_tries = max_num_tries
                if path['rewards'].sum() > filter_thresh and path['observations'][-1,0] < joint_thresh:
                    returns.append(path['rewards'].sum())
                    demoX.append(path['nonimage_obs'])
                    demoU.append(path['actions'])
                    videos.append(path['image_obs'])
        if len(returns) >= demos_per_expert:
            demoX = np.array(demoX)
            demoU = np.array(demoU)
            with open(output_dir + str(expert_i) + '.pkl', 'wb') as f:
                pickle.dump({'demoX': demoX, 'demoU': demoU, 'xml':prefix+suffix}, f, protocol=2)
            video_dir = output_dir + 'object_' + str(expert_i) + '/'
            os.mkdir(video_dir)
            for demo_index in range(demos_per_expert):
                imageio.mimwrite(video_dir + 'cond' + str(demo_index) + '.samp0.gif', list(videos[demo_index]), format='gif')
    tf.reset_default_graph()


