import argparse
import glob
import imageio
import joblib
import pickle
import numpy as np
import random
import tensorflow as tf
from PIL import Image
from sampler_utils import rollout
import sys

USE_RLLAB = True
CROP = False

if USE_RLLAB:
    from rllab.envs.mujoco.pusher_vision_env import PusherVisionEnv
    from rllab.envs.normalized_env import normalize
    from sandbox.rocky.tf.envs.base import TfEnv
else:
    from gym.envs.mujoco.pusher import PusherEnv

XML_PATH = '/home/cfinn/code/rllab/vendor/local_mujoco_models/'


class TFAgent(object):
    def __init__(self, tf_weights_file, scale_bias_file, sess, lstm=False):
        self.sess = sess
        new_saver = tf.train.import_meta_graph(tf_weights_file)
        new_saver.restore(self.sess, tf_weights_file[:-5])

        if scale_bias_file:
            with open(scale_bias_file, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
            self.scale = data['scale']
            self.bias = data['bias']
            self.statea = tf.get_default_graph().get_tensor_by_name('statea:0')
            self.stateb = tf.get_default_graph().get_tensor_by_name('stateb:0')
        else:
            self.scale = None
        self.lstm = lstm

        self.imagea = tf.get_default_graph().get_tensor_by_name('obsa:0')
        self.actiona = tf.get_default_graph().get_tensor_by_name('actiona:0')
        self.imageb = tf.get_default_graph().get_tensor_by_name('obsb:0')
        self.actionb = tf.get_default_graph().get_tensor_by_name('output_action:0')


    def reset(self):
        pass

    def set_demo(self, demo_gif, demoX, demoU):
        #import pdb; pdb.set_trace()

        # concatenate demos in time
        demo_gif = np.array(demo_gif)
        N, T, H, W, C = demo_gif.shape
        self.update_batch_size = N
        self.T = T
        demo_gif = np.reshape(demo_gif, [N*T, H, W, C])
        demo_gif = np.array(demo_gif)[:,:,:,:3].transpose(0,3,2,1).astype('float32') / 255.0
        self.demoVideo = demo_gif.reshape(1, N*T, -1)
        self.demoX = demoX
        self.demoU = demoU

    def get_loss(self, demo_gif, demoX, demoU):
        np.set_printoptions(2)
        demo_gif = np.array(demo_gif)[:,:,:,:3].transpose(0,3,2,1).astype('float32') / 255.0
        T = demo_gif.shape[0]
        demo_gif = demo_gif.reshape(1, T, -1)

        # for final state model
        all_loss_vals = []

        for i in range(100):
            demoX_t = demoX[:,i:i+1,:]
            demoU_t = demoU[:,i:i+1,:]
            demo_gif_t = demo_gif[:,i:i+1,:]
            if self.scale is not None:
                actions = self.sess.run(self.actionb,
                                   {self.statea: self.demoX.dot(self.scale) + self.bias,
                                   self.imagea: self.demoVideo,
                                   self.actiona: self.demoU,
                                   self.stateb: demoX_t.dot(self.scale) + self.bias,
                                   self.imageb: demo_gif_t})
            else:
                actions = self.sess.run(self.actionb,
                                    {self.imagea: self.demoVideo,
                                    self.actiona: self.demoU,
                                    self.imageb: demo_gif_t})
            all_loss_vals.append(np.mean((50*(actions-demoU_t))**2))
        print(np.array(all_loss_vals))
        loss_val = np.mean(all_loss_vals)
        print('Loss value: ' + str(loss_val))
        # compare actions to demoU
        import pdb; pdb.set_trace()

    def get_action(self, obs):
        obs = obs.reshape((1,1,23))
        action = self.sess.run(self.actionb, {self.statea: self.demoX.dot(self.scale) + self.bias,
                               self.actiona: self.demoU,
                               self.stateb: obs.dot(self.scale) + self.bias})
        return action, dict()

    def get_vision_action(self, image, obs, t=-1):
        if CROP:
            image = np.array(Image.fromarray(image).crop((40,25,120,90)))

        image = np.expand_dims(image, 0).transpose(0,3,2,1).astype('float32') / 255.0
        image = image.reshape((1, 1, -1))

        obs = obs.reshape((1,1,20))

        if t == 0 and self.lstm:
            self.all_image = np.zeros((1, self.T, image.shape[-1]))
            self.all_obs = np.zeros((1, self.T, obs.shape[-1]))
        if self.lstm:
            self.all_obs[:,t:t+1,:] = obs
            self.all_image[:,t:t+1,:] =  image
            obs = self.all_obs
            image = self.all_image


        if self.scale is not None:
            action = self.sess.run(self.actionb,
                               {self.statea: self.demoX.dot(self.scale) + self.bias,
                               self.imagea: self.demoVideo,
                               self.actiona: self.demoU,
                               self.stateb: obs.dot(self.scale) + self.bias,
                               self.imageb: image})
        else:
            action = self.sess.run(self.actionb,
                                {self.imagea: self.demoVideo,
                                self.actiona: self.demoU,
                                self.imageb: image})
        if self.lstm:
            # TODO get t slice from action
            action = action[:,t,:]
        return action, dict()

def load_env(demo_info, args):
    xml_filepath = demo_info['xml']
    suffix = xml_filepath[xml_filepath.index('pusher'):]
    if args.train:
        prefix = XML_PATH + 'train_ensure_woodtable_distractor_'
    else:
        if args.test2:
            prefix = XML_PATH + 'test2_ensure_woodtable_distractor_'
        else:
            prefix = XML_PATH + 'test_ensure_woodtable_distractor_'
    #prefix = XML_PATH + 'ensure_woodtable_distractor_'
    xml_filepath = str(prefix + suffix)
    print(xml_filepath)

    if USE_RLLAB:
        pusher_env = PusherVisionEnv(**{'xml_file':xml_filepath, 'distractors': True})
        env = TfEnv(normalize(pusher_env))
    else:
        env = PusherEnv(**{'xml_file':xml_filepath, 'distractors': True})
    return env

def load_demo(task_id, demo_dir, demo_inds, get_loss=False):
    demo_info = pickle.load(open(demo_dir+task_id+'.pkl', 'rb'))
    #demoX = demo_info['demoX'][demo_ind:demo_ind+1,:,:]
    #demoU = demo_info['demoU'][demo_ind:demo_ind+1,:,:]
    demoX = demo_info['demoX'][demo_inds,:,:]
    demoU = demo_info['demoU'][demo_inds,:,:]
    d1, d2, _ = demoX.shape
    demoX = np.reshape(demoX, [1, d1*d2, -1])
    demoU = np.reshape(demoU, [1, d1*d2, -1])

    if get_loss:
        demo_test = random.randint(0, 24)
        print(demo_inds)
        print(demo_test)
        demoX_test = demo_info['demoX'][demo_test:demo_test+1,:,:]
        demoU_test = demo_info['demoU'][demo_test:demo_test+1,:,:]
        demo_gif_test = imageio.mimread(demo_dir + 'object_'+task_id+'/cond%d.samp0.gif' % demo_test)
        demo_test = [demoX_test, demoU_test, demo_gif_test]
    else:
        demo_test = None

    # read in demo video
    if CROP:
        demo_gifs = [imageio.mimread(demo_dir+'crop_object_'+task_id+'/cond%d.samp0.gif' % demo_ind) for demo_ind in demo_inds]
    else:
        demo_gifs = [imageio.mimread(demo_dir+'object_'+task_id+'/cond%d.samp0.gif' % demo_ind) for demo_ind in demo_inds]

    return demoX, demoU, demo_gifs, demo_test, demo_info


def eval_success(path):
      obs = path['observations']
      target = obs[:, -3:-1]
      obj = obs[:, -6:-4]
      dists = np.sum((target-obj)**2, 1)  # distances at each timestep
      return np.sum(dists < 0.017) >= 10

def get_expert(task_id):
    # TODO - this function won't work for newer demos.
    #task_id += 1
    #file_id = task_id % 100
    file_id = (task_id + 1) % 50

    #experts = glob.glob('data/s3/rllab-fixed-push-experts/*0'+str(file_id)+'/*itr_300*')
    # TODO - this assumes test tasks
    #experts = glob.glob('data/s3/init5-push-experts-test/*0'+str(file_id)+'/*itr_300*')
    experts = glob.glob('data/s3/init5-push-experts-test/*/*itr_300*')
    for expert_i, expert in enumerate(experts):
        debug_file = expert[:-11] + 'debug.log'
        with open(debug_file, 'r') as f:
            for line in f:
                if 'xml:' in line:
                    string = line[line.index('xml:'):]
                    xml_file = string[5:-1]
                    suffix = int(xml_file[xml_file.index('pusher')+6:-4])
                    break
        if suffix == task_id:
            break
        if expert_i == len(experts) - 1:
            import pdb; pdb.set_trace()

    return expert



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the meta graph')
    parser.add_argument('--scale_file', type=str, default=None,
                        help='path to the scale and bias ')
    parser.add_argument('--id', type=int, default=-1,
                        help='ID of pickle file')
    parser.add_argument('--save_video', type=int, default=0,
                        help='whether or not to save video (only happens if animate=1)')
    parser.add_argument('--get_loss', type=int, default=0,
                        help='evaluate loss instead of rolling out one-shot policy')
    parser.add_argument('--expert', type=int, default=0,
                        help='whether or not to use expert. If nonzero, ignores file')
    parser.add_argument('--train', type=int, default=0,
                        help='whether or not to use training data.')
    parser.add_argument('--num_input_demos', type=int, default=1,
                        help='number of input demos as input (e.g. 1 for 1-shot).')
    parser.add_argument('--lstm', type=int, default=0,
                        help='whther or not model is an LSTM')
    parser.add_argument('--test2', type=int, default=0,
                        help='whther or not to use test set with held out textures.')
    args = parser.parse_args()

    # need breakpoint here if you want to use a breakpoint in the future
    import pdb; pdb.set_trace()

    # load demos
    if args.train:
        #demo_dir = 'data/ensure_push_demos_filter_vision24/train/'
        demo_dir = 'data/paired_consistent_push_demos/'
        files = glob.glob(demo_dir + '*.pkl')
        #all_ids = [int(f[f.index('train/')+6:-4]) for f in files]
        all_ids = [int(f[f.index('demos/')+6:-4]) for f in files]
    else:
        if args.test2:
            demo_dir = 'data/test2_paired_consistent_push_demos/'
        else:
            demo_dir = 'data/test_paired_consistent_push_demos/'
        #demo_dir = 'data/ensure_push_demos_filter_vision24/test/'
        files = glob.glob(demo_dir + '*.pkl')
        #all_ids = [int(f[f.index('test/')+5:-4]) for f in files]
        all_ids = [int(f[f.index('demos/')+6:-4]) for f in files]

    all_ids.sort()
    num_success = 0
    num_trials = 0
    trials_per_task = 5

    if args.id == -1:
        task_ids = all_ids
    else:
        task_ids = [args.id]
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    random.seed(5)
    np.random.seed(5)

    for task_id in task_ids:
        demo_inds = [1] # for consistency of comparison
        if args.num_input_demos > 1:
            demo_inds += range(12, 12+int(args.num_input_demos / 2))
            demo_inds += range(2, 2+int((args.num_input_demos-1) / 2))
        assert len(demo_inds) == args.num_input_demos

        # TODO update load demo func
        demoX, demoU, demo_gifs, demo_test, demo_info = load_demo(str(task_id), demo_dir, demo_inds, args.get_loss)

        # load xml file
        env = load_env(demo_info, args)

        with tf.Session(config=config) as sess:
            if args.expert:
                expert = get_expert(task_id)
                data = joblib.load(expert)
                policy = data['policy']
            else:
                policy = TFAgent(args.file, args.scale_file, sess, lstm=args.lstm)
                policy.set_demo(demo_gifs, demoX, demoU)

            if args.get_loss:
                [demoX_test, demoU_test, demo_gif_test] = demo_test
                policy.get_loss(demo_gif_test, demoX_test, demoU_test)
            returns = []

            while True:
                # should replace demo_inds[0] with args.num_input_demos
                video_suffix = str(args.id) + 'demo_' + str(demo_inds[0]) + '_' + str(len(returns)) + '.gif'
                if 'final_state' in args.file:
                    video_suffix = 'final_state' + video_suffix

                path = rollout(env, policy, max_path_length=100, env_reset=True, #args.max_path_length,
                               animated=True, speedup=1, always_return_paths=True, save_video=bool(args.save_video), video_filename=video_suffix, vision=True)
                num_trials += 1
                if eval_success(path):
                    num_success += 1
                print('Return: '+str(path['rewards'].sum()))
                returns.append(path['rewards'].sum())
                print('Average Return so far: ' + str(np.mean(returns)))
                print('Success Rate so far: ' + str(float(num_success)/num_trials))
                sys.stdout.flush()
                if len(returns) > trials_per_task:
                    break
        tf.reset_default_graph()

