import matplotlib
from matplotlib import pyplot as plt
import pickle as pkl

import argparse
import os.path as osp
import random
from multiprocessing import cpu_count
import numpy as np
import tensorflow as tf

from rllab import config
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.zero_baseline import ZeroBaseline
from rllab.envs.normalized_env import normalize
from rllab.misc import logger
from rllab.misc.instrument import VariantGenerator
from rllab.misc.instrument import run_experiment_lite
from sandbox.carlos.point_env_rand2goals import PointEnvRand2Goal, StraightDemo2goals
from sandbox.rocky.tf.algos.sensitive_lfd_trpo import SensitiveLfD_TRPO
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.policies.sens_lfd_minimal_gauss_mlp_policy import SensitiveLfdGaussianMLPPolicy
from rllab.sampler.utils import rollout, rollout_demo
from sandbox.rocky.tf.envs.vec_env_executor import VecEnvExecutor
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.young_clgan.logging import HTMLReport
from sandbox.young_clgan.logging import format_dict


vg = VariantGenerator()

vg.add('env_noise', [0.1, 0.05])
vg.add('tolerance', [0.5])
vg.add('max_path_length', [100])
# vg.add('num_grad_updates', [1, 10, 100])
vg.add('step_size', [0.01, 0.05])
vg.add('demo_batch_size', [100, 10, 1])  # num of demos
vg.add('discount', [0.99])
vg.add('seed', range(0, 50, 10))

EXPERIMENT_TYPE = osp.basename(__file__).split('.')[0]
exp_prefix = 'lfd-grads2goals'
default_log_dir = osp.join(config.LOG_DIR, 'local', exp_prefix)


def main(v):
    log_dir = logger.get_snapshot_dir()
    exp_name = osp.split(log_dir)[-1]
    report = HTMLReport(osp.join(log_dir, 'report.html'), images_per_row=5)
    report.add_header("{}: {}".format(EXPERIMENT_TYPE, exp_name))
    report.add_text(format_dict(v))
    report.save()

    random.seed(v['seed'])

    env = TfEnv(normalize(PointEnvRand2Goal(noise=v['env_noise'], tolerance=v['tolerance'])))

    policy = SensitiveLfdGaussianMLPPolicy(
        name="policy",
        env_spec=env.spec,
        # grad_step_size=v['fast_learning_rate'],  # todo: check that this is not used here
        hidden_nonlinearity=tf.nn.relu,
        hidden_sizes=(100, 100),
    )

    demo_policy = StraightDemo2goals(env_spec=env.spec)

    obs_demo_vars = env.observation_space.new_tensor_variable(
        'demo_obs',
        extra_dims=1,
        )
    action_demo_vars = env.action_space.new_tensor_variable(
        'demo_actions',
        extra_dims=1,
        )

    dist = policy.distribution

    old_dist_info_demo_vars = {
        k: tf.placeholder(tf.float32, shape=[None] + list(shape), name='old_%s' % (k))
        for k, shape in dist.dist_info_specs
        }
    old_dist_info_vars_demo_list = [old_dist_info_demo_vars[k] for k in dist.dist_info_keys] # list version of the dict

    input_list = [obs_demo_vars, action_demo_vars]

    dist_info_demo_vars, params = policy.dist_info_sym(obs_demo_vars, all_params=policy.all_params)

    kl = dist.kl_sym(old_dist_info_demo_vars, dist_info_demo_vars)

    logli_demo = dist.log_likelihood_sym(action_demo_vars, dist_info_demo_vars)

    bc_loss = - tf.reduce_mean(logli_demo)
    # tf.summary.scalar('bc_loss', bc_loss)


    def update_dist_info_lfd_sym(surr_obj, params, step_size=0.01):

        gradients = dict(
            zip(update_param_keys, tf.gradients(surr_obj, [params[key] for key in update_param_keys])))
        new_params = dict(
            zip(update_param_keys, [params[key] - step_size * gradients[key] for key in update_param_keys]))

        return new_params

    update_param_keys = params.keys()

    new_params = update_dist_info_lfd_sym(bc_loss, params, v['step_size'])
    new_dist_info_demo_vars, new_params = policy.dist_info_sym(obs_demo_vars, all_params=new_params)  # try to replace policy!!

    with tf.Session() as sess:
        initializer = tf.global_variables_initializer()
        sess.run(initializer)

        demos = [rollout_demo(env, demo_policy, v['max_path_length']) for _ in range(v['demo_batch_size'])]

        demos_flat_obs = np.concatenate([demo['observations'] for demo in demos], axis=0)
        demos_flat_act = np.concatenate([demo['actions'] for demo in demos], axis=0)
        demos_avg_success = np.mean([np.any(demo['env_infos']['goal_reached']) for demo in demos])
        demos_avg_rewards = np.mean([np.sum(p['rewards']) for p in demos])
        demos_avg_len = np.mean([len(p['rewards']) for p in demos])
        summary_text = 'AvgDemRew: {:.2f}, AvgDemSucc: {:.2f}, AvgLen: {:.2f}\n'.format(np.mean(demos_avg_rewards),
                                                                                         np.mean(demos_avg_success),
                                                                                         np.mean(demos_avg_len))
        for i in range(min(v['demo_batch_size'], 5)):
            env.plot_paths(demo_obs=demos[i]['observations'], goal_idx=demos[i]['env_infos']['obj'][0],
                           report=report, summary_text=summary_text)
        report.new_row()

        input_list_dict = {obs_demo_vars: demos_flat_obs, action_demo_vars: demos_flat_act}

        # tensorboard
        # merged = tf.summary.merge_all()
        # train_writer = tf.summary.FileWriter('data/tf_summaries/', sess.graph)
        # summary, bc_loss_val, new_params_vals = sess.run([merged, bc_loss, new_params], feed_dict=input_list_dict)
        # train_writer.add_summary(summary, 0)
        bc_loss_val, new_params_vals = sess.run([bc_loss, new_params], feed_dict=input_list_dict)
        # logger.record_tabular('bc_loss', bc_loss_val)

        vect_env = VecEnvExecutor(envs=[env], max_path_length=v['max_path_length'])

        for i in range(10000):  # for more steps I should
            bc_loss_val, new_params_vals = sess.run([bc_loss, new_params], feed_dict=input_list_dict)
            policy.assign_params(policy.all_params, new_params_vals)
            if i % 50 == 0:
                logger.record_tabular('bc_loss', bc_loss_val)
                print('Loss after update %d: %f' % (i, bc_loss_val))
                eval_paths = [rollout(vect_env, policy, v['max_path_length']) for _ in range(10)]
                avg_reward = np.mean([np.sum(p['rewards']) for p in eval_paths])
                avg_success = np.mean([np.any(p['env_infos']['goal_reached']) for p in eval_paths])
                avg_len = np.mean([len(p['rewards']) for p in eval_paths])
                logger.record_tabular('AvgReward', avg_reward)
                logger.record_tabular('AvgLen', avg_len)
                env.log_diagnostics(eval_paths, report=report)
                summary_text = 'AvgRew: {:.2f}, AvgSucc: {:.2f}, AvgLen: {:.2f}\n'.format(np.mean(avg_reward),
                                                                                          np.mean(avg_success),
                                                                                          np.mean(avg_len))
                for j in range(4):
                    env.plot_paths(postupdate_obs=eval_paths[j]['observations'],
                                   goal_idx=int(eval_paths[j]['env_infos']['obj'][0]),
                                   report=report, summary_text=summary_text)
                report.new_row()
                logger.dump_tabular()


for v in vg.variants():
    run_experiment_lite(
        use_cloudpickle=True,
        stub_method_call=main,
        exp_prefix=exp_prefix,
        # Number of parallel workers for sampling
        n_parallel=0,
        # Only keep the snapshot parameters for the last iteration
        snapshot_mode="last",
        # Specifies the seed for the experiment. If this is not provided, a random seed
        # will be used
        seed=v["seed"],
        mode='local',
        # mode="local_docker",
        variant=v,
        # plot=True,
        # terminate_machine=False,
        pre_commands=[
            'pip install --upgrade pip',
            'pip install --upgrade cloudpickle',
            'export MPLBACKEND=Agg',
            'pip install --upgrade -I tensorflow',
            'pip install git+https://github.com/tflearn/tflearn.git',
            'pip install dominate',
            # # 'pip install multiprocessing_on_dill',
            'pip install scikit-image',
            'conda install numpy -n rllab3 -y',
        ],
    )
