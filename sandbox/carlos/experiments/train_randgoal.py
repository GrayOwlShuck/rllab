from rllab.misc.instrument import stub, run_experiment_lite
import argparse
import sys
import time

import os.path as osp
import random
import tensorflow as tf
from rllab import config
from rllab.misc import logger
from rllab.misc.instrument import VariantGenerator
from rllab.envs.normalized_env import normalize
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.baselines.zero_baseline import ZeroBaseline
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.algos.vpg import VPG

from sandbox.young_clgan.logging import HTMLReport
from sandbox.young_clgan.logging import format_dict
from sandbox.young_clgan.logging.logger import ExperimentLogger

from sandbox.carlos.point_env_randgoal import PointEnvRandGoal

parser = argparse.ArgumentParser()
parser.add_argument('--ec2', '-e', action='store_true', default=False, help="add flag to run in ec2")
parser.add_argument('--local_docker', '-d', action='store_true', default=False,
                    help="add flag to run in local dock")
parser.add_argument('--type', '-t', type=str, default='', help='set instance type')
parser.add_argument('--price', '-p', type=str, default='', help='set betting price')
parser.add_argument('--subnet', '-sn', type=str, default='', help='set subnet like us-west-1a')
parser.add_argument('--name', '-n', type=str, default='', help='set exp prefix name and new file name')
parser.add_argument('--debug', action='store_true', default=False, help="run code without multiprocessing")
args = parser.parse_args()


# setup ec2
subnets = None
n_parallel = 0
# subnets = [
#     'us-west-1a', 'us-west-1b', 'ap-southeast-1b', 'eu-west-1c', 'eu-west-1b'
# ]
ec2_instance = args.type if args.type else 'c4.4xlarge'
# configure instance
info = config.INSTANCE_TYPE_INFO[ec2_instance]
config.AWS_INSTANCE_TYPE = ec2_instance
config.AWS_SPOT_PRICE = str(info["price"])
if args.ec2:
    mode = 'ec2'
elif args.local_docker:
    mode = 'local_docker'
else:
    mode = 'local'

exp_prefix = 'vpg-point-randgoal'
vg = VariantGenerator()
# for algo
vg.add('fix_batch_obj', [True, False])
vg.add('inner_batch', [20, 50])  # number of rollouts to estimate every grad update
vg.add('inner_iters', lambda fix_batch_obj: [1] if not fix_batch_obj else [1, 3])   # number of grad updates on SAME task
vg.add('learning_rate', [0.001, 0.01])  # for inner algorithm
vg.add('baseline', ['zero'])

# for outer updates
vg.add('n_itr', [1000])
vg.add('eval_itr', [5])  # equivalent to inner_iters in Reptile (although here only for finetune)
vg.add('eval_interval', [10])

# env
vg.add('obs_append_objective', [False])
vg.add('sparse_reward', [True])
vg.add('env_noise', [0])
vg.add('tolerance', [0.5])
vg.add('max_path_length', [1000])

vg.add('seed', range(0, 30, 10))

EXPERIMENT_TYPE = osp.basename(__file__).split('.')[0]


def main(vv):
    """
    Load data and train a model on it.
    """
    random.seed(vv['seed'])

    logger.log("Initializing report...")
    log_dir = logger.get_snapshot_dir()  # problem with logger module here!!
    exp_name = osp.split(log_dir)[-1]
    report = HTMLReport(osp.join(log_dir, 'report.html'), images_per_row=5)
    report.add_header("{}: {}".format(EXPERIMENT_TYPE, exp_name))
    report.add_text(format_dict(v))
    report.save()

    env = TfEnv(normalize(PointEnvRandGoal(tolerance=v['tolerance'],
                                           obs_append_objective=v['obs_append_objective'],
                                           sparse_reward=v['sparse_reward'])))
    policy = GaussianMLPPolicy('policy', env_spec=env.spec)

    baseline = ZeroBaseline(env_spec=env.spec)
    if v['baseline'] is 'mlp':
        # baseline = LinearFeatureBaseline(env_spec=env.spec)
        baseline = GaussianMLPBaseline(env_spec=env.spec)

    algo = VPG(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=v['inner_batch'] * v['max_path_length'],  # adjust for the vectorized sampler...
        max_path_length=v['max_path_length'],
        n_itr=v['inner_iters'],
        optimizer_args={'init_learning_rate': v['learning_rate']},
    )

    print('Training...')
    with tf.Session() as sess:
        algo.initialize_variables(sess)
        time_iter = time.time()
        train_paths = []
        for i in range(v['n_itr']):
            print("********* itr: {} ***********".format(i))
            with ExperimentLogger(log_dir, 'last', snapshot_mode='last', hold_outter_log=True, hold_inner_log=True):
                if v['fix_batch_obj']:
                    env.reset(clean_reset=True)
                    objective_params = env.objective_params
                    paths = algo.train(sess=sess, force_expand=True, objective_params=objective_params)
                else:
                    paths = algo.train(sess=sess, clean_reset=True)
            train_paths.append(paths[-1])  # only log_diag on paths from last itr of algo.train ("adapted")

            if i % v['eval_interval'] == 0:
                # run additional log_diag on fixed goals
                old_param_vals = policy.get_param_values()
                old_n_itr = algo.n_itr
                algo.n_itr = v['eval_itr']
                with ExperimentLogger(log_dir, 'eval', snapshot_mode='last', hold_outter_log=True, hold_inner_log=True):
                    all_preupdate_paths = {}
                    all_postupdate_paths = {}
                    for k, test_obj in enumerate(env.test_objective_params):
                        policy.set_param_values(old_param_vals)
                        new_paths = algo.train(sess, force_expand=True, objective_params=test_obj)
                        all_preupdate_paths[k] = new_paths[0]
                        all_postupdate_paths[k] = new_paths[-1]

                algo.env.log_diagnostics_2(postupdate_paths=all_postupdate_paths, preupdate_paths=all_preupdate_paths,
                                           report=report)
                logger.record_tabular('TimeIterations', time.time() - time_iter)
                time_iter = time.time()
                # log_diag on train_paths and empty them
                algo.log_diagnostics([p for paths in train_paths for p in paths])
                train_paths = []

                params = algo.get_itr_snapshot(i, samples_data=None)  # , **kwargs)
                if algo.store_paths:
                    params["paths"] = [all_preupdate_paths, all_postupdate_paths]
                logger.save_itr_params(i, params)
                logger.dump_tabular(with_prefix=False)
                # make as if we hadn't run the eval steps
                policy.set_param_values(old_param_vals)
                algo.n_itr = old_n_itr


print("\n" + "**********" * 10 + "\nexp_prefix: {}\nvariants: {}".format(exp_prefix, vg.size))
print('Running on type {}, with price {}, parallel {} on the subnets: '.format(config.AWS_INSTANCE_TYPE,
                                                                               config.AWS_SPOT_PRICE, n_parallel),
      subnets)
for v in vg.variants():
    # choose subnet
    if subnets is not None:
        subnet = random.choice(subnets)
        config.AWS_REGION_NAME = subnet[:-1]
        config.AWS_KEY_NAME = config.ALL_REGION_AWS_KEY_NAMES[
            config.AWS_REGION_NAME]
        config.AWS_IMAGE_ID = config.ALL_REGION_AWS_IMAGE_IDS[
            config.AWS_REGION_NAME]
        config.AWS_SECURITY_GROUP_IDS = \
            config.ALL_REGION_AWS_SECURITY_GROUP_IDS[
                config.AWS_REGION_NAME]
        config.AWS_NETWORK_INTERFACES = [
            dict(
                SubnetId=config.ALL_SUBNET_INFO[subnet]["SubnetID"],
                Groups=config.AWS_SECURITY_GROUP_IDS,
                DeviceIndex=0,
                AssociatePublicIpAddress=True,
            )
        ]

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
        mode=mode,
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
    if mode in ['local', 'local_docker']:
        sys.exit()
