import argparse
import os.path as osp
import random
from multiprocessing import cpu_count

import tensorflow as tf

from rllab import config
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.zero_baseline import ZeroBaseline
from rllab.envs.normalized_env import normalize
from rllab.misc import logger
from rllab.misc.instrument import VariantGenerator
from rllab.misc.instrument import run_experiment_lite
from sandbox.carlos.point_env_randgoal import PointEnvRandGoal, StraightDemo
from sandbox.rocky.tf.algos.sensitive_lfd_trpo import SensitiveLfD_TRPO
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.policies.sens_lfd_minimal_gauss_mlp_policy import SensitiveLfdGaussianMLPPolicy
from sandbox.young_clgan.logging import HTMLReport
from sandbox.young_clgan.logging import format_dict

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
subnets = [
    'us-west-1a', 'us-west-1b', 'ap-southeast-1b', 'eu-west-1c', 'eu-west-1b'
]
ec2_instance = args.type if args.type else 'm4.large'
# configure instance
info = config.INSTANCE_TYPE_INFO[ec2_instance]
config.AWS_INSTANCE_TYPE = ec2_instance
config.AWS_SPOT_PRICE = str(info["price"])
n_parallel = int(info["vCPU"] / 2)  # make the default 4 if not using ec2
if args.ec2:
    mode = 'ec2'
elif args.local_docker:
    mode = 'local_docker'
    n_parallel = cpu_count() if not args.debug else 1
else:
    mode = 'local'
    n_parallel = cpu_count() if not args.debug else 1


vg = VariantGenerator()
# fast updates
vg.add('demo_batch_size', [10, 20])  # numb of traj!
vg.add('batch_size', [50])  # numb of traj!
vg.add('baseline', ['linear'])
vg.add('max_path_length', [100])
vg.add('num_grad_updates', [1, 10, 20])
vg.add('fast_learning_rate', [0.05])  # 0.08 is already too large, and even 0.05 breaks sometimes...
# meta
vg.add('use_meta', [True])  # if False it won't update the initial params from one itr to the next
vg.add('meta_itr', lambda use_meta: [50] if use_meta else [1])
vg.add('meta_step_size', [0.01])
vg.add('meta_batch_size', [40])  # 10 works but much less stable, 20 is fairly stable, 40 is more stable --> n_env!!!!
# env
vg.add('env_noise', [0, 0.1])
vg.add('tolerance', [0.1])
vg.add('seed', range(0, 20, 10))
EXPERIMENT_TYPE = osp.basename(__file__).split('.')[0]


def run_task(v):

    logger.log("Initializing report...")

    report = HTMLReport(osp.join(logger.get_snapshot_dir(), 'report.html'), images_per_row=5)
    report.add_header("{}".format(EXPERIMENT_TYPE))
    report.add_text(format_dict(v))
    report.save()

    env = TfEnv(normalize(PointEnvRandGoal(noise=v['env_noise'], tolerance=v['tolerance'])))
    policy = SensitiveLfdGaussianMLPPolicy(
        name="policy",
        env_spec=env.spec,
        grad_step_size=v['fast_learning_rate'],
        hidden_nonlinearity=tf.nn.relu,
        hidden_sizes=(100, 100),
    )

    demo_policy = StraightDemo(env_spec=env.spec)

    if v['baseline'] == 'zero':
        baseline = ZeroBaseline(env_spec=env.spec)
    elif 'linear' in v['baseline']:
        baseline = LinearFeatureBaseline(env_spec=env.spec)
    else:
        baseline = GaussianMLPBaseline(env_spec=env.spec)

    algo = SensitiveLfD_TRPO(
        env=env,
        policy=policy,
        demo_policy=demo_policy,
        baseline=baseline,
        batch_size=v['batch_size'],  # number of trajs for RL grad update
        demo_batch_size=v['demo_batch_size'],
        max_path_length=v['max_path_length'],
        meta_batch_size=v['meta_batch_size'],
        num_grad_updates=v['num_grad_updates'],
        n_itr=v['meta_itr'],
        use_meta=v['use_meta'],
        step_size=v['meta_step_size'],
        plot=False,
        report=report,
        preupdate_samples=True,
        log_all_grad_steps=True,  # todo: this is intense logging
    )

    algo.train()


exp_prefix = 'lfd_mesh_grad_demos'
print("\n" + "**********" * 10 + "\nexp_prefix: {}\nvariants: {}".format(exp_prefix, vg.size))
print('Running on type {}, with price {}, parallel {} on the subnets: '.format(config.AWS_INSTANCE_TYPE,
                                                                               config.AWS_SPOT_PRICE, n_parallel),
      *subnets)
for vv in vg.variants():
    if mode in ['ec2', 'local_docker']:
        # choose subnet
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
            # use_cloudpickle=True,
            stub_method_call=run_task,
            variant=vv,
            mode=mode,
            n_parallel=4,
            snapshot_mode="last",
            seed=vv['seed'],
            exp_prefix=exp_prefix,
            # exp_name='tf1lfd_trposens' + str(int(use_sensitive)) + '_fbs' + str(fast_batch_size) + '_mbs' + str(
            #     meta_batch_size) + '_flr_' + str(fast_learning_rate) + 'metalr_' + str(
            #     meta_step_size) + '_env_noise' + str(env_noise) + '_step1' + str(num_grad_updates) + '_s' + str(s),
            plot=False,
            sync_s3_pkl=True,
            sync_s3_html=True,
            # # use this ONLY with ec2 or local_docker!!!
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
    else:
        run_experiment_lite(
            # use_cloudpickle=False,
            stub_method_call=run_task,
            variant=vv,
            mode=mode,
            n_parallel=4,
            snapshot_mode="last",
            seed=vv['seed'],
            exp_prefix='lfd_mesh_grad_demos',
            # exp_name='tf1lfd_trposens' + str(int(use_sensitive)) + '_fbs' + str(fast_batch_size) + '_mbs' + str(
            #     meta_batch_size) + '_flr_' + str(fast_learning_rate) + 'metalr_' + str(
            #     meta_step_size) + '_env_noise' + str(env_noise) + '_step1' + str(num_grad_updates) + '_s' + str(s),
            plot=False,
        )

