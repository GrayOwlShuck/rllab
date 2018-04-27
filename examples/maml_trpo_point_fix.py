from sandbox.rocky.tf.algos.sensitive_trpo import SensitiveTRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.baselines.zero_baseline import ZeroBaseline
from examples.point_env_rand2goal import PointEnvRandGoal
from examples.point_env_randgoal_oracle import PointEnvRandGoalOracle
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.misc import logger
# from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.policies.sens_minimal_gauss_mlp_policy import SensitiveGaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.misc.instrument import VariantGenerator

import tensorflow as tf

vg = VariantGenerator()
# fast updates

vg.add('batch_size', [50])  # numb of traj!
vg.add('baseline', ['linear'])
vg.add('max_path_length', [100])
vg.add('num_grad_updates', [1, 5, 10, 20])
vg.add('fast_learning_rate', [0.5])  # 0.08 is already too large, and even 0.05 breaks sometimes...
# meta
vg.add('use_meta', [True])  # if False it won't update the initial params from one itr to the next
vg.add('meta_itr', lambda use_meta: [50] if use_meta else [1])
vg.add('meta_step_size', [0.01])
vg.add('meta_batch_size', [40])  # 10 works but much less stable, 20 is fairly stable, 40 is more stable --> n_env!!!!
# env
vg.add('env_noise', [0, 0.1])
vg.add('tolerance', [0.1])
vg.add('seed', range(0, 20, 10))
# 1e-3 for sensitive, 1e-2 for oracle, non-sensitive [is this still true?]
learning_rates = [1e-2]  # 1e-3 works well for 1 step, trying lower for 2 step, trying 1e-2 for large batch


def run_task(v):

    env = TfEnv(normalize(PointEnvRandGoal()))
    policy = SensitiveGaussianMLPPolicy(
        name="policy",
        env_spec=env.spec,
        grad_step_size=v['fast_learning_rate'],
        hidden_nonlinearity=tf.nn.relu,
        hidden_sizes=(100, 100),
    )
    if v['baseline'] == 'zero':
        baseline = ZeroBaseline(env_spec=env.spec)
    elif 'linear' in v['baseline']:
        baseline = LinearFeatureBaseline(env_spec=env.spec)
    else:
        baseline = GaussianMLPBaseline(env_spec=env.spec)

    algo = SensitiveTRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=v['batch_size'],  # number of trajs for grad update
        max_path_length=v['max_path_length'],
        meta_batch_size=v['meta_batch_size'],
        num_grad_updates=v['num_grad_updates'],
        n_itr=v['meta_itr'],
        use_sensitive=v['use_meta'],
        step_size=v['meta_step_size'],
        plot=False,
    )

    algo.train()


for vv in vg.variants():

    run_experiment_lite(
        stub_method_call=run_task,
        variant=vv,
        n_parallel=4,
        snapshot_mode="last",
        seed=vv['seed'],
        exp_prefix='vpg_sensitive_point100',
        plot=False,
    )
