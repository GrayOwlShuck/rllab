from rllab.misc import ext
from rllab.misc.overrides import overrides
import rllab.misc.logger as logger
from sandbox.rocky.tf.algos.batch_sensitive_lfd_polopt import BatchSensitiveLfD_Polopt
from sandbox.rocky.tf.optimizers.penalty_lbfgs_optimizer import PenaltyLbfgsOptimizer
from sandbox.rocky.tf.optimizers.first_order_optimizer import FirstOrderOptimizer
from sandbox.rocky.tf.misc import tensor_utils
import tensorflow as tf


class SensitiveLfD_NPO(BatchSensitiveLfD_Polopt):
    """
    Natural Policy Optimization.
    """

    def __init__(
            self,
            optimizer=None,
            optimizer_args=None,
            step_size=0.01,
            use_sensitive=True,
            **kwargs):
        assert optimizer is not None  # only for use with Sensitive TRPO
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict()
            optimizer = PenaltyLbfgsOptimizer(**optimizer_args)
        if not use_sensitive:
            default_args = dict(
                batch_size=None,
                max_epochs=1,
            )
            optimizer = FirstOrderOptimizer(**default_args)
        self.optimizer = optimizer
        self.step_size = step_size
        self.use_sensitive = use_sensitive
        self.kl_constrain_step = -1  # needs to be 0 or -1 (original pol params, or new pol params)
        super(SensitiveLfD_NPO, self).__init__(**kwargs)

    def make_vars(self, stepnum='0'):
        # lists over the meta_batch_size
        obs_vars, action_vars, adv_vars = [], [], []
        for i in range(self.meta_batch_size):
            obs_vars.append(self.env.observation_space.new_tensor_variable(
                'obs' + stepnum + '_' + str(i),
                extra_dims=1,
            ))
            action_vars.append(self.env.action_space.new_tensor_variable(
                'action' + stepnum + '_' + str(i),
                extra_dims=1,
            ))
            adv_vars.append(tensor_utils.new_tensor(
                name='advantage' + stepnum + '_' + str(i),
                ndim=1, dtype=tf.float32,
            ))
        return obs_vars, action_vars, adv_vars

    def make_demo_vars(self, stepnum='0'):
        # lists over the meta_batch_size
        obs_demo_vars, action_demo_vars = [], []
        for i in range(self.meta_batch_size):
            obs_demo_vars.append(self.env.observation_space.new_tensor_variable(
                'obs' + stepnum + '_' + str(i),
                extra_dims=1,
                ))
            action_demo_vars.append(self.env.action_space.new_tensor_variable(
                'action' + stepnum + '_' + str(i),
                extra_dims=1,
                ))
        return obs_demo_vars, action_demo_vars

    @overrides
    def init_opt(self):
        is_recurrent = int(self.policy.recurrent)
        assert not is_recurrent  # not supported

        dist = self.policy.distribution

        old_dist_info_demo_vars, old_dist_info_vars_demo_list = [], []
        for i in range(self.meta_batch_size):
            old_dist_info_demo_vars.append({
                k: tf.placeholder(tf.float32, shape=[None] + list(shape), name='old_%s_%s' % (i, k))
                for k, shape in dist.dist_info_specs
                })
            old_dist_info_vars_demo_list += [old_dist_info_demo_vars[i][k] for k in dist.dist_info_keys]

        state_info_vars, state_info_vars_list = {}, []

        all_surr_objs, input_list = [], []
        new_params = None

        # add demo vars
        obs_demo_vars, action_demo_vars = self.make_demo_vars()  # we are only given demos once
        input_list += obs_demo_vars + action_demo_vars

        for j in range(self.num_grad_updates + 1):  # j=0 does not do any update.
            # obs_vars, action_vars, adv_vars = self.make_vars(str(j))
            surr_objs = []

            cur_params = new_params
            new_params = []  # if there are several grad_updates the new_params are overwritten, but they already served their purpose
            kls = []

            # build the graph of the updates
            for i in range(self.meta_batch_size):  # these are the number of different envs used (inner update independent)
                if j == 0:
                    # dist_info_vars, params = self.policy.dist_info_sym(obs_vars[i], state_info_vars, all_params=self.policy.all_params)
                    dist_info_demo_vars, params = self.policy.dist_info_sym(obs_demo_vars[i], all_params=self.policy.all_params)  # todo: check this makes copy of params!
                    if self.kl_constrain_step == 0:
                        kl = dist.kl_sym(old_dist_info_demo_vars[i], dist_info_demo_vars)
                        kls.append(kl)
                else:
                    dist_info_demo_vars, params = self.policy.updated_dist_info_lfd_sym(i, all_surr_objs[-1][i], obs_demo_vars[i], params_dict=cur_params[i])

                new_params.append(params)
                logli_demo = dist.log_likelihood_sym(action_demo_vars[i], dist_info_demo_vars)

                surr_objs.append(- tf.reduce_mean(logli_demo))

            # input_list += obs_vars + action_vars + adv_vars + state_info_vars_list
            if j == 0:
                # For computing the fast update for sampling  # CF ???
                self.policy.set_init_surr_obj(input_list, surr_objs)
                init_input_list = input_list

            all_surr_objs.append(surr_objs)

        # compute the surrogate obj based on the LAST update. This is over what we take the outer meta updates.
        obs_test_vars, action_test_vars, adv_test_vars = self.make_vars('test')
        old_dist_info_test_vars, old_dist_info_vars_test_list = [], []   # this will also be given once new paths are collected
        for i in range(self.meta_batch_size):
            old_dist_info_test_vars.append({
                k: tf.placeholder(tf.float32, shape=[None] + list(shape), name='old_%s_%s' % (i, k))
                for k, shape in dist.dist_info_specs
            })
            old_dist_info_vars_test_list += [old_dist_info_test_vars[i][k] for k in dist.dist_info_keys]

        surr_objs = []
        kls_test = []

        for i in range(self.meta_batch_size):
            # these are the vars that cary all the updates and will allow the optimization based on last step!
            dist_info_test_vars, _ = self.policy.dist_info_sym(obs_test_vars[i], all_params=new_params[i])  # does this make a copy ??
            kl_test = dist.kl_sym(old_dist_info_test_vars[i], dist_info_test_vars)
            kls_test.append(kl_test)

            lr = dist.likelihood_ratio_sym(action_test_vars[i], old_dist_info_test_vars[i], dist_info_test_vars)
            surr_objs.append(- tf.reduce_mean(lr*adv_test_vars[i]))

        if self.use_sensitive:
            surr_obj = tf.reduce_mean(tf.stack(surr_objs, 0))  # mean over meta_batch_size (the diff tasks)
            input_list += obs_test_vars + action_test_vars + adv_test_vars + old_dist_info_vars_test_list
        else:
            surr_obj = tf.reduce_mean(tf.stack(all_surr_objs[0], 0))  # if not meta, just use the first surr_obj
            input_list = init_input_list

        if self.use_sensitive:
            mean_test_kl = tf.reduce_mean(tf.concat(kls_test, 0))
            max_test_kl = tf.reduce_max(tf.concat(kls_test, 0))

            self.optimizer.update_opt(
                loss=surr_obj,
                target=self.policy,
                leq_constraint=(mean_test_kl, self.step_size),
                inputs=input_list,
                constraint_name="mean_kl"
            )
        else:
            self.optimizer.update_opt(
                loss=surr_obj,
                target=self.policy,
                inputs=input_list,
            )
        return dict()

    @overrides
    def optimize_policy(self, itr, sampled_data):
        sampled_demos, sampled_data = sampled_data

        if not self.use_sensitive:
            sampled_data = [sampled_data[0]]

        input_list = []
        test_obs_list, test_action_list, test_adv_list = [], [], []
        demo_obs_list, demo_action_list = [], []

        for i in range(self.meta_batch_size):

            test_inputs = ext.extract(
                sampled_data[i],
                "observations", "actions", "advantages"
            )
            test_obs_list.append(test_inputs[0])
            test_action_list.append(test_inputs[1])
            test_adv_list.append(test_inputs[2])

            demo_inputs = ext.extract(
                sampled_demos[i],
                "observations", "actions"
            )
            demo_obs_list.append(demo_inputs[0])
            demo_action_list.append(demo_inputs[1])

        input_list += demo_obs_list + demo_action_list + test_obs_list + test_action_list + test_adv_list

        if self.use_sensitive:
            dist_info_list = []
            for i in range(self.meta_batch_size):
                agent_infos = sampled_data[i]['agent_infos']
                dist_info_list += [agent_infos[k] for k in self.policy.distribution.dist_info_keys]
            input_list += tuple(dist_info_list)
            logger.log("Computing KL before")
            mean_kl_before = self.optimizer.constraint_val(input_list)

        logger.log("Computing loss before")
        loss_before = self.optimizer.loss(input_list)
        logger.log("Optimizing")
        self.optimizer.optimize(input_list)
        logger.log("Computing loss after")
        loss_after = self.optimizer.loss(input_list)
        if self.use_sensitive:
            logger.log("Computing KL after")
            mean_kl = self.optimizer.constraint_val(input_list)
            logger.record_tabular('MeanKLBefore', mean_kl_before)  # this now won't be 0!
            logger.record_tabular('MeanKL', mean_kl)
        logger.record_tabular('LossBefore', loss_before)
        logger.record_tabular('LossAfter', loss_after)
        logger.record_tabular('dLoss', loss_before - loss_after)
        return dict()

    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            policy=self.policy,
            baseline=self.baseline,
            env=self.env,
        )
