import matplotlib
matplotlib.use('Pdf')

import time
from rllab.algos.base import RLAlgorithm
import rllab.misc.logger as logger
import rllab.plotter as plotter
from sandbox.rocky.tf.policies.base import Policy
import tensorflow as tf
from sandbox.rocky.tf.samplers.batch_sampler import BatchSampler

from sandbox.rocky.tf.samplers.vectorized_sampler import VectorizedSampler
from sandbox.rocky.tf.samplers.vectorized_demo_sampler import VectorizedDemoSampler
import numpy as np
import os.path as osp
from rllab.misc import logger
from sandbox.young_clgan.logging import HTMLReport
from sandbox.young_clgan.logging import format_dict


class BatchSensitiveLfD_Polopt(RLAlgorithm):
    """
    Base class for batch sampling-based policy optimization methods, with sensitive learning.
    This includes various policy gradient methods like vpg, npg, ppo, trpo, etc.
    """

    def __init__(
            self,
            env,
            policy,
            baseline,
            scope=None,
            n_itr=500,
            start_itr=0,
            # Note that the number of trajectories for grad upate = batch_size
            # Defaults are 10 trajectories of length 500 for gradient update
            demo_batch_size=100,
            batch_size=100,  # test batch size, keeps original name such that it's the default for sampler
            max_path_length=500,
            meta_batch_size = 100,
            num_grad_updates=1,
            discount=0.99,
            gae_lambda=1,
            plot=False,
            pause_for_plot=False,
            center_adv=True,
            positive_adv=False,
            store_paths=False,
            whole_paths=True,
            fixed_horizon=False,
            sampler_cls=None,
            sampler_args=None,
            demo_sampler_cls=None,
            demo_sampler_args=None,
            demo_policy=None,
            force_batch_sampler=False,
            use_meta=True,
            load_policy=None,
            report=None,
            preupdate_samples=False,
            log_all_grad_steps=False,
            **kwargs
    ):
        """
        :param env: Environment
        :param policy: Policy
        :type policy: Policy
        :param baseline: Baseline
        :param scope: Scope for identifying the algorithm. Must be specified if running multiple algorithms
        simultaneously, each using different environments and policies
        :param n_itr: Number of iterations.
        :param start_itr: Starting iteration.
        :param demo_batch_size: Number of demo traj per iter.
        :param batch_size: Number of test traj per iter.  #
        :param max_path_length: Maximum length of a single rollout.
        :param meta_batch_size: Number of tasks sampled per meta-update
        :param num_grad_updates: Number of fast gradient updates
        :param discount: Discount.
        :param gae_lambda: Lambda used for generalized advantage estimation.
        :param plot: Plot evaluation run after each iteration.
        :param pause_for_plot: Whether to pause before contiuing when plotting.
        :param center_adv: Whether to rescale the advantages so that they have mean 0 and standard deviation 1.
        :param positive_adv: Whether to shift the advantages so that they are always positive. When used in
        conjunction with center_adv the advantages will be standardized before shifting.
        :param store_paths: Whether to save all paths data to the snapshot.
        :return:
        """
        self.env = env
        self.policy = policy
        self.load_policy=load_policy
        self.baseline = baseline
        self.scope = scope
        self.start_itr = start_itr
        self.demo_batch_size = demo_batch_size * max_path_length * meta_batch_size
        self.batch_size = batch_size * max_path_length * meta_batch_size
        self.max_path_length = max_path_length
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.plot = plot
        self.pause_for_plot = pause_for_plot
        self.center_adv = center_adv
        self.positive_adv = positive_adv
        self.store_paths = store_paths
        self.whole_paths = whole_paths
        self.fixed_horizon = fixed_horizon
        self.num_grad_updates = num_grad_updates # number of gradient steps during training
        self.use_meta = use_meta
        self.n_itr = n_itr
        self.meta_batch_size = meta_batch_size # number of tasks per batch
        self.report = report
        self.preupdate_samples = preupdate_samples
        self.log_all_grad_steps = log_all_grad_steps

        if sampler_cls is None:
            sampler_cls = VectorizedSampler
        if demo_sampler_cls is None:
            demo_sampler_cls = VectorizedDemoSampler
        if sampler_args is None:
            sampler_args = dict()
        if demo_sampler_args is None:
            demo_sampler_args = dict()
        sampler_args['n_envs'] = demo_sampler_args['n_envs'] = self.meta_batch_size
        sampler_args['batch_size'] = self.batch_size
        demo_sampler_args['batch_size'] = self.demo_batch_size
        self.sampler = sampler_cls(self, **sampler_args)
        if demo_policy is None:
            print("***Your demos are dumb!***")
            demo_policy = self.policy
        self.demo_sampler = demo_sampler_cls(self, policy=demo_policy, **demo_sampler_args)
        #self.init_opt()  # init_opt now happens in train()

    def start_worker(self):
        self.sampler.start_worker()
        self.demo_sampler.start_worker()
        if self.plot:
            plotter.init_plot(self.env, self.policy)

    def shutdown_worker(self):
        self.sampler.shutdown_worker()
        self.demo_sampler.shutdown_worker()

    def obtain_samples(self, itr, *reset_args, **reset_kwargs):
        # This obtains samples using self.policy, and calling policy.get_actions(obses)
        # return_dict specifies how the samples should be returned (dict separates samples
        # by task)
        paths = self.sampler.obtain_samples(itr, return_dict=True, *reset_args, **reset_kwargs)
        assert type(paths) == dict
        return paths

    def obtain_demo_samples(self, itr, *reset_args, **reset_kwargs):
        paths = self.demo_sampler.obtain_samples(itr, return_dict=True, *reset_args, **reset_kwargs)
        assert type(paths) == dict
        return paths

    def train(self):
        # TODO - make this a util
        flatten_list = lambda l: [item for sublist in l for item in sublist]

        with tf.Session() as sess:

            # Code for loading a previous policy. Somewhat hacky because needs to be in sess.
            if self.load_policy is not None:
                import joblib
                self.policy = joblib.load(self.load_policy)['policy']
            logger.log("initializing the comutation graph")
            self.init_opt()
            # initialize uninitialized vars  (only initialize vars that were not loaded)
            uninit_vars = []
            for var in tf.global_variables():
                # note - this is hacky, may be better way to do this in newer TF.
                try:
                    sess.run(var)
                except tf.errors.FailedPreconditionError:
                    uninit_vars.append(var)
            sess.run(tf.variables_initializer(uninit_vars))
            logger.log("about to start workers in the train method of batch_lfd_polopt")

            self.start_worker()
            start_time = time.time()
            for itr in range(self.start_itr, self.n_itr):
                itr_start_time = time.time()
                with logger.prefix('itr #%d | ' % itr):
                    # logger.log("clean reset of the env")
                    # self.env.reset(clean_reset=True)  # todo: need this?
                    self.start_worker()  # this re-instanciates all the envs! --> does it put to None all the goals?
                    # import pdb; pdb.set_trace()  # when it's instantiated, is it giving a None goal?
                    self.policy.switch_to_init_dist()  # Switch to pre-update policy

                    logger.log("Obtaining demos...")
                    unprocessed_demos = self.obtain_demo_samples(itr)
                    # for logging purpose only
                    self.demo_sampler.process_samples(itr, flatten_list(unprocessed_demos.values()), prefix='demo_', log=True)

                    if isinstance(self.sampler, VectorizedSampler):
                        learner_env_goals = [env.wrapped_env.wrapped_env.objective_params for env in self.demo_sampler.vec_env.envs]
                    else:
                        learner_env_goals = self.env.wrapped_env.wrapped_env.objective_params
                    logger.log("Processing demo samples...")  # needed to split the paths by obs/act/...
                    sampled_demos = {}
                    for key in unprocessed_demos.keys(): # the keys are the tasks. Process each of them sequentially
                        # don't log because this will spam the consol with every task.
                        sampled_demos[key] = self.demo_sampler.process_samples(itr, unprocessed_demos[key], log=False)

                    unprocessed_preupdate_paths = None
                    unprocessed_all_paths = None
                    if self.preupdate_samples:
                        logger.log("Obtaining preupdate samples...")
                        unprocessed_preupdate_paths = self.obtain_samples(itr, objective_params=learner_env_goals)
                        # for logging purpose only
                        self.sampler.process_samples(itr, flatten_list(unprocessed_preupdate_paths.values()), prefix='preUpdate_', log=True)
                        if self.log_all_grad_steps:
                            unprocessed_all_paths = [unprocessed_preupdate_paths]

                    for step in range(self.num_grad_updates):
                        logger.log('** Step ' + str(step) + ' **')
                        if step < self.num_grad_updates:
                            logger.log("Computing policy updates...")
                            self.policy.compute_updated_lfd_dists(sampled_demos, step + 1)  # todo: this should log loss. HERE!!
                            if self.log_all_grad_steps:
                                logger.log("Obtaining post %i update samples..." % (step + 1))
                                new_unprocessed_preupdate_paths = self.obtain_samples(itr, objective_params=learner_env_goals)
                                # for logging purpose only
                                self.sampler.process_samples(itr, flatten_list(new_unprocessed_preupdate_paths.values()), prefix='post%iUpdate_'%(step+1), log=True)
                                unprocessed_all_paths.append(new_unprocessed_preupdate_paths)

                    logger.log("Obtaining final samples...")
                    unprocessed_paths = self.obtain_samples(itr, objective_params=learner_env_goals)
                    # for logging purpose only
                    self.sampler.process_samples(itr, flatten_list(unprocessed_paths.values()), prefix='postUpdate_', log=True)

                    sampled_data = {}
                    for key in unprocessed_paths.keys():  # the keys are the tasks. Process each of them sequentially
                        # don't log because this will spam the consol with every task.
                        sampled_data[key] = self.sampler.process_samples(itr, unprocessed_paths[key], log=False)

                    if self.use_meta:
                        logger.log("Optimizing policy...")
                        # This needs to take all samples_data so that it can construct graph for meta-optimization.
                        self.optimize_policy(itr, [sampled_demos, sampled_data])

                    logger.log("Logging diagnostics...")  # assumes list of dicts of list of paths
                    self.env.log_diagnostics(unprocessed_paths, plot=True, demos=unprocessed_demos,
                                             preupdate_paths=unprocessed_preupdate_paths,
                                             all_updates_paths=unprocessed_all_paths, prefix='',
                                             report=self.report, policy=self.policy,
                                             goals=learner_env_goals)

                    logger.log("Saving snapshot...")
                    params = self.get_itr_snapshot(itr, sampled_data)  # , **kwargs)
                    if self.store_paths:
                        params["paths"] = sampled_data["paths"]
                    logger.save_itr_params(itr, params)
                    logger.log("Saved")
                    logger.record_tabular('Time', time.time() - start_time)
                    logger.record_tabular('ItrTime', time.time() - itr_start_time)

                    # Below commented out code is useful for visualizing trajectories across a few different tasks.
                    """
                    if itr % 2 == 0 and self.env.observation_space.shape[0] <= 4: # point-mass
                        logger.log("Saving visualization of paths")
                        import matplotlib.pyplot as plt;
                        for ind in range(min(5, self.meta_batch_size)):
                            plt.clf()
                            plt.plot(learner_env_goals[ind][0], learner_env_goals[ind][1], 'k*', markersize=10)
                            plt.hold(True)

                            preupdate_paths = all_paths[0]
                            postupdate_paths = all_paths[-1]

                            pre_points = preupdate_paths[ind][0]['observations']
                            post_points = postupdate_paths[ind][0]['observations']
                            plt.plot(pre_points[:,0], pre_points[:,1], '-r', linewidth=2)
                            plt.plot(post_points[:,0], post_points[:,1], '-b', linewidth=1)

                            pre_points = preupdate_paths[ind][1]['observations']
                            post_points = postupdate_paths[ind][1]['observations']
                            plt.plot(pre_points[:,0], pre_points[:,1], '--r', linewidth=2)
                            plt.plot(post_points[:,0], post_points[:,1], '--b', linewidth=1)

                            pre_points = preupdate_paths[ind][2]['observations']
                            post_points = postupdate_paths[ind][2]['observations']
                            plt.plot(pre_points[:,0], pre_points[:,1], '-.r', linewidth=2)
                            plt.plot(post_points[:,0], post_points[:,1], '-.b', linewidth=1)

                            plt.plot(0,0, 'k.', markersize=5)
                            plt.xlim([-0.8, 0.8])
                            plt.ylim([-0.8, 0.8])
                            plt.legend(['goal', 'preupdate path', 'postupdate path'])
                            plt.savefig('/home/cfinn/prepost_path'+str(ind)+'.png')
                    elif itr % 2 == 0:  # swimmer or cheetah
                        logger.log("Saving visualization of paths")
                        import matplotlib.pyplot as plt;
                        for ind in range(min(5, self.meta_batch_size)):
                            plt.clf()
                            goal_vel = learner_env_goals[ind]
                            plt.title('Swimmer paths, goal vel='+str(goal_vel))
                            plt.hold(True)


                            prepathobs = all_paths[0][ind][0]['observations']
                            postpathobs = all_paths[-1][ind][0]['observations']
                            plt.plot(prepathobs[:,0], prepathobs[:,1], '-r', linewidth=2)
                            plt.plot(postpathobs[:,0], postpathobs[:,1], '--b', linewidth=1)
                            plt.plot(prepathobs[-1,0], prepathobs[-1,1], 'r*', markersize=10)
                            plt.plot(postpathobs[-1,0], postpathobs[-1,1], 'b*', markersize=10)
                            plt.xlim([-1.0, 5.0])
                            plt.ylim([-1.0, 1.0])

                            plt.legend(['preupdate path', 'postupdate path'], loc=2)
                            plt.savefig('/home/cfinn/swim1d_prepost_itr'+str(itr)+'_id'+str(ind)+'.pdf')
                    """

                    logger.dump_tabular(with_prefix=False)
                    #if self.plot:
                    #    self.update_plot()
                    #    if self.pause_for_plot:
                    #        input("Plotting evaluation run: Press Enter to "
                    #              "continue...")
        self.shutdown_worker()

    def log_diagnostics(self, paths, prefix):  # assume paths is a list (num_steps) of dicts (n_envs)
        """Does not call the log_diagnostics of the env (because could need demos)"""
        self.policy.log_diagnostics(paths, prefix)
        self.baseline.log_diagnostics(paths)

    def init_opt(self):
        """
        Initialize the optimization procedure. If using tensorflow, this may
        include declaring all the variables and compiling functions
        """
        raise NotImplementedError

    def get_itr_snapshot(self, itr, samples_data):
        """
        Returns all the data that should be saved in the snapshot for this
        iteration.
        """
        raise NotImplementedError

    def optimize_policy(self, itr, samples_data):
        raise NotImplementedError

    def update_plot(self):
        if self.plot:
            plotter.update_plot(self.policy, self.max_path_length)
