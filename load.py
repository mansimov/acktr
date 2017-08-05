import gym
from utils import *
import utils
from filters import ZFilter, IdentityFilter, ClipFilter
from normalized_env import NormalizedEnv # used only for rescaling actions
from rgb_env_load import RGBEnv
import numpy as np
import random
import tensorflow as tf
import time
import os
import logging
import tempfile
import sys
import argparse
import kfac
import shutil
import pickle

parser = argparse.ArgumentParser(description="Run commands")
# GENERAL HYPERPARAMETERS
parser.add_argument('-e', '--env-id', type=str, default="Pendulum-v0",
                    help="Environment id")
parser.add_argument('-mt', '--max-timesteps', default=100000000, type=int,
                    help="Maximum number of timesteps")
parser.add_argument('-tpb', '--timesteps-per-batch', default=1000, type=int,
                    help="Minibatch size")
parser.add_argument('-g', '--gamma', default=0.99, type=float,
                    help="Discount Factor")
parser.add_argument('-l', '--lam', default=0.97, type=float,
                    help="Lambda value to reduce variance see GAE")
parser.add_argument('-s', '--seed', default=1, type=int,
                    help="Seed")
parser.add_argument('--log-dir', default="/tmp/cont_control/unknown", type=str,
                    help="Folder to save")
# NEURAL NETWORK ARCHITECTURE
parser.add_argument('--weight-decay-fc', default=3e-4, type=float, help="weight decay for fc layer")
parser.add_argument('--weight-decay-conv', default=4e-3, type=float, help="weight decay for conv layer")
parser.add_argument('--use-pixels', default=False, type=bool, help="use rgb instead of low dim state rep")
# GENERAL KFAC arguments
parser.add_argument('--async-kfac', default=True, type=bool, help="use async version")
# POLICY HYPERPARAMETERS
parser.add_argument('--use-adam', default=False, type=bool, help="use adam for actor")
parser.add_argument('--use-sgd', default=False, type=bool, help="use sgd with momentum for actor")
parser.add_argument('--adapt-lr', default=True, type=bool, help="adapt lr")
parser.add_argument('--upper-bound-kl', default=False, type=bool, help="upper bound kl")
parser.add_argument('--lr', default=0.03, type=float, help="Learning Rate")
parser.add_argument('--mom', default=0.9, type=float, help="Momentum")
parser.add_argument('--kl-desired', default=0.001, type=float, help="desired kl div")
parser.add_argument('--kfac-update', default=2, type=int,
                    help="Update Fisher Matrix every number of steps")
parser.add_argument('--cold-iter', default=1, type=int,
                    help="Number of cold iterations using sgd")
parser.add_argument('--epsilon', default=1e-2, type=float, help="Damping factor")
parser.add_argument('--stats-decay', default=0.99,type=float, help="decay running average of stats factor")
# VALUE FUNCTION HYPERPARAMETERS
parser.add_argument('--use-adam-vf', default=False, type=bool, help="use adam for vf")
parser.add_argument('--use-sgd-vf', default=False, type=bool, help="use sgd with momentum for vf")
parser.add_argument('--lr-vf', default=0.003, type=float, help="Learning Rate vf")
parser.add_argument('--cold-lr-vf', default=0.001, type=float, help="Learning Rate vf")
parser.add_argument('--mom-vf', default=0.9, type=float, help="Momentum")
parser.add_argument('--kl-desired-vf', default=0.3, type=float, help="desired kl div")
parser.add_argument('--deterministic', default=False, type=bool, help="deterministic sample")
parser.add_argument('--epsilon-vf', default=0.1, type=float, help="Damping factor")
parser.add_argument('--stats-decay-vf', default=0.95, type=float, help="Damping factor")
parser.add_argument('--kfac-update-vf', default=2, type=int,
                    help="Update Fisher Matrix every number of steps")
parser.add_argument('--cold-iter-vf', default=50, type=int,
                    help="Number of cold iterations using sgd")
parser.add_argument('--train-iter-vf', default=25, type=int,
                    help="Number of cold iterations using sgd")
parser.add_argument('--moving-average-vf', default=0.0, type=float,
                    help="Moving average of VF parameters")
parser.add_argument('--load-model', default=False, type=bool,
                    help="Load trained model")
parser.add_argument('--load-dir', default="/tmp/cont_control/unknown", type=str,
                    help="Folder to load from")
parser.add_argument('--save', default=False, type=bool,
                    help="Save results")
parser.add_argument('--save-dir', default="./dummy/", type=str,
                    help="Save directory")

class AsyncNGAgent(object):

    def __init__(self, env, args):
        self.env = env
        self.config = config = args
        self.config.max_pathlength = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps') or 1000
        # set weight decay for fc and conv layers
        utils.weight_decay_fc = self.config.weight_decay_fc
        utils.weight_decay_conv = self.config.weight_decay_conv

        # hardcoded for now
        if self.config.use_adam:
            self.config.kl_desired = 0.002
            self.lr = 1e-4

        # print all the flags
        print '##################'
        # save hyperparams to txt file
        hyperparams_txt = ""
        for key,value in vars(self.config).iteritems():
            print key, value
            hyperparams_txt = hyperparams_txt + "{} {}\n".format(key, value)
        if os.path.exists(self.config.log_dir):
            shutil.rmtree(self.config.log_dir)
        os.mkdir(self.config.log_dir)
        txt_file = open(os.path.join(self.config.log_dir, "hyperparams.txt"), "w")
        txt_file.write(hyperparams_txt)
        txt_file.close()
        print (self.config.log_dir)
        print '##################'
        print("Observation Space", env.observation_space)
        print("Action Space", env.action_space)
        config_tf = tf.ConfigProto()
        config_tf.gpu_options.allow_growth=True # don't take full gpu memory
        self.session = tf.Session(config=config_tf)
        self.train = True
        self.solved = False
        self.obs_shape = obs_shape = list(env.observation_space.shape)
        self.prev_obs = np.zeros([1] + list(obs_shape))
        self.prev_action = np.zeros((1, env.action_space.shape[0]))
        obs_shape[-1] *= 2 # include previous frame in a state
        if self.config.use_pixels:
            self.obs = obs = tf.placeholder(
                dtype, shape=[None] + obs_shape, name="obs")
        else:
            self.obs = obs = tf.placeholder(
                dtype, shape=[None, 2*env.observation_space.shape[0] + env.action_space.shape[0]], name="obs")

        self.action = action = tf.placeholder(dtype, shape=[None, env.action_space.shape[0]], name="action")
        self.advant = advant = tf.placeholder(dtype, shape=[None], name="advant")
        self.oldaction_dist = oldaction_dist = tf.placeholder(dtype, shape=[None, env.action_space.shape[0]*2], name="oldaction_dist")

        if self.config.use_pixels:
            self.ob_filter = IdentityFilter()
            self.reward_filter = ZFilter((1,), demean=False, clip=10)
        else:
            self.ob_filter = ZFilter((env.observation_space.shape[0],), clip=5)
            self.reward_filter = ZFilter((1,), demean=False, clip=10)

        # Create summary writer
        self.summary_writer = tf.summary.FileWriter(self.config.log_dir)

    def init_policy_train_op(self, loss_policy, loss_policy_sampled, wd_dict):
          if self.config.use_adam:
                self.stepsize = tf.Variable(np.float32(np.array(1e-4)), dtype=tf.float32)
                self.updates = tf.train.AdamOptimizer(self.stepsize).minimize(loss_policy)
                self.queue_runner = None
          elif self.config.use_sgd:
                self.stepsize = tf.Variable(np.float32(np.array(self.config.lr)), dtype=tf.float32)
                self.updates = tf.train.MomentumOptimizer(self.stepsize*(1.-self.config.mom), self.config.mom).minimize(loss_policy)
                self.queue_runner = None
          else:
                self.stepsize = tf.Variable(np.float32(np.array(self.config.lr)), dtype=tf.float32)
                self.updates, self.queue_runner = kfac.KfacOptimizer(
                                                 learning_rate=self.stepsize,
                                                 cold_lr=self.stepsize/3.,
                                                 momentum=self.config.mom,
                                                 clip_kl=self.config.kl_desired,
                                                 upper_bound_kl=self.config.upper_bound_kl,
                                                 epsilon=self.config.epsilon,
                                                 stats_decay=self.config.stats_decay,
                                                 async=self.config.async_kfac,
                                                 kfac_update = self.config.kfac_update,
                                                 cold_iter=self.config.cold_iter,
                                                 weight_decay_dict= wd_dict).minimize(
                                                      loss_policy,
                                                      loss_policy_sampled,
                                                      self.policy_var_list)

          return self.updates, self.queue_runner

    # Function that creates computational graph for actor
    def init_policy(self):
        # Create neural network
        if self.config.use_pixels:
            action_dist_n, self.policy_weight_decay_dict = create_policy_net_rgb(self.obs, env.action_space.shape[0])
        else:
            action_dist_n, self.policy_weight_decay_dict = create_policy_net(self.obs, [64,64], [True, True], env.action_space.shape[0])

        # get weight decay losses for actor
        policy_losses = tf.get_collection('policy_losses', None)
        eps = 1e-6
        self.action_dist_n = action_dist_n
        N = tf.shape(self.obs)[0]
        Nf = tf.cast(N, dtype)

        logp_n = loglik(self.action, action_dist_n, env.action_space.shape[0])
        oldlogp_n = loglik(self.action, self.oldaction_dist, env.action_space.shape[0])

        self.surr = surr = -tf.reduce_mean(tf.exp(logp_n - oldlogp_n) * self.advant)
        self.surr_fisher = surr_fisher = -tf.reduce_mean(tf.exp(logp_n - oldlogp_n))

        self.kl = kl = tf.reduce_mean(kl_div(self.oldaction_dist, action_dist_n, env.action_space.shape[0]))

        # var_list should only contain actor's variables
        self.policy_var_list = tf.trainable_variables()
        for var in self.policy_var_list:
            if "policy" not in var.name:
                self.policy_var_list.remove(var)

        ## weight decay
        self.total_policy_loss = surr + tf.add_n(policy_losses)

        return self.total_policy_loss, self.surr_fisher, self.policy_weight_decay_dict

    def act(self, obs, *args):
        if self.config.use_pixels == False:
            obs = self.ob_filter(obs, update=self.train)
        else:
            obs = self.ob_filter(obs)

        obs = np.expand_dims(obs, 0)
        if self.config.use_pixels:
            obs_new = np.concatenate([obs, self.prev_obs], -1)
        else:
            obs_new = np.concatenate([obs, self.prev_obs, self.prev_action], 1)

        action_dist_n = self.session.run(self.action_dist_n, {self.obs: obs_new})

        if self.config.deterministic:
            action = np.float32(gaussian_sample(action_dist_n, self.env.action_space.shape[0]))
        else:
            action = np.float32(deterministic_sample(action_dist_n, self.env.action_space.shape[0]))

        self.prev_action = np.expand_dims(np.copy(action),0)
        self.prev_obs = obs
        return action, action_dist_n, np.squeeze(obs_new)

    def learn(self):
        config = self.config
        numeptotal = 0
        self.iter = 0

        total_timesteps = 0
        benchmark_results = []
        benchmark_results.append({"env_id": config.env_id})

        # Create saver
        if config.load_model:
            self.train = False
            self.saver = tf.train.import_meta_graph('{}/model.ckpt.meta'.format(config.load_dir))
            self.saver.restore(self.session, \
                tf.train.latest_checkpoint("{}".format(config.load_dir)))
            if config.use_pixels == False:
                ob_filter_path = os.path.join(config.load_dir, "ob_filter.pkl")
                with open(ob_filter_path, 'rb') as ob_filter_input:
                    self.ob_filter = pickle.load(ob_filter_input)

        policy_vars = []
        # recreate policy net
        for var in tf.global_variables():
            if var.name.startswith("policy"):
                policy_vars.append(var)
                print (var.name)
                print (self.session.run(var, feed_dict={}).shape)


        if self.config.use_pixels:
            self.action_dist_n = load_policy_net_rgb(self.obs, policy_vars, env.action_space.shape[0])
        else:
            self.action_dist_n = load_policy_net(self.obs, policy_vars, [64,64], [True, True], self.env.action_space.shape[0])

        while total_timesteps < self.config.max_timesteps:
            # Generating paths.
            print("Rollout")
            t1_rollout = time.time()
            paths, timesteps_sofar = load_rollout(
                self.env,
                self,
                config.max_pathlength,
                config.timesteps_per_batch,
                save=config.save,
                save_dir=config.save_dir)
            total_timesteps += timesteps_sofar
            t2_rollout = time.time()
            print ("Time for rollout")
            print (t2_rollout - t1_rollout)
            start_time = time.time()

            episoderewards = np.array(
                [path["rewards"].sum() for path in paths])

            print "\n********** Iteration %i ************" % self.iter
            if episoderewards.mean() >= self.env.spec.reward_threshold:
                print "Solved Env"
                self.solved = True

            stats = {}
            numeptotal += len(episoderewards)
            stats["Total number of episodes"] = numeptotal
            stats["Average sum of rewards per episode"] = episoderewards.mean()
            for k, v in stats.iteritems():
                print(k + ": " + " " * (40 - len(k)) + str(v))

            self.iter += 1


if __name__ == '__main__':

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)
    env = gym.make(args.env_id)
    if args.use_pixels:
        env = RGBEnv(env)
    else:
        env = NormalizedEnv(env)
    agent = AsyncNGAgent(env, args)
    agent.learn()
