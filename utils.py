import numpy as np
import tensorflow as tf
import random
import scipy.signal
import scipy.optimize
import sys
import kfac
import json
import time
from PIL import Image
import os
import shutil
import copy

dtype = tf.float32
weight_decay_fc = 0.0
weight_decay_conv = 0.0

def discount(x, gamma):
    assert x.ndim >= 1
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

def save_ob(ob, folder, timesteps_sofar):
    Image.fromarray((copy.deepcopy(ob) * 255.).astype(np.uint8)).save(folder + '/ob_{}.jpg'.format(timesteps_sofar))

def save_obs(ob_raw, ob, folder, timesteps_sofar):
    Image.fromarray((copy.deepcopy(ob_raw)).astype(np.uint8)).save(folder + '/ob_raw_{}.jpg'.format(timesteps_sofar))
    Image.fromarray((copy.deepcopy(ob) * 255.).astype(np.uint8)).save(folder + '/ob_{}.jpg'.format(timesteps_sofar))


def remkdir(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.mkdir(folder)

# Sample only 1 episode
def load_rollout(env, agent, max_pathlength, n_timesteps, save=False, save_dir="./dummy/"):
    paths = []
    timesteps_sofar = 0

    obs, actions, rewards, rewards_filtered, action_dists = [], [], [], [], []
    ob_raw, ob = env.reset()
    if save and agent.config.use_pixels:
        folder = os.path.join(save_dir, "episode_{}".format(agent.iter))
        # create folder if doesn't exists (remove if exists)
        if agent.iter == 0:
            remkdir(save_dir)
        remkdir(folder)
        save_obs(ob_raw, ob, folder, timesteps_sofar)


    agent.prev_action *= 0.0
    agent.prev_obs *= 0.0
    terminated = False

    for _ in xrange(max_pathlength):
        action, action_dist, ob = agent.act(ob)
        obs.append(ob)
        actions.append(action)
        action_dists.append(action_dist)
        res = env.step(action)
        timesteps_sofar += 1
        reward_filtered = agent.reward_filter(np.asarray([res[2]]))[0]
        ob_raw = res[0]
        ob = res[1]
        rewards.append(res[2])
        rewards_filtered.append(reward_filtered)
        if save and agent.config.use_pixels:
            folder = os.path.join(save_dir, "episode_{}".format(agent.iter))
            save_obs(ob_raw, ob, folder, timesteps_sofar)

        if res[3]:
            terminated = True
            break

    path = {"obs": np.concatenate(np.expand_dims(obs, 0)),
            "action_dists": np.concatenate(action_dists),
            "rewards": np.array(rewards),
            "rewards_filtered": np.array(rewards_filtered),
            "actions": np.array(actions),
            "terminated": terminated,}
    paths.append(path)
    agent.prev_action *= 0.0
    agent.prev_obs *= 0.0
    timesteps_sofar += len(path["rewards"])
    return paths, timesteps_sofar

def rollout(env, agent, max_pathlength, n_timesteps):
    paths = []
    timesteps_sofar = 0
    while timesteps_sofar < n_timesteps:
        obs, actions, rewards, rewards_filtered, action_dists = [], [], [], [], []
        ob = env.reset()
        agent.prev_action *= 0.0
        agent.prev_obs *= 0.0
        terminated = False

        for _ in xrange(max_pathlength):
            action, action_dist, ob = agent.act(ob)
            obs.append(ob)
            actions.append(action)
            action_dists.append(action_dist)
            res = env.step(action)
            reward_filtered = agent.reward_filter(np.asarray([res[1]]))[0]
            ob = res[0]
            rewards.append(res[1])
            rewards_filtered.append(reward_filtered)
            if res[2]:
                terminated = True
                break

        path = {"obs": np.concatenate(np.expand_dims(obs, 0)),
                "action_dists": np.concatenate(action_dists),
                "rewards": np.array(rewards),
                "rewards_filtered": np.array(rewards_filtered),
                "actions": np.array(actions),
                "terminated": terminated,}
        paths.append(path)
        agent.prev_action *= 0.0
        agent.prev_obs *= 0.0
        timesteps_sofar += len(path["rewards"])
    return paths, timesteps_sofar

def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

def ortho_init(scale=1.0):
    def _ortho_init(shape, dtype, partition_info=None):
        #lasagne ortho init for tf
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4: # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v # pick the one with the correct shape
        q = q.reshape(shape)
        return (scale * q[:shape[0], :shape[1]]).astype(np.float32)
    return _ortho_init

class VF(object):
    coeffs = None

    def __init__(self, config, session):
        self.net = None
        self.config = config
        self.session = session
        # use exponential average when computing baseline
        self.averager = tf.train.ExponentialMovingAverage(decay=self.config.moving_average_vf)

    def init_vf(self,paths):
        if self.config.use_pixels:
            featmat = np.concatenate([self._features_rgb(path) for path in paths])
            return self.create_net(featmat.shape[1:])
        else:
            featmat = np.concatenate([self._features(path) for path in paths])
            return self.create_net([featmat.shape[1]])

    def fc_net(self, x, weight_loss_dict=None, reuse=None):
        net = x
        hidden_sizes = [64,64]
        for i in range(len(hidden_sizes)):
            net = linear(net, hidden_sizes[i], "vf/l{}".format(i), initializer=normalized_columns_initializer(1.0), weight_loss_dict=weight_loss_dict, reuse=reuse)
            net = tf.nn.elu(net)

        net = linear(net, 1, "vf/value", initializer=None, weight_loss_dict=weight_loss_dict, reuse=reuse)
        net = tf.reshape(net, (-1, ))
        return net, weight_loss_dict

    def conv_net(self, x, weight_loss_dict=None, reuse=None):

        # Conv Layers
        for i in range(2):
            x = tf.nn.elu(conv2d(x, 32, "vf/l{}".format(i), [3, 3], [2, 2], \
                initializer=ortho_init(np.sqrt(2)), weight_loss_dict=weight_loss_dict, reuse=reuse))

        x = flatten(x)
        # One more linear layer
        x = linear(x, 256, "vf/l{}".format(i+1), \
            initializer=ortho_init(np.sqrt(2)), weight_loss_dict=weight_loss_dict, reuse=reuse)
        x = tf.nn.elu(x)

        x = linear(x, 1, "vf/value", \
            initializer=ortho_init(1), weight_loss_dict=weight_loss_dict, reuse=reuse)
        x = tf.reshape(x, (-1, ))

        return x, weight_loss_dict

    def create_net(self, shape):
        self.x = tf.placeholder(tf.float32, shape=[None] + list(shape), name="x")
        self.y = tf.placeholder(tf.float32, shape=[None], name="y")
        self.vf_weight_loss_dict = {}
        with tf.name_scope('train_vf'):
            if self.config.use_pixels:
                self.net, self.vf_weight_loss_dict = self.conv_net(self.x, self.vf_weight_loss_dict)
            else:
                self.net, self.vf_weight_loss_dict = self.fc_net(self.x, self.vf_weight_loss_dict)

        self.bellman_error = (self.net - self.y)
        l2 = tf.reduce_mean(self.bellman_error * self.bellman_error)
        # get weight decay losses for value function
        vf_losses = tf.get_collection('vf_losses', None)

        self.loss = loss = l2 + tf.add_n(vf_losses)

        var_list_all = tf.trainable_variables()
        self.var_list = var_list = []
        for var in var_list_all:
            if "vf" in str(var.name):
                var_list.append(var)

        self.update_averages = self.averager.apply(self.var_list)

        # build test net with exponential moving averages for inference
        with tf.name_scope('test_vf'):
            if self.config.use_pixels:
                self.test_net, _ = self.conv_net(self.x, None, reuse=True)
            else:
                self.test_net, _ = self.fc_net(self.x, None, reuse=True)

        if self.config.use_adam_vf:
            self.loss_fisher = None
        else:
            sample_net = self.net + tf.random_normal(tf.shape(self.net))
            self.loss_fisher = loss_fisher = tf.reduce_mean(tf.pow(self.net - tf.stop_gradient(sample_net), 2))

        return self.loss, self.loss_fisher, self.vf_weight_loss_dict

    def init_vf_train_op(self, loss_vf, loss_vf_sampled, wd_dict):
        if self.config.use_adam_vf:
            # 0.001
            self.update_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss_vf)
            self.queue_runner = None
        elif self.config.use_sgd_vf:
            # 0.001*(1.-0.9), 0.9
            self.update_op = tf.train.MomentumOptimizer(0.001*(1.-0.9), 0.9).minimize(loss_vf)
            self.queue_runner = None
        else:
            self.update_op, self.queue_runner = kfac.KfacOptimizer(
                                             learning_rate=self.config.lr_vf,
                                             cold_lr=self.config.lr_vf/3.,
                                             momentum=self.config.mom_vf,
                                             clip_kl=self.config.kl_desired_vf,
                                             upper_bound_kl=False,
                                             epsilon=self.config.epsilon_vf,
                                             stats_decay=self.config.stats_decay_vf,
                                             async=self.config.async_kfac,
                                             kfac_update=self.config.kfac_update_vf,
                                             cold_iter=self.config.cold_iter_vf,
                                             weight_decay_dict=wd_dict).minimize(
                                                  loss_vf,
                                                  loss_vf_sampled,
                                                  self.var_list)

        with tf.control_dependencies([self.update_op]):
            self.train = tf.group(self.update_averages)

        return self.train, self.queue_runner

    def _features(self, path):
        o = path["obs"].astype('float32')
        o = o.reshape(o.shape[0], -1)
        act = path["action_dists"].astype('float32')
        l = len(path["rewards"])
        al = np.arange(l).reshape(-1, 1) / 10.0
        ret = np.concatenate([o, act, al, np.ones((l, 1))], axis=1)
        return ret

    def _features_rgb(self, path):
        o = path["obs"].astype('float32')
        return o

    def get_feed_dict(self, paths):
        if self.config.use_pixels:
            featmat = np.concatenate([self._features_rgb(path) for path in paths])
        else:
            featmat = np.concatenate([self._features(path) for path in paths])
        returns = np.concatenate([path["returns"] for path in paths])
        return {self.x: featmat, self.y: returns}

    def fit(self, paths):
        if self.config.use_pixels:
            featmat = np.concatenate([self._features_rgb(path) for path in paths])
        else:
            featmat = np.concatenate([self._features(path) for path in paths])
        if self.net is None:
            self.create_net(featmat.shape[1:])
        returns = np.concatenate([path["returns"] for path in paths])

        self.session.run(self.train, {self.x: featmat, self.y: returns})

    def predict_many(self, paths):
        if self.net is None:
            return np.zeros(len(path["rewards"]))
        else:
            if self.config.use_pixels:
                featmat = np.concatenate([self._features_rgb(path) for path in paths])
            else:
                featmat = np.concatenate([self._features(path) for path in paths])
        ret = self.session.run(self.test_net, {self.x: featmat})
        ret = np.reshape(ret, (ret.shape[0], ))
        return ret

    def predict(self, path):
        if self.net is None:
            return np.zeros(len(path["rewards"]))
        else:
            if self.config.use_pixels:
                ret = self.session.run(self.test_net, {self.x: self._features_rgb(path)})
            else:
                ret = self.session.run(self.test_net, {self.x: self._features(path)})
            ret = np.reshape(ret, (ret.shape[0], ))
            return ret

def linear(x, size, name, initializer=None, bias_init=0, weight_loss_dict=None, reuse=None):
#    assert len(name.split('/')) == 2 # make sure that name has format policy/l1 or vf/l1

    with tf.variable_scope(name, reuse=reuse):
        w = tf.get_variable("w", [x.get_shape()[1], size], initializer=initializer)
        b = tf.get_variable("b", [size], initializer=tf.constant_initializer(bias_init))

        if weight_decay_fc > 0.0 and weight_loss_dict is not None:
            weight_decay = tf.mul(tf.nn.l2_loss(w), weight_decay_fc, name='weight_decay_loss')
            if weight_loss_dict is not None:
                weight_loss_dict[w] = weight_decay_fc
                weight_loss_dict[b] = 0.0
            tf.add_to_collection(name.split('/')[0] + '_' + 'losses', weight_decay)

        return tf.nn.bias_add(tf.matmul(x, w), b)

def linearnobias(x, size, name, initializer=None, weight_loss_dict=None, reuse=None):
    #assert len(name.split('/')) == 2 # make sure that name has format policy/l1 or vf/l1
    with tf.variable_scope(name, reuse=reuse):
        w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=initializer)

        if weight_decay_fc > 0.0 and weight_loss_dict is not None:
            weight_decay = tf.mul(tf.nn.l2_loss(w), weight_decay_fc, name='weight_decay_loss')
            if weight_loss_dict is not None:
                weight_loss_dict[w] = weight_decay_fc
            tf.add_to_collection(name.split('/')[0] + '_' + 'losses', weight_decay)

        return tf.matmul(x, w)

def gaussian_sample(action_dist, action_size):
    return np.random.randn(action_size) * action_dist[0,action_size:] + action_dist[0,:action_size]

def deterministic_sample(action_dist, action_size):
    return action_dist[0,:action_size]

# returns mean and std of gaussian distribution
def get_moments(action_dist, action_size):
    mean = tf.reshape(action_dist[:, :action_size], [tf.shape(action_dist)[0], action_size])
    std = (tf.reshape(action_dist[:, action_size:], [tf.shape(action_dist)[0], action_size]))
    return mean, std


def loglik(action, action_dist, action_size):
    mean, std = get_moments(action_dist, action_size)
    return -0.5 * tf.reduce_sum(tf.square((action-mean) / std),reduction_indices=-1) \
            -0.5 * tf.log(2.0*np.pi)*action_size - tf.reduce_sum(tf.log(std),reduction_indices=-1)

def kl_div(action_dist1, action_dist2, action_size):
    mean1, std1 = get_moments(action_dist1, action_size)
    mean2, std2 = get_moments(action_dist2, action_size)
    numerator = tf.square(mean1 - mean2) + tf.square(std1) - tf.square(std2)
    denominator = 2 * tf.square(std2) + 1e-8
    return tf.reduce_sum(
        numerator/denominator + tf.log(std2) - tf.log(std1),reduction_indices=-1)

def entropy(action_dist, action_size):
    _, std = get_moments(action_dist, action_size)
    return tf.reduce_sum(tf.log(std),reduction_indices=-1) + .5 * np.log(2*np.pi*np.e) * action_size

def conv2d_loaded(x, weights, biases, num_filters, filter_size=(3, 3), stride=(1, 1), pad="SAME"):
    filter_shape = [filter_size[0], filter_size[1], int(x.get_shape()[3]), num_filters]
    stride_shape = [1, stride[0], stride[1], 1]

    return tf.nn.bias_add(tf.nn.conv2d(x, weights, stride_shape, pad), biases)

# Bits and pieces taken from Jimmy and universe-starter-agent
def conv2d(x, num_filters, name, filter_size=(3, 3), stride=(1, 1), pad="SAME", initializer=None, bias_init=0, weight_loss_dict=None, reuse=None):
    with tf.variable_scope(name, reuse=reuse):
        if initializer == None:
            stddev = 0.01
            initializer = tf.random_normal_initializer(stddev=stddev)

        filter_shape = [filter_size[0], filter_size[1], int(x.get_shape()[3]), num_filters]
        stride_shape = [1, stride[0], stride[1], 1]

        weights = tf.get_variable('weights', filter_shape,
                                  initializer=initializer)
        biases = tf.get_variable(
            'biases', [num_filters], initializer=tf.constant_initializer(0.))

        if weight_decay_conv > 0.0 and weight_loss_dict is not None:
            weight_decay = tf.mul(tf.nn.l2_loss(weights), weight_decay_conv, name='weight_decay_loss')
            if weight_loss_dict is not None:
                weight_loss_dict[weights] = weight_decay_conv
                weight_loss_dict[biases] = 0.0
            tf.add_to_collection(name.split('/')[0] + '_' + 'losses', weight_decay)

        return tf.nn.bias_add(tf.nn.conv2d(x, weights, stride_shape, pad), biases)

def flatten(x):
    return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])

# universe-starter-agent 42x42 net
def create_policy_net_rgb(obs, action_size):
    x = obs
    weight_loss_dict = {}

    # Conv Layers
    for i in range(2):
        x = tf.nn.relu(conv2d(x, 32, "policy/l{}".format(i), [3, 3], [2, 2], \
            initializer=ortho_init(np.sqrt(2)), weight_loss_dict=weight_loss_dict))

    x = flatten(x)
    # One more linear layer
    x = linear(x, 256, "policy/l{}".format(i+1), \
            initializer=ortho_init(np.sqrt(2)), weight_loss_dict=weight_loss_dict)
    x = tf.nn.relu(x)

    mean = linear(x, action_size, "policy/mean", ortho_init(1), weight_loss_dict=weight_loss_dict)
    log_std = tf.Variable(tf.zeros([action_size]), name="policy/log_std")
    log_std_expand = tf.expand_dims(log_std, 0)
    std = tf.tile(tf.exp(log_std_expand), [tf.shape(mean)[0], 1])
    output = tf.concat(1, [tf.reshape(mean, [-1, action_size]), tf.reshape(std, [-1, action_size])])

    return output, weight_loss_dict

def load_policy_net_rgb(obs, policy_vars, action_size):
    x = obs

    # Conv Layers
    for i in range(2):
        x = tf.nn.relu(conv2d_loaded(x, policy_vars[2*i], policy_vars[2*i+1], 32, [3,3], [2,2]))
    i+=1
    x = flatten(x)
    x = tf.nn.bias_add(tf.matmul(x, policy_vars[2*i]), policy_vars[2*i+1])
    x = tf.nn.relu(x)
    i += 1
    # Linear layer
    mean = tf.nn.bias_add(tf.matmul(x, policy_vars[2*i]), policy_vars[2*i+1])
    log_std = policy_vars[-1]
    log_std_expand = tf.expand_dims(log_std, 0)
    std = tf.tile(tf.exp(log_std_expand), [tf.shape(mean)[0], 1])
    output = tf.concat(1, [tf.reshape(mean, [-1, action_size]), tf.reshape(std, [-1, action_size])])

    return output

def load_policy_net(obs, policy_vars, hidden_sizes, nonlinear, action_size):
    x = obs
    for i in range(len(hidden_sizes)):
        x = tf.nn.bias_add(tf.matmul(x, policy_vars[2*i]), policy_vars[2*i+1])
        if nonlinear[i]:
            x = tf.nn.tanh(x)
    i+=1
    mean = tf.nn.bias_add(tf.matmul(x, policy_vars[2*i]), policy_vars[2*i+1])
    log_std = policy_vars[-1]
    log_std_expand = tf.expand_dims(log_std, 0)
    std = tf.tile(tf.exp(log_std_expand), [tf.shape(mean)[0], 1])
    output = tf.concat(1, [tf.reshape(mean, [-1, action_size]), tf.reshape(std, [-1, action_size])])

    return output

def create_policy_net(obs, hidden_sizes, nonlinear, action_size):
    x = obs
    weight_loss_dict = {}
    for i in range(len(hidden_sizes)):
        x = linear(x, hidden_sizes[i], "policy/l{}".format(i), initializer=normalized_columns_initializer(1.0), weight_loss_dict=weight_loss_dict)
        if nonlinear[i]:
            x = tf.nn.tanh(x)
    mean = linear(x, action_size, "policy/mean", initializer=normalized_columns_initializer(0.1), weight_loss_dict=weight_loss_dict)
    log_std = tf.Variable(tf.zeros([action_size]), name="policy/log_std")
    log_std_expand = tf.expand_dims(log_std, 0)
    std = tf.tile(tf.exp(log_std_expand), [tf.shape(mean)[0], 1])
    output = tf.concat(1, [tf.reshape(mean, [-1, action_size]), tf.reshape(std, [-1, action_size])])

    return output, weight_loss_dict

def var_shape(x):
    out = [k.value for k in x.get_shape()]
    assert all(isinstance(a, int) for a in out), \
        "shape function assumes that shape is fully known"
    return out


def numel(x):
    return np.prod(var_shape(x))

def flatgrad(loss, var_list):
    grads = tf.gradients(loss, var_list)
    return tf.concat([tf.reshape(grad, [numel(v)])
                      for (v, grad) in zip(var_list, grads)], 0)


class dict2(dict):
    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self

def explained_variance(ypred, y):
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)
    return np.nan if vary==0 else 1 - np.var(y-ypred)/vary
