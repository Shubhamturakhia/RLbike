from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from nets.nn_ops import *
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# Soft target update param
TAU = 0.001
# Base learning rate for the Actor network
ACTOR_LEARNING_RATE = 0.0001
# Base learning rate for the Critic Network
CRITIC_LEARNING_RATE = 0.001

class BaseNetwork(object):
    def __init__(self, state_dim, action_dim, name, initializer=tf.keras.initializers.glorot_normal()):
        """
        Abstarct class for creating networks
        :param state_dim:
        :param action_dim:
        :param stddev:
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.initializer = initializer

        # build network
        self.network = self.build(name)
        self.network_param = [v for v in tf.get_collection(tf.GraphKeys.MODEL_VARIABLES) if name in v.name and
                              "target" not in v.name]

        # build target
        self.target_network = self.build("target_%s" % name)
        self.target_network_param = [v for v in tf.get_collection(tf.GraphKeys.MODEL_VARIABLES) if name in v.name and
                             "target" in v.name]

        self.gradients = None

        # optimizer
        self.optimizer = self.create_optimizer()

    def create_update_op(self):
        update_op = [tf.assign(target_network_param, (1 - TAU) * target_network_param + TAU * network_param)
                        for target_network_param, network_param in zip(self.target_network_param, self.network_param)]

        return update_op

    def create_train_op(self):
        return self.optimizer.apply_gradients([(g, v) for g, v in zip(self.gradients, self.network_param)])

    def build(self, name):
        """
        Abstract method, to be implemented by child classes
        """
        raise NotImplementedError("Not implemented")

    def create_optimizer(self):
        """
        Abstract method, to be implemented by child classes
        """
        raise NotImplementedError("Not implemented")

    def compute_gradient(self):
        """
        Abstract method, compute gradient in order to be used by self.optimizer
        """
        raise NotImplementedError("Not implemented")


class CriticNetwork(BaseNetwork):
    def __init__(self, state_dim, action_dim, name="critic"):
        """
        Initialize critic network. The critic network maintains a copy of itself and target updating ops
        Args
            state_dim: dimension of input space, if is length one, we assume it is low dimension.
            action_dim: dimension of action space.
        """
        super(CriticNetwork, self).__init__(state_dim, action_dim, name=name)

        self.update_op = self.create_update_op()

        # online critic
        self.network, self.state, self.action = self.network

        #target critic
        self.target_network, self.target_state, self.target_action = self.target_network

        # for critic network, the we need one more input variable: y to compute the loss
        # this input variable is fed by: r + gamma * target(s_t+1, action(s_t+1))
        self.y = tf.placeholder(tf.float32, shape=None, name="target_q")
        self.mean_loss = tf.reduce_mean(tf.squared_difference(self.y, self.network))
        self.loss = tf.squared_difference(self.y, self.network)

        # get gradients
        self.gradients = self.compute_gradient()

        # get action gradients
        self.action_gradient = self.compute_action_gradient()

        self.train = self.create_train_op()

    def create_optimizer(self):
        return tf.train.AdamOptimizer(CRITIC_LEARNING_RATE)

    def build(self, name):
        x = tf.placeholder(dtype=tf.float32, shape=(None, ) + self.state_dim, name="%s_input" % name)
        action = tf.placeholder(tf.float32, shape=[None, self.action_dim], name="%s_action" % name)
        with tf.variable_scope(name):
            net = tf.nn.relu(dense_layer(x, 400, use_bias=True, scope="fc1", initializer=self.initializer))
            net = dense_layer(tf.concat((net, action),1), 300, use_bias=True, scope="fc2",
                                  initializer=self.initializer)
            net = tf.nn.relu(net)

            # for low dim, weights are from uniform[-3e-3, 3e-3]
            net = dense_layer(net, 1, initializer=tf.random_uniform_initializer(-3e-3, 3e-3), scope="q",
                                  use_bias=True)
        return tf.squeeze(net), x, action

    def compute_gradient(self):
        grad = tf.gradients(self.mean_loss, self.network_param, name="critic_gradients")
        return grad

    def compute_action_gradient(self):
        action_gradient = tf.gradients(self.network, self.action, name="action_gradients")
        return action_gradient


class ActorNetwork(BaseNetwork):
    def __init__(self, state_dim, action_dim, name="actor"):
        """
        Initialize actor network
        """
        super(ActorNetwork, self).__init__(state_dim, action_dim, name=name)

        self.update_op = self.create_update_op()

        # online actor
        self.network, self.state = self.network

        # target actor
        self.target_network, self.target_state = self.target_network

        # for actor network, we need to know the action gradient in critic network
        self.action_gradient = tf.placeholder(tf.float32, shape=(None, action_dim), name="action_gradients")
        self.gradients = self.compute_gradient()

        self.train = self.create_train_op()

    def create_optimizer(self):
        return tf.train.AdamOptimizer(ACTOR_LEARNING_RATE)

    def build(self, name):
        x = tf.placeholder(dtype=tf.float32, shape=(None, ) + self.state_dim, name="%s_input" % name)
        with tf.variable_scope(name):
            net = tf.nn.relu(dense_layer(x, 400, use_bias=True, scope="fc1", initializer=self.initializer))
            net = tf.nn.relu(dense_layer(net, 300, use_bias=True, scope="fc2", initializer=self.initializer))

            # use tanh to normalize output between [-1, 1]
            net = tf.nn.tanh(dense_layer(net, self.action_dim,
                                             initializer=tf.random_uniform_initializer(-3e-3, 3e-3),
                                             scope="pi", use_bias=True))
            return net, x

    def compute_gradient(self):
        grads = tf.gradients(self.network, self.network_param, -self.action_gradient)
        return grads