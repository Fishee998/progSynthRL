"""
This part of code is the DQN brain, which is a brain of the agent.
All decisions are made in here.
Using Tensorflow to build the neural network.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.7.3
"""
from collections import deque
from compiler.ast import flatten
from random import choice
import example
import numpy as np
import pandas as pd
import tensorflow as tf
import math
import random

np.random.seed(1)
tf.set_random_seed(1)


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions1,
            # n_actions2,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=10,
            memory_size=100,
            batch_size=32,
            e_greedy_increment=0.0001,
            output_graph=False,
    ):
        self.n_actions1 = n_actions1
        # self.n_actions2 = n_actions2
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.memory_size_good = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max


        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        # self.memory = np.zeros((self.memory_size, self.ob_features * 2 + 3))
        self.memory = []
        self.memory_good = []

        # consist of [target_net, evaluate_net]
        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def _build_net(self):
        # ------------------ build target_net ------------------
        # self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # input
        self.nodes_ = tf.placeholder(tf.float32, shape=(None, None, self.n_features), name='node_')
        self.children_ = tf.placeholder(tf.int32, shape=(None, None, None), name='children_')
        self.action1_ = tf.placeholder(tf.float32, shape=(None, 135), name='action1_')
        with tf.variable_scope('target_net'):
            # c_names(collections_names) are the collections to store variables
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                conv2 = self.conv_layer(1, 100, self.nodes_, self.children_, self.n_features)
                pooling2 = self.pooling_layer(conv2)
                self.pooling2 = pooling2 = tf.concat((pooling2, self.action1_), axis=1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2_1'):
                self.q_next1 = self.hidden_layer(pooling2, 235, self.n_actions1)

        # ------------------ build evaluate_net ------------------

        self.nodes = tf.placeholder(tf.float32, shape=(None, None, self.n_features), name='node')
        self.children = tf.placeholder(tf.int32, shape=(None, None, None), name='children')
        self.action1 = tf.placeholder(tf.float32, shape=(None, 135), name='action1')

        self.q_target1 = tf.placeholder(tf.float32, [None, self.n_actions1], name='Q_target')
        # for calculating loss
        with tf.variable_scope('eval_net'):
            # c_names(collections_names) are the collections to store variables
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 10, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                conv1 = self.conv_layer(1, 100, self.nodes, self.children, self.n_features)
                pooling = self.pooling_layer(conv1)
                pooling = tf.concat((pooling, self.action1), axis=1)
                #pooling = tf.concat(1, pooling, self.action1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2_1'):
                self.q_eval1 = self.hidden_layer(pooling, 235, self.n_actions1)

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target1, self.q_eval1))

        with tf.variable_scope('train'):
            self._train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def make_tfrecord(self):
        out_name = 'transition.tfrecord'
        self.tfrecord_wrt = tf.python_io.TFRecordWriter(out_name)
        # return tfrecord_wrt

    def store_transition(self, nodes, children, a1, a, r, nodes_, children_, a1_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = {}

        transition['nodes'] = nodes
        transition['children'] = children
        transition['action1'] = a1
        transition['nodes_'] = nodes_
        transition['children_'] = children_
        transition['action1_'] = a1_
        transition['acRe'] = [a, r]

        index = self.memory_counter % self.memory_size
        if self.memory_counter < self.memory_size:
            self.memory.append(transition)
        else:
            self.memory[index] = transition
        self.memory_counter += 1


    def store_transition_good(self, nodes, children, a1, a, r, nodes_, children_, a1_):
        if not hasattr(self, 'memory_counter_good'):
            self.memory_counter_good = 0

        transition = {}

        transition['nodes'] = nodes
        transition['children'] = children
        transition['action1'] = a1
        transition['nodes_'] = nodes_
        transition['children_'] = children_
        transition['action1_'] = a1_
        transition['acRe'] = [a, r]

        index = self.memory_counter_good % self.memory_size_good
        if self.memory_counter_good < self.memory_size_good:
            self.memory_good.append(transition)
        else:
            self.memory_good[index] = transition
        self.memory_counter_good += 1


    def reshapeChildNodes(self, nodes, children):
        childre = []
        for index in range(len(children)):
            child = children[index]
            if len(child) < 5:
                while len(child) < 5:
                    child.append(0)
                childre.append(child)
            else:
                childre.append(child)
        children1 = [childre]
        nodes = [nodes]
        return nodes, children1

        # action1 action2 legal

    def action2set(self, action2s):
        # actionLen = example.getLength(action2s)
        action2Seleced = []
        index = 0
        while example.get_action2(action2s, index) != 100:
            action2Seleced.append(example.get_action2(action2s, index))
            index += 1
        return action2Seleced

    def getAction(self, actSet, action1_value):
        action1Set = np.nonzero(actSet)
        action1_prob = []
        for index in action1Set:
            action1_prob.append(action1_value[0][index])
        action1 = np.argmax(action1_prob)
        action = action1Set[action1]
        real_action = actSet[action1]
        return action, real_action
        # no restrict
        # action1 = np.argmax(action1_value[0])

    def getAction1_random(self, act1Set):
        action1s = np.nonzero(act1Set)
        action1 = random.choice(action1s[0])

        real_action = act1Set[action1]
        return action1, real_action

    def getAction2(self, candidate, action1_real, action2_value):
        action2 = example.getLegalAction2(candidate, action1_real)
        action2set = self.action2set(action2)
        action2_prob = []
        for index in list(set(action2set)):
            action2_prob.append(action2_value[0][index])
        action2_index = np.argmax(action2_prob)

        action2 = action2set[action2_index]

        # action2  = np.argmax(action2_value[0])
        return action2

    def getAction2_random(self, candidate, action1_real):
        action2_ = example.getLegalAction2(candidate, action1_real)
        action2set = self.action2set(action2_)
        action2 = choice(action2set)
        return action2

    def getAction(self, action1, action2):
        action = []
        action.append(action1)
        action.append(action2)
        action = tuple(action)
        return action

    def getAction1(self,candidate, act1Set, action1, action1_value):
        action1s = np.nonzero(act1Set[:-1])
        # act1Set[action1 - 1] != 0 and
        '''
        if act1Set[action1 - 1] != -1 and act1Set[action1 - 1] != 0:
            action2set = np.array(self.action2set(example.getLegalAction2(candidate, action1))) + 42
            # actions = action2set
            actions = np.append(action1s, action2set)
        else:
            actions = action1s
        '''
        actions = action1s[0]
        action1_prob = []
        for index in actions:
            action1_prob.append(action1_value[0][index])
        action1 = np.argmax(action1_prob)
        action = actions[action1]
        return action

    def getChoosenActions(self, candidate, act1Set, action1):
        action1s = np.nonzero(act1Set)[0]
        # act1Set[action1 - 1] != 0 and
        if act1Set[action1 - 1] != -1 and act1Set[action1 - 1] != 0:
            action2set = np.array(self.action2set(example.getLegalAction2(candidate, action1))) + 42
            # actions = action2set
            actions = np.append(action1s, action2set)
        else:
            actions = action1s

        return actions

    def getAction_random(self, candidate, act1Set, action1):
        # print("act1Set", act1Set)
        action1s = np.nonzero(act1Set[:-1])
        action1 = int(action1)
        '''
        # (act1Set[action1-1] != 0)
        if act1Set[action1 - 1] != 0 and act1Set[action1 - 1] != -1:
            # action2 = np.array(range(42, 92))
            action2set = np.array(self.action2set(example.getLegalAction2(candidate, action1))) + 42
            # actions = action2set
            actions = np.append(action1s, action2set)
        else:
            actions = action1s[0]
        actions = actions.flatten()
        '''
        # actions = np.append(action1s, action2set)
        action = random.choice(action1s[0])

        # real_action = act1Set[action1]
        return action

    def choose_action(self, action1, nodes, children1, onehotaction1, candidate):
        #observation = observation[np.newaxis, :]
        action_value = action1
        if np.random.uniform() < self.epsilon:
            action1_ = []
            action1_.append(onehotaction1)
            nodes, children1 = self._pad_batch_(nodes, children1)

            actions_value1 = self.sess.run(self.q_eval1, feed_dict={self.nodes: nodes, self.children: children1, self.action1: action1_})
            action = self.getAction1(candidate, onehotaction1[42:], int(action1), actions_value1)
        else:
            # print("choose by random")
            action = self.getAction_random(candidate, onehotaction1[42:], int(action1))
        action = int(action)
        action_store = action
        if action < 42:
            action = action + 1
        else:
            #action_store = action
            action = action - 42

        return action, action_store

    def _pad_batch_(self, nodes1, children1):
        nodes = []
        nodes.append(nodes1)
        children = []
        children.append(children1)

        max_nodes = max([len(x) for x in nodes])
        max_children = max([len(x) for x in children])
        feature_len = len(nodes[0][0])
        child_len = max([len(c) for n in children for c in n])

        nodes = [n + [[0] * feature_len] * (max_nodes - len(n)) for n in nodes]
        # pad batches so that every batch has the same number of nodes
        children = [n + ([[]] * (max_children - len(n))) for n in children]
        # pad every child sample so every node has the same number of children
        children = [[c + [0] * (child_len - len(c)) for c in sample] for sample in children]

        return nodes, children

    def _pad_batch(self, nodes, children, action1, a, r, nodes_, children_, action1_):
        if not nodes:
            return [], [], []
        max_nodes = max([len(x) for x in nodes])
        max_children = max([len(x) for x in children])
        feature_len = len(nodes[0][0])
        child_len = max([len(c) for n in children for c in n])

        nodes = [n + [[0] * feature_len] * (max_nodes - len(n)) for n in nodes]
        # pad batches so that every batch has the same number of nodes
        children = [n + ([[]] * (max_children - len(n))) for n in children]
        # pad every child sample so every node has the same number of children
        children = [[c + [0] * (child_len - len(c)) for c in sample] for sample in children]

        max_nodes_ = max([len(y) for y in nodes_])
        max_children_ = max([len(y) for y in children_])
        feature_len_ = len(nodes_[0][0])
        child_len_ = max([len(c) for n in children_ for c in n])

        nodes_ = [n + [[0] * feature_len_] * (max_nodes_ - len(n)) for n in nodes_]
        # pad batches so that every batch has the same number of nodes
        children_ = [n + ([[]] * (max_children_ - len(n))) for n in children_]
        # pad every child sample so every node has the same number of children
        children_ = [[c + [0] * (child_len_ - len(c)) for c in sample] for sample in children_]

        return nodes, children, action1, a, r, nodes_, children_, action1_

    def x(self, a):
        children_ = []
        nodes_ = []
        action1_ = []
        action1 = []
        nodes = []
        children = []
        action = []
        reward = []
        for i in range(len(a)):
            children_.append(a[i]["children_"])
            nodes_.append(a[i]["nodes_"])
            action1_.append(a[i]["action1_"])
            action1.append(a[i]["action1"])
            nodes.append(a[i]["nodes"])
            children.append(a[i]["children"])
            action.append(a[i]["acRe"][0])
            reward.append(a[i]["acRe"][1])
            yield (nodes, children, action1, action, reward, nodes_, children_, action1_)

    def batch_samples_(self, gen, batch_size):
        """Batch samples from a generator"""
        nodes, children, action1, action, reward, nodes_, children_, action1_ = [], [] ,[], [] ,[], [], [], []
        samples = 0
        for n, c, a1, a, r, n_, c_, a1_ in gen:
            nodes.append(n[samples])
            children.append(c[samples])
            action1.append(a1[samples])
            nodes_.append(n_[samples])
            children_.append(c_[samples])
            action1_.append(a1_[samples])
            action.append(a[samples])
            reward.append(r[samples])
            # labels.append(l)
            samples += 1
            if samples >= batch_size:
                yield self._pad_batch(nodes, children, action1, action, reward, nodes_, children_, action1_)
                nodes, children, action1, action, reward, nodes_, children_, action1_ = [], [], [], [], [], [], [], []
                samples = 0

        if nodes:
            yield self._pad_batch(nodes, children, action1, action, reward, nodes_, children_, action1_)

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            # print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index_good = np.random.choice(self.memory_size_good, size=self.batch_size - 16)
            sample_index = np.random.choice(self.memory_size, size=16)
        else:
            sample_index_good = np.random.choice(self.memory_size_good, size=self.batch_size -16)
            sample_index = np.random.choice(self.memory_size, size=16)

        batch_memory = []
        for index in sample_index:
            batch_memory.append(self.memory[index])

        for index in sample_index_good:
            batch_memory.append(self.memory_good[index])

        for i, batch in enumerate(self.batch_samples_(self.x(batch_memory[:]), self.batch_size)):
            nodes, children, action1, action, reward, nodes_, children_, action1_ = batch

            pool, q_next1, q_eval1 = self.sess.run(
                [self.pooling2, self.q_next1, self.q_eval1],
                feed_dict={
                    self.nodes: nodes,
                    self.children: children,
                    self.action1: action1,
                    self.nodes_: nodes_,  # fixed params
                    self.children_: children_,
                    self.action1_: action1_,
                })

            q_target1 = q_eval1.copy()

            batch_index = np.arange(self.batch_size, dtype=np.int32)

            q_next1_ = []
            for index in range(self.batch_size):
                actionchoosen = action1[index][42:][:-1]
                actions = np.nonzero(actionchoosen)
                action1_prob = []
                for index1 in actions[0]:
                    action1_prob.append(q_next1[index][index1])
                bestReward = np.max(action1_prob)
                q_next1_.append(bestReward)

            q_target1[batch_index, action] = reward + self.gamma * np.array(q_next1_)

            # train eval network
            _, self.cost = self.sess.run([self._train_op, self.loss],
                                         feed_dict={self.nodes: nodes,  # fixed params
                                                    self.children: children,
                                                    self.action1: action1,
                                                    self.q_target1: q_target1,
                                                    })
            self.cost_his.append(self.cost)

        # increasing epsilon
        if self.epsilon < self.epsilon_max:
            self.epsilon = self.epsilon + self.epsilon_increment if self.learn_step_counter %10 == 0 else self.epsilon
        else:
            self.epsilon = self.epsilon_max
        self.learn_step_counter += 1
        #self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        #self.learn_step_counter += 1


    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()


    def conv_layer(self, num_conv, output_size, nodes, children, feature_size):
        """Creates a convolution layer with num_conv convolutions merged together at
        the output. Final output will be a tensor with shape
        [batch_size, num_nodes, output_size * num_conv]"""

        with tf.name_scope('conv_layer'):
            self.nodes1 = nodes = [
                self.conv_node(nodes, children, feature_size, output_size)
                for _ in range(num_conv)
            ]
            a = tf.concat(nodes, axis=2)
            return tf.concat(nodes, axis=2)

    def pooling_layer(self, nodes):
        """Creates a max dynamic pooling layer from the nodes."""
        with tf.name_scope("pooling"):
            pooled = tf.reduce_max(nodes, axis=1)
            return pooled

    def hidden_layer(self, pooled, input_size, output_size):
        """Create a hidden feedforward layer."""
        with tf.name_scope("hidden"):
            weights = tf.Variable(
                tf.truncated_normal(
                    [input_size, output_size], stddev=1.0 / math.sqrt(input_size)
                ),
                name='weights'
            )

            init = tf.truncated_normal([output_size, ], stddev=math.sqrt(2.0 / input_size))
            # init = tf.zeros([output_size,])
            biases = tf.Variable(init, name='biases')

            with tf.name_scope('summaries'):
                tf.summary.histogram('weights', [weights])
                tf.summary.histogram('biases', [biases])

            return tf.nn.sigmoid(tf.matmul(pooled, weights) + biases)


    def conv_step(self, nodes, children, feature_size, w_t, w_r, w_l, b_conv):
        """Convolve a batch of nodes and children.

        Lots of high dimensional tensors in this function. Intuitively it makes
        more sense if we did this work with while loops, but computationally this
        is more efficient. Don't try to wrap your head around all the tensor dot
        products, just follow the trail of dimensions.
        """
        with tf.name_scope('conv_step'):
            # nodes is shape (batch_size x max_tree_size x feature_size)
            # children is shape (batch_size x max_tree_size x max_children)

            with tf.name_scope('trees'):
                # children_vectors will have shape
                # (batch_size x max_tree_size x max_children x feature_size)
                children_vectors = self.children_tensor(nodes, children, feature_size)
                # print('children_vectors', children_vectors)
                # add a 4th dimension to the nodes tensor
                nodes = tf.expand_dims(nodes, axis=2)
                # tree_tensor is shape
                # (batch_size x max_tree_size x max_children + 1 x feature_size)
                # tree_tensor = tf.concat([nodes, children_vectors], axis=2, name='trees')

                self.tree_tensor = tf.concat([nodes, children_vectors], axis=2, name='trees')
                tree_tensor = self.tree_tensor

            with tf.name_scope('coefficients'):
                # coefficient tensors are shape (batch_size x max_tree_size x max_children + 1)
                c_t = self.eta_t(children)
                c_r = self.eta_r(children, c_t)
                c_l = self.eta_l(children, c_t, c_r)

                # concatenate the position coefficients into a tensor
                # (batch_size x max_tree_size x max_children + 1 x 3)
                coef = tf.stack([c_t, c_r, c_l], axis=3, name='coef')

            with tf.name_scope('weights'):
                # stack weight matrices on top to make a weight tensor
                # (3, feature_size, output_size)
                weights = tf.stack([w_t, w_r, w_l], axis=0)

            with tf.name_scope('combine'):
                batch_size = tf.shape(children)[0]
                max_tree_size = tf.shape(children)[1]
                max_children = tf.shape(children)[2]

                # reshape for matrix multiplication
                x = batch_size * max_tree_size
                y = max_children + 1
                result = tf.reshape(tree_tensor, (x, y, feature_size))
                coef = tf.reshape(coef, (x, y, 3))
                result = tf.matmul(result, coef, transpose_a=True)
                result = tf.reshape(result, (batch_size, max_tree_size, 3, feature_size))

                # output is (batch_size, max_tree_size, output_size)
                result = tf.tensordot(result, weights, [[2, 3], [0, 1]])

                # output is (batch_size, max_tree_size, output_size)
                return tf.nn.tanh(result + b_conv, name='conv')

    def conv_node(self, nodes, children, feature_size, output_size):
        """Perform convolutions over every batch sample."""
        with tf.name_scope('conv_node'):
            std = 1.0 / math.sqrt(feature_size)
            w_t, w_l, w_r = (
                tf.Variable(tf.truncated_normal([feature_size, output_size], stddev=std), name='Wt'),
                tf.Variable(tf.truncated_normal([feature_size, output_size], stddev=std), name='Wl'),
                tf.Variable(tf.truncated_normal([feature_size, output_size], stddev=std), name='Wr'),
            )
            init = tf.truncated_normal([output_size, ], stddev=math.sqrt(2.0 / feature_size))
            # init = tf.zeros([output_size,])
            b_conv = tf.Variable(init, name='b_conv')

            with tf.name_scope('summaries'):
                tf.summary.histogram('w_t', [w_t])
                tf.summary.histogram('w_l', [w_l])
                tf.summary.histogram('w_r', [w_r])
                tf.summary.histogram('b_conv', [b_conv])
            return self.conv_step(nodes, children, feature_size, w_t, w_r, w_l, b_conv)


    def eta_t(self, children):
        """Compute weight matrix for how much each vector belongs to the 'top'"""
        with tf.name_scope('coef_t'):
            # children is shape (batch_size x max_tree_size x max_children)
            batch_size = tf.shape(children)[0]
            max_tree_size = tf.shape(children)[1]
            max_children = tf.shape(children)[2]
            # eta_t is shape (batch_size x max_tree_size x max_children + 1)
            return tf.tile(tf.expand_dims(tf.concat(
                [tf.ones((max_tree_size, 1)), tf.zeros((max_tree_size, max_children))],
                axis=1), axis=0,
            ), [batch_size, 1, 1], name='coef_t')

    def eta_r(self, children, t_coef):
        """Compute weight matrix for how much each vector belogs to the 'right'"""
        with tf.name_scope('coef_r'):
            # children is shape (batch_size x max_tree_size x max_children)
            children = tf.cast(children, tf.float32)
            batch_size = tf.shape(children)[0]
            max_tree_size = tf.shape(children)[1]
            max_children = tf.shape(children)[2]

            # num_siblings is shape (batch_size x max_tree_size x 1)
            num_siblings = tf.cast(
                tf.count_nonzero(children, axis=2, keep_dims=True),
                dtype=tf.float32
            )
            # num_siblings is shape (batch_size x max_tree_size x max_children + 1)
            num_siblings = tf.tile(
                num_siblings, [1, 1, max_children + 1], name='num_siblings'
            )
            # creates a mask of 1's and 0's where 1 means there is a child there
            # has shape (batch_size x max_tree_size x max_children + 1)
            mask = tf.concat(
                [tf.zeros((batch_size, max_tree_size, 1)),
                 tf.minimum(children, tf.ones(tf.shape(children)))],
                axis=2, name='mask'
            )

            # child indices for every tree (batch_size x max_tree_size x max_children + 1)
            child_indices = tf.multiply(tf.tile(
                tf.expand_dims(
                    tf.expand_dims(
                        tf.range(-1.0, tf.cast(max_children, tf.float32), 1.0, dtype=tf.float32),
                        axis=0
                    ),
                    axis=0
                ),
                [batch_size, max_tree_size, 1]
            ), mask, name='child_indices')

            # weights for every tree node in the case that num_siblings = 0
            # shape is (batch_size x max_tree_size x max_children + 1)
            singles = tf.concat(
                [tf.zeros((batch_size, max_tree_size, 1)),
                 tf.fill((batch_size, max_tree_size, 1), 0.5),
                 tf.zeros((batch_size, max_tree_size, max_children - 1))],
                axis=2, name='singles')

            # eta_r is shape (batch_size x max_tree_size x max_children + 1)
            return tf.where(
                tf.equal(num_siblings, 1.0),
                # avoid division by 0 when num_siblings == 1
                singles,
                # the normal case where num_siblings != 1
                tf.multiply((1.0 - t_coef), tf.divide(child_indices, num_siblings - 1.0)),
                name='coef_r'
            )

    def eta_l(self, children, coef_t, coef_r):
        """Compute weight matrix for how much each vector belongs to the 'left'"""
        with tf.name_scope('coef_l'):
            children = tf.cast(children, tf.float32)
            batch_size = tf.shape(children)[0]
            max_tree_size = tf.shape(children)[1]
            # creates a mask of 1's and 0's where 1 means there is a child there
            # has shape (batch_size x max_tree_size x max_children + 1)
            mask = tf.concat(
                [tf.zeros((batch_size, max_tree_size, 1)),
                 tf.minimum(children, tf.ones(tf.shape(children)))],
                axis=2,
                name='mask'
            )

            # eta_l is shape (batch_size x max_tree_size x max_children + 1)
            return tf.multiply(
                tf.multiply((1.0 - coef_t), (1.0 - coef_r)), mask, name='coef_l'
            )


    def children_tensor(self, nodes, children, feature_size):
        """Build the children tensor from the input nodes and child lookup."""
        with tf.name_scope('children_tensor'):
            max_children = tf.shape(children)[2]
            batch_size = tf.shape(nodes)[0]
            num_nodes = tf.shape(nodes)[1]

            # replace the root node with the zero vector so lookups for the 0th
            # vector return 0 instead of the root vector
            # zero_vecs is (batch_size, num_nodes, 1)
            zero_vecs = tf.zeros((batch_size, 1, feature_size))
            # vector_lookup is (batch_size x num_nodes x feature_size)
            vector_lookup = tf.concat([zero_vecs, nodes[:, 1:, :]], axis=1)
            # children is (batch_size x num_nodes x num_children x 1)
            children = tf.expand_dims(children, axis=3)
            # prepend the batch indices to the 4th dimension of children
            # batch_indices is (batch_size x 1 x 1 x 1)
            batch_indices = tf.reshape(tf.range(0, batch_size), (batch_size, 1, 1, 1))
            # batch_indices is (batch_size x num_nodes x num_children x 1)
            batch_indices = tf.tile(batch_indices, [1, num_nodes, max_children, 1])
            # children is (batch_size x num_nodes x num_children x 2)
            children = tf.concat([batch_indices, children], axis=3)
            # output will have shape (batch_size x num_nodes x num_children x feature_size)
            # NOTE: tf < 1.1 contains a bug that makes backprop not work for this!
            return tf.gather_nd(vector_lookup, children, name='children')

    def one_hot_action1(self, action1):
        one_hot_action1 = np.zeros(42)
        one_hot_action1[action1 - 1] = 1
        #one_hot_action1 = [one_hot_action1]
        return np.array(one_hot_action1)

    def one_hot_actionchsen(self,action2):
        one_hot_action2 = np.zeros(92)
        for i in action2:
            one_hot_action2[i] = 1
        return one_hot_action2