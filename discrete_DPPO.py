"""
A simple version of OpenAI's Proximal Policy Optimization (PPO). [https://arxiv.org/abs/1707.06347]

Distributing workers in parallel to collect data, then stop worker's roll-out and train PPO on collected data.
Restart workers once PPO is updated.

The global PPO updating rule is adopted from DeepMind's paper (DPPO):
Emergence of Locomotion Behaviours in Rich Environments (Google Deepmind): [https://arxiv.org/abs/1707.02286]

View more on my tutorial website: https://morvanzhou.github.io/tutorials

Dependencies:
tensorflow 1.8.0
gym 0.9.2
"""
from __future__ import division
from RL_brain import DeepQNetwork
import cook

from maze_env import Maze
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym, threading, queue
import os
import math
import example
import pickle
import astEncoder
import sampling
import time


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


EP_MAX = 10000
EP_LEN = 100
N_WORKER = 1                # parallel workers
GAMMA = 0.8                 # reward discount factor
A_LR = 0.0001               # learning rate for actor
C_LR = 0.0001               # learning rate for critic
MIN_BATCH_SIZE = 32         # minimum batch size for updating PPO
UPDATE_STEP = 30            # loop update operation n-steps
EPSILON = 0.001            # for clipping surrogate objective
GAME = 'CartPole-v0'

env = Maze()
# env = gym.make(GAME)
S_DIM = env.observation_space.shape[0]
# S_DIM = 100
A_DIM = env.action_space.n

RL = DeepQNetwork(env.action_space.n,
                      # env.action_space.spaces[1].n,
                      env.observation_space.shape[0],
                      learning_rate=0.001,
                      reward_decay=0.8,
                      e_greedy=0.9,
                      replace_target_iter=10,
                      memory_size=100,
                      # output_graph=True
                      )

start = time.clock()

class PPONet(object):
    def __init__(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        # self.tfs = tf.placeholder(tf.float32, [None, S_DIM], 'state')
        self.n_features = 10
        self.nodes = tf.placeholder(tf.float32, shape=(None, None, self.n_features), name='nodes')
        self.children = tf.placeholder(tf.int32, shape=(None, None, None), name='children')
        self.action1 = tf.placeholder(tf.float32, shape=(None, 43), name='action1')
        self.children_tb = None
        self.vector_lookup_tb = None
        self.kkk = None

        # critic
        w_init = tf.random_normal_initializer(0., .1)

        conv2 = self.conv_layer(1, 100, self.nodes, self.children, self.n_features)
        pooling = self.pooling_layer(conv2)
        self.pooling2 = tf.concat((pooling, self.action1), axis=1)

        lc = tf.layers.dense(self.pooling2, 100, tf.nn.relu, kernel_initializer=w_init, name='lc')
        # lc = tf.layers.dense(self.tfs, 100, tf.nn.relu, kernel_initializer=w_init, name='lc')
        # lc = tf.layers.dense(self.tfs, 100, tf.nn.relu, kernel_initializer=w_init, name='lc')
        self.v = tf.layers.dense(lc, 1)
        self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
        self.advantage = self.tfdc_r - self.v
        self.closs = tf.reduce_mean(tf.square(self.advantage))
        self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)

        # actor
        self.pi, pi_params = self._build_anet('pi', trainable=True)
        oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)
        
        self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        self.tfa = tf.placeholder(tf.int32, [None, ], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')

        a_indices = tf.stack([tf.range(tf.shape(self.tfa)[0], dtype=tf.int32), self.tfa], axis=1)
        pi_prob = tf.gather_nd(params=self.pi, indices=a_indices)   # shape=(None, )
        oldpi_prob = tf.gather_nd(params=oldpi, indices=a_indices)  # shape=(None, )
        ratio = pi_prob/oldpi_prob
        surr = ratio * self.tfadv                       # surrogate loss

        self.aloss = -tf.reduce_mean(tf.minimum(        # clipped surrogate objective
            surr,
            tf.clip_by_value(ratio, 1. - EPSILON, 1. + EPSILON) * self.tfadv))

        self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)

        self.sess.run(tf.global_variables_initializer())
        with open('./vectors.pkl', 'rb') as fh:
            self.embeddings, self.embed_lookup = pickle.load(fh)
            # num_feats = len(embeddings[0])


    def update(self):
        global GLOBAL_UPDATE_COUNTER
        while not COORD.should_stop():
            if GLOBAL_EP < EP_MAX:
                UPDATE_EVENT.wait()                     # wait until get batch of data
                self.sess.run(self.update_oldpi_op)     # copy pi to old pi
                #x = QUEUE.get()
                # y = QUEUE.qsize()
                data = [QUEUE.get() for _ in range(QUEUE.qsize())]      # collect data from all workers
                data = np.vstack(data)
                S_DIM_ = S_DIM + 43
                ch, s, a1, a, r, p = data[:, :43], data[:, 43: S_DIM_], data[:, S_DIM_: S_DIM_ + 42], data[:, S_DIM_+42: S_DIM_ + 43].ravel(), data[:, -2:-1],  data[:, -1:]
                nodes = []
                children = []
                for cand in p:
                    example.printAstint(cand[0])
                    nodes_, children_ = sampling.gen_samplesint(self.embeddings)
                    nodes.append(nodes_)
                    children.append(children_)
                nodes, children = sampling._pad_batch(nodes, children)
                # input_ = self.sess.run(self.pooling2, {self.nodes: nodes, self.children: children})
                adv = self.sess.run(self.advantage, {self.nodes: nodes, self.children: children, self.action1: ch, self.tfdc_r: r})
                # update actor and critic in a update loop
                [self.sess.run(self.atrain_op, {self.nodes: nodes, self.children: children, self.action1: ch, self.tfa: a, self.tfadv: adv}) for _ in range(UPDATE_STEP)]
                [self.sess.run(self.ctrain_op, {self.nodes: nodes, self.children: children, self.action1: ch, self.tfdc_r: r}) for _ in range(UPDATE_STEP)]
                #[self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in range(UPDATE_STEP)]
                #[self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(UPDATE_STEP)]
                UPDATE_EVENT.clear()        # updating finished
                GLOBAL_UPDATE_COUNTER = 0   # reset counter
                ROLLING_EVENT.set()         # set roll-out available

    def _build_net(self):
        # ------------------ build target_net ------------------
        # self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # input
        self.nodes_ = tf.placeholder(tf.float32, shape=(None, None, self.n_features), name='nodes')
        self.children_ = tf.placeholder(tf.int32, shape=(None, None, None), name='children')
        self.action1_ = tf.placeholder(tf.float32, shape=(None, 135), name='action1_')
        with tf.variable_scope('target_net'):
            # c_names(collections_names) are the collections to store variables
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                conv2 = self.conv_layer(1, 100, self.nodes_, self.children_, self.n_features)
                self.pooling2 = pooling2 = self.pooling_layer(conv2)
                #self.pooling2 = pooling2 = tf.concat((pooling2, self.action1_), axis=1)


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

    def _build_anet(self, name, trainable):

        with tf.variable_scope(name):
            conv2 = self.conv_layer(1, 100, self.nodes, self.children, self.n_features)
            self.pooling2 = self.pooling_layer(conv2)
            # self.pooling2 = pooling2 = tf.concat((pooling2, self.action1_), axis=1)
            l_a = tf.layers.dense(self.pooling2, 64, tf.nn.relu, trainable=trainable)
            # l_a = tf.layers.dense(self.tfs, 100, tf.nn.relu, trainable=trainable)
            a_prob = tf.layers.dense(l_a, A_DIM, tf.nn.softmax, trainable=trainable)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return a_prob, params

    def choose_action(self, s, candidate, action1):  # run by a local
        prob_weights = self.sess.run(self.pi, feed_dict={self.tfs: s[None, :]})
        # print(prob_weights)

    def sample(self, prog, vectors):
        nodes = []
        children = []
        child = []
        first = 0
        length = example.get_action2(prog, 0)
        prog_index = 1
        while prog_index < length:
            temp = example.get_action2(prog, prog_index)
            if temp != -1:
                if first == 0:
                    nodes.append(vectors[temp])
                    first = 1
                else:
                    if temp != -2:
                        child.append(temp)
            else:
                children.append(child)
                child = []
                first = 0
            prog_index = prog_index + 1
        return nodes, children

    def choose_action(self,nodes, children, action1chosen,  s, candidate, action1):  # run by a local
        # prob_weights = self.sess.run(self.pi, feed_dict={self.tfs: s[None, :]})
        prob_weights = self.sess.run(self.pi, feed_dict={self.nodes: nodes, self.children: children, self.action1: action1chosen})
        # prob_weights = self.sess.run(self.pi, feed_dict={self.nodes_: nodes, self.children_: children})
        a = prob_weights.shape[1]
        b = prob_weights.ravel()
        observation = s[np.newaxis, :]
        action_value = action1
        legalAction = RL.getLegalAction_prob(candidate, observation[0][:-43], action1)

        for prob_index in range(92):
            if prob_index not in legalAction:
                b[prob_index] = 0

        sum_prob = np.sum(b)

        for prob_index in range(92):
            b[prob_index] = b[prob_index] / sum_prob

        # action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # select action w.r.t the actions prob
        action = np.random.choice(range(prob_weights.shape[1]), p = b)
        print(action)

        action_store = action
        if action < 42:
            # action = action + 1
            action_value = s[action]
        else:
            action_store = action
            action = action - 42
        # print(action_store)
        assert action_value != 0
        return action, action_value, action_store

    def get_v(self, nodes, children, action1):
        #if s.ndim < 2: s = s[np.newaxis, :]
        #return self.sess.run(self.v, {self.tfs: s})[0, 0]
        return self.sess.run(self.v, {self.nodes: nodes, self.children: children, self.action1: action1})[0, 0]

    '''
    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s})[0, 0]
        # return self.sess.run(self.v, {self.nodes_: nodes, self.children_: children})[0, 0]
    '''

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
                self.children_vectors11 = children_vectors = self.children_tensor(nodes, children, feature_size)
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
            # vector_lookup1 is (batch_size x num_nodes x feature_size)
            vector_lookup_tb = tf.concat([zero_vecs, nodes[:, 1:, :]], axis=1)
            # children is (batch_size x num_nodes x num_children x 1)
            children = tf.expand_dims(children, axis=3)
            # prepend the batch indices to the 4th dimension of children
            # batch_indices is (batch_size x 1 x 1 x 1)
            batch_indices = tf.reshape(tf.range(0, batch_size), (batch_size, 1, 1, 1))
            # batch_indices is (batch_size x num_nodes x num_children x 1)
            batch_indices = tf.tile(batch_indices, [1, num_nodes, max_children, 1])
            # children is (batch_size x num_nodes x num_children x 2)
            children_tb = tf.concat([batch_indices, children], axis=3)
            # output will have shape (batch_size x num_nodes x num_children x feature_size)
            # NOTE: tf < 1.1 contains a bug that makes backprop not work for this!
            return tf.gather_nd(vector_lookup_tb, children_tb, name='children_')

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



class Worker(object):
    def __init__(self, wid):
        self.wid = wid
        self.env = Maze()
        self.ppo = GLOBAL_PPO

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

    def one_hot_actionchsen(self, action2):
        one_hot_action2 = np.zeros(92)
        for i in action2:
            one_hot_action2[i] = 1
        return one_hot_action2

    def one_hot_action1(self, action1):
        one_hot_action1 = np.zeros(42)
        one_hot_action1[action1 - 1] = 1
        # one_hot_action1 = [one_hot_action1]
        return np.array(one_hot_action1)

    def action2set(self, action2s):
        # actionLen = example.getLength(action2s)
        action2Seleced = []
        index = 0
        while example.get_action2(action2s, index) != 100:
            action2Seleced.append(example.get_action2(action2s, index))
            index += 1
        return action2Seleced


    def work(self):
        global GLOBAL_EP, GLOBAL_RUNNING_R, GLOBAL_UPDATE_COUNTER
        while not COORD.should_stop():
            with open('./vectors.pkl', 'rb') as fh:
                embeddings, embed_lookup = pickle.load(fh)
                num_feats = len(embeddings[0])
            action1 = 1

            if GLOBAL_EP == 0:
                info_ = self.env.reset()
            else:
                maxCand = info_.maxCandidate
                info_ = self.env.reset_(maxCand)

            # s = RL.obs(info_, action1)

            ep_r = 0
            buffer_s, buffer_a1, buffer_a, buffer_r, buffer_candidate, buffer_ch = [], [], [], [], [], []
            for t in range(EP_LEN):
                if not ROLLING_EVENT.is_set():                  # while global PPO is updating
                    ROLLING_EVENT.wait()                        # wait until PPO is updated
                    buffer_s, buffer_a, buffer_r, buffer_a1, buffer_candidate, buffer_ch = [], [], [], [], [], []   # clear history buffer, use new policy to collect data
                for index_ in range(len(info_.candidates)):
                    candidate = example.copyProgram(info_.candidates[index_])
                    s = RL.obs(candidate, action1)
                    example.printAstint(candidate)
                    nodes, children = sampling.gen_samplesint(embeddings)
                    nodes_, children_ = self._pad_batch_(nodes, children)
                    fitness = example.get_fitness(candidate)

                    action_choosen = self.getChoosenActions(candidate, s, action1)
                    one_hot_actionchoosen = self.one_hot_actionchsen(action_choosen)
                    one_hot_action1 = self.one_hot_action1(action1)
                    one_hot_act1chsn = np.concatenate(
                        (one_hot_action1, [fitness / 100.00]), axis=0)

                    # one_hot_act1chsn: [?, 135]
                    action, action_value, action_store = self.ppo.choose_action(nodes_, children_, [one_hot_act1chsn], s, candidate, action1)
                    if action_store < 42:
                        action1 = action
                        s_ = RL.obs(info_.candidates[index_], action1)
                        r = -0.1
                        done = False
                        nodes_, children_ = self._pad_batch_(nodes, children)
                        action_choosen_ = self.getChoosenActions(candidate, s_, action1)
                        one_hot_actionchoosen_ = self.one_hot_actionchsen(action_choosen_)
                        one_hot_action1_ = self.one_hot_action1(action1)
                        one_hot_act1chsn_ = np.concatenate(
                            (one_hot_action1_, [fitness / 100.00]),
                            axis=0)
                    else:
                        action2 = action
                        a = RL.getAction(action1, action2)
                        r, done, info_ = self.env.step(index_, a)
                        if done:
                            print('time: {time:.1f}'.format(time=time.clock() - start))

                        info_.candidates[index_] = info_.candidate
                        s_ = RL.obs(info_.candidate, action1)
                        candidate_ = example.copyProgram(info_.candidates[index_])
                        example.printAstint(candidate_)
                        nodes, children = sampling.gen_samplesint(embeddings)
                        nodes_, children_ = self._pad_batch_(nodes, children)

                        action_choosen_ = self.getChoosenActions(candidate_, s_, action1)
                        one_hot_actionchoosen_ = self.one_hot_actionchsen(action_choosen_)
                        one_hot_action1_ = self.one_hot_action1(action1)
                        one_hot_act1chsn_ = np.concatenate(
                            (one_hot_action1_, [fitness / 100.00]),
                            axis=0)

                    buffer_candidate.append(candidate)
                    buffer_a1.append(one_hot_action1_)
                    buffer_s.append(s)
                    # a = action_store
                    buffer_a.append(action_store)
                    buffer_r.append(r)
                    buffer_ch.append(one_hot_act1chsn)
                    # s = s_
                    ep_r += r

                    GLOBAL_UPDATE_COUNTER += 1                      # count to minimum batch size, no need to wait other workers
                    if t == EP_LEN - 1 or GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE or done:
                        if done:
                            v_s_ = 0                                # end of episode
                        else:
                            v_s_ = self.ppo.get_v(nodes_, children_, [one_hot_act1chsn_])
                            # v_s_ = self.ppo.get_v(s_)

                        discounted_r = []                           # compute discounted reward
                        for r in buffer_r[::-1]:
                            v_s_ = r + GAMMA * v_s_
                            discounted_r.append(v_s_)
                        discounted_r.reverse()

                        bch = np.vstack(buffer_ch)
                        be = np.vstack(buffer_candidate)
                        ba1 = np.vstack(buffer_a1)
                        bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, None]
                        buffer_s, buffer_a, buffer_r, buffer_candidate, buffer_a1, buffer_ch = [], [], [], [], [], []
                        # QUEUE.put(np.hstack((bs, ba, br)))
                        QUEUE.put(np.hstack((bch, bs, ba1, ba, br, be)))           # put data in the queue
                        if GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE:
                            ROLLING_EVENT.clear()       # stop collecting data
                            UPDATE_EVENT.set()          # globalPPO update

                        if GLOBAL_EP >= EP_MAX:         # stop training
                            COORD.request_stop()
                            break

                        if done: break

            # record reward changes, plot later
            if len(GLOBAL_RUNNING_R) == 0: GLOBAL_RUNNING_R.append(ep_r)
            else: GLOBAL_RUNNING_R.append(GLOBAL_RUNNING_R[-1]*0.9+ep_r*0.1)
            GLOBAL_EP += 1

            print('{prob:.1f}% |W{b} |Ep_r: {c}'.format(prob=(GLOBAL_EP/EP_MAX*100), b=self.wid, c=ep_r))


if __name__ == '__main__':
    GLOBAL_PPO = PPONet()
    UPDATE_EVENT, ROLLING_EVENT = threading.Event(), threading.Event()
    UPDATE_EVENT.clear()            # not update now
    ROLLING_EVENT.set()             # start to roll out
    workers = [Worker(wid=i) for i in range(N_WORKER)]


    GLOBAL_UPDATE_COUNTER, GLOBAL_EP = 0, 0
    GLOBAL_RUNNING_R = []
    COORD = tf.train.Coordinator()
    QUEUE = queue.Queue()           # workers putting data in this queue
    threads = []
    for worker in workers:          # worker threads
        t = threading.Thread(target=worker.work, args=())
        t.start()                   # training
        threads.append(t)
    # add a PPO updating thread
    threads.append(threading.Thread(target=GLOBAL_PPO.update,))
    threads[-1].start()
    COORD.join(threads)

    '''
    # plot reward change and test
    plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
    plt.xlabel('Episode'); plt.ylabel('Moving reward'); plt.ion(); plt.show()
    env = gym.make('CartPole-v0')
    while True:
        s = env.reset()
        for t in range(1000):
            env.render()
            s, r, done, info = env.step(GLOBAL_PPO.choose_action(s))
            if done:
                break

    '''
