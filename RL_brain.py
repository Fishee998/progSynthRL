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
            n_actions2,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=0.01,
            output_graph=False,
    ):
        self.n_actions1 = n_actions1
        self.n_actions2 = n_actions2
        self.n_features = n_features
        self.ob_features = 300
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        # self.memory = np.zeros((self.memory_size, self.ob_features * 2 + 3))
        self.memory = []

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
        self.nodes_ = tf.placeholder(tf.float32, shape=(None, None, self.n_features), name='tree_')
        self.children_ = tf.placeholder(tf.int32, shape=(None, None, None), name='children_')
        with tf.variable_scope('target_net'):
            # c_names(collections_names) are the collections to store variables
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                conv2 = self.conv_layer(1, 100, self.nodes_, self.children_, self.n_features)
                pooling2 = self.pooling_layer(conv2)
                # w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                # b1 = tf.get_variable('b1', [1, n_
                #
                #
                # l1], initializer=b_initializer, collections=c_names)
                # l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2_1'):
                self.q_next1 = self.hidden_layer(pooling2, 100, self.n_actions1)
                # w2_1 = tf.get_variable('w2', [n_l1, self.n_actions1], initializer=w_initializer, collections=c_names)
                # b2_1 = tf.get_variable('b2', [1, self.n_actions1], initializer=b_initializer, collections=c_names)
                # self.q_next1 = tf.matmul(l1, w2_1) + b2_1

            with tf.variable_scope('l2_2'):
                self.q_next2 = self.hidden_layer(pooling2, 100, self.n_actions2)
                # w2_2 = tf.get_variable('w2', [n_l1, self.n_actions2], initializer=w_initializer, collections=c_names)
                # b2_2 = tf.get_variable('b2', [1, self.n_actions2], initializer=b_initializer, collections=c_names)
                # self.q_next2 = tf.matmul(l1, w2_2) + b2_2

        # ------------------ build evaluate_net ------------------
        # self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input

        self.nodes = tf.placeholder(tf.float32, shape=(None, None, self.n_features), name='tree')
        self.children = tf.placeholder(tf.int32, shape=(None, None, None), name='children')

        self.q_target1 = tf.placeholder(tf.float32, [None, self.n_actions1], name='Q_target')
        self.q_target2 = tf.placeholder(tf.float32, [None, self.n_actions2], name='Q_target')  # for calculating loss
        with tf.variable_scope('eval_net'):
            # c_names(collections_names) are the collections to store variables
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 10, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                conv1 = self.conv_layer(1, 100, self.nodes, self.children, self.n_features)
                pooling = self.pooling_layer(conv1)
                # w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                # b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                # l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2_1'):
                self.q_eval1 = self.hidden_layer(pooling, 100, self.n_actions1)
                # w2 = tf.get_variable('w2', [n_l1, self.n_actions1], initializer=w_initializer, collections=c_names)
                # b2 = tf.get_variable('b2', [1, self.n_actions1], initializer=b_initializer, collections=c_names)
                # self.q_eval1 = tf.matmul(l1, w2) + b2

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2_2'):
                self.q_eval2 = self.hidden_layer(pooling, 100, self.n_actions2)
                # w2_2 = tf.get_variable('w2', [n_l1, self.n_actions2], initializer=w_initializer, collections=c_names)
                # b2_2 = tf.get_variable('b2', [1, self.n_actions2], initializer=b_initializer, collections=c_names)
                # self.q_eval2 = tf.matmul(l1, w2_2) + b2_2

        with tf.variable_scope('loss'):
            self.loss1 = tf.reduce_mean(tf.squared_difference(self.q_target1, self.q_eval1))
            self.loss2 = tf.reduce_mean(tf.squared_difference(self.q_target2, self.q_eval2))
            self.loss = self.loss1 + self.loss2

        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

    def make_tfrecord(self):
        out_name = 'transition.tfrecord'
        self.tfrecord_wrt = tf.python_io.TFRecordWriter(out_name)
        # return tfrecord_wrt

    def store_transition(self, nodes, children, a1, a2, r, nodes_, children_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = {}
        # nodesShape = nodes.shape
        # childrenShape = children.shape
        # nodes_shape = nodes_.shape
        # children_shape = children_.shape
        # a = [a1, a2, r]
        # a_shape = a.shape

        # transition = np.hstack((s, [a1, a2, r], s_))

         #transition['nodes'] = tf.train.Feature(float_list = tf.train.FloatList(Value = nodes))
        # transition['children'] = tf.train.Feature(int64_list=tf.train.Int64List(value=list(childrenShape)))
        # transition['nodes_'] = tf.train.Feature(float_list = tf.train.FloatList(Value = nodes_))
        # transition['children_'] = tf.train.Feature(int64_list=tf.train.Int64List(value=list(children_shape)))
        # transition['acRe'] = tf.train.Feature(int64_list=tf.train.Int64List(value=list(a_shape)))

        transition['nodes'] = nodes
        transition['children'] = children
        transition['nodes_'] = nodes_
        transition['children_'] = children_
        transition['acRe'] = [a1, a2, r]

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        if self.memory_counter < self.memory_size:
            self.memory.append(transition)
        else:
            self.memory[index] = transition
        self.memory_counter += 1
        # self.memory.append(transition)
        # self.memory_counter += 1

        # exmp = tf.train.Example(features=tf.train.Features(feature=transition))
        # exmp_serial = exmp.SerializeToString()
        # self.tfrecord_wrt.write(exmp_serial)
        # self.tfrecord_wrt.close()
        # return tfrecord_wrt
    '''
    def get_observation(self, nodes, children):

        tree_tensor = self.sess.run(self.conv1,
                                    feed_dict={self.nodes: nodes, self.children: children})

        print(len(flatten(tree_tensor)[0].flatten()))
    '''

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

    def getAction1(self, act1Set, action1_value):
        action1Set = np.nonzero(act1Set)[0]
        action1_prob = []
        for index in action1Set:
            action1_prob.append(action1_value[0][index])
        action1 = np.argmax(action1_prob)
        action1 = action1Set[action1]
        real_action = act1Set[action1]
        return action1, real_action
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

    def choose_action(self, nodes, children1, act1Set, candidate):
        # to have batch dimension when feed into tf placeholder
        # observation = observation[np.newaxis, :]
        # self.epsilon = 0.8 * (0.993) ** episode

        # nodes, children1 = self.reshapeChildNodes(nodes, children1)
        # to have batch dimension when feed into tf placeholder

        if np.random.uniform() < self.epsilon:
            actions_value1 = self.sess.run(self.q_eval1, feed_dict={self.nodes: nodes, self.children: children1})
            actions_value2 = self.sess.run(self.q_eval2, feed_dict={self.nodes: nodes, self.children: children1})
            action1 = np.argmax(actions_value1)
            action2 = np.argmax(actions_value2)
            # action1, action1_real = self.getAction1(act1Set, actions_value1)
            # action2 = self.getAction2(candidate, action1_real, actions_value2)
        else:
            action1 = np.random.randint(0, 48)
            action2 = np.random.randint(0, 49)
            # action1, action1_real = self.getAction1_random(act1Set)
            # action2 = self.getAction2_random(candidate, action1_real)
        action_real = action = self.getAction(action1, action2)
        # action_real = self.getAction(action1_real, action2)
        return action, action_real

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            # print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)

        # batch_memory = self.memory[sample_index, :]

        batch_memory = []
        for index in range(len(sample_index)):
            batch_memory.append(self.memory[sample_index[index]])

        a = batch_memory[:]
        '''
        nodes = deque()
        #nodes2 = np.narray(batch_memory[:][0]['nodes'])
        children = deque()
        # children = []
        nodes_ = []
        children_ = []

        nodes_1 = batch_memory[:][0]['nodes_']
        children_1 = batch_memory[:][0]['children_']
        '''
        for i in range(len(a)):
            nodes = batch_memory[:][i]['nodes']
            children = batch_memory[:][i]['children']
            nodes_ = batch_memory[:][i]['nodes_']
            children_ = batch_memory[:][i]['children_']

            q_next1, q_eval1 = self.sess.run(
                [self.q_next1, self.q_eval1],
                feed_dict={
                    self.nodes_: nodes_,
                    self.children_: children_,
                    self.nodes: nodes,  # fixed params
                    self.children: children,
                   # newest params
                })

            '''
            q_next1 = self.sess.run(
                [self.q_next1],
                feed_dict={
                    self.nodes_: nodes_,
                    self.children_: children_,

                    # newest params
                })

            q_eval1 = self.sess.run(
                [self.q_eval1],
                feed_dict={
                    self.nodes: nodes,  # fixed params
                    self.children: children,
                    # newest params
                })
            '''

            q_next2, q_eval2 = self.sess.run(
                [self.q_next2, self.q_eval2],
                feed_dict={
                    self.nodes_: nodes_,
                    self.children_: children_,
                    self.nodes: nodes,  # fixed params
                    self.children: children,

                })

            # change q_target w.r.t q_eval's action
            #q_target1 = q_eval1.copy()
            #q_target2 = q_eval2.copy()

            q_target1 = q_eval1[:]
            q_target2 = q_eval2[:]

            batch_index = np.arange(self.batch_size, dtype=np.int32)
            batch_index = np.arange(1, dtype=np.int32)
            eval_act_index1 = batch_memory[:][0]['acRe'][0]
            eval_act_index2 = batch_memory[:][0]['acRe'][1]
            reward = batch_memory[:][0]['acRe'][-1]
            # eval_act_index = batch_memory[:, self.ob_features].astype(int)
            # reward = batch_memory[:, self.ob_features + 1]

            q_target1[batch_index, eval_act_index1] = reward + self.gamma * np.max(q_next1, axis=1)
            q_target2[batch_index, eval_act_index2] = reward + self.gamma * np.max(q_next2, axis=1)

            """
            For example in this batch I have 2 samples and 3 actions:
            q_eval =
            [[1, 2, 3],
             [4, 5, 6]]
    
            q_target = q_eval =
            [[1, 2, 3],
             [4, 5, 6]]
    
            Then change q_target with the real q_target value w.r.t the q_eval's action.
            For example in:
                sample 0, I took action 0, and the max q_target value is -1;
                sample 1, I took action 2, and the max q_target value is -2:
            q_target =
            [[-1, 2, 3],
             [4, 5, -2]]
    
            So the (q_target - q_eval) becomes:
            [[(-1)-(1), 0, 0],
             [0, 0, (-2)-(6)]]
    
            We then backpropagate this error w.r.t the corresponding action to network,
            leave other action as error=0 cause we didn't choose it.
            """

            # train eval network
            _, self.cost = self.sess.run([self._train_op, self.loss],
                                         feed_dict={self.nodes: nodes,  # fixed params
                                                    self.children: children,
                                                    self.q_target1: q_target1,
                                                    self.q_target2: q_target2})
            self.cost_his.append(self.cost)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1


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

            return tf.nn.tanh(tf.matmul(pooled, weights) + biases)


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
