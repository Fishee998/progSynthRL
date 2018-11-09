"""
This part of code is the DQN brain, which is a brain of the agent.
All decisions are made in here.
Using Tensorflow to build the neural network.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.7.3
"""
from random import choice
import example
import numpy as np
import pandas as pd
import tensorflow as tf
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
            replace_target_iter=10,
            memory_size=100,
            batch_size=32,
            e_greedy_increment=0.00018,
            output_graph=False,
    ):
        self.n_actions1 = n_actions1
        self.n_actions2 = n_actions2
        self.n_features = n_features
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
        self.memory = np.zeros((self.memory_size, n_features * 2 + 3))
        self.memory_candidate = []
        self.memory_action1 = []


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
        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        self.q_target1 = tf.placeholder(tf.float32, [None, self.n_actions1], name='Q_target')
        self.q_target2 = tf.placeholder(tf.float32, [None, self.n_actions2], name='Q_target')  # for calculating loss
        with tf.variable_scope('eval_net'):
            # c_names(collections_names) are the collections to store variables
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 10, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2_1'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions1], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions1], initializer=b_initializer, collections=c_names)
                self.q_eval1 = tf.matmul(l1, w2) + b2

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2_2'):
                w2_2 = tf.get_variable('w2', [n_l1, self.n_actions2], initializer=w_initializer, collections=c_names)
                b2_2 = tf.get_variable('b2', [1, self.n_actions2], initializer=b_initializer, collections=c_names)
                self.q_eval2 = tf.matmul(l1, w2_2) + b2_2

        with tf.variable_scope('loss'):
            self.loss1 = tf.reduce_mean(tf.squared_difference(self.q_target1, self.q_eval1))
            self.loss2 = tf.reduce_mean(tf.squared_difference(self.q_target2, self.q_eval2))
            self.loss = self.loss1 + self.loss2

        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # input
        with tf.variable_scope('target_net'):
            # c_names(collections_names) are the collections to store variables
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2_1'):
                w2_1 = tf.get_variable('w2', [n_l1, self.n_actions1], initializer=w_initializer, collections=c_names)
                b2_1 = tf.get_variable('b2', [1, self.n_actions1], initializer=b_initializer, collections=c_names)
                self.q_next1 = tf.matmul(l1, w2_1) + b2_1

            with tf.variable_scope('l2_2'):
                w2_2 = tf.get_variable('w2', [n_l1, self.n_actions2], initializer=w_initializer, collections=c_names)
                b2_2 = tf.get_variable('b2', [1, self.n_actions2], initializer=b_initializer, collections=c_names)
                self.q_next2 = tf.matmul(l1, w2_2) + b2_2

    def store_transition(self, s, a1, a2, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a1, a2, r], s_))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1

    def store_action1(self, act1_index_store):
        if not hasattr(self, 'action1_counter'):
            self.action1_counter = 0

        if self.action1_counter < self.memory_size:
            self.memory_action1.append(act1_index_store)
        else:
            # replace the old memory with new memory
            index = self.action1_counter % self.memory_size
            self.memory_action1[index] = act1_index_store

        self.action1_counter += 1

    def store_candidate(self, candidate):
        if not hasattr(self, 'candidate_counter'):
            self.candidate_counter = 0

        if self.candidate_counter < self.memory_size:
            self.memory_candidate.append(candidate)
        else:
            # replace the old memory with new memory
            index = self.candidate_counter % self.memory_size
            self.memory_candidate[index] = candidate

        self.candidate_counter += 1

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

    def getAction1_(self, act1Set, action1_value):
        action1Set = np.nonzero(act1Set)[0]
        action1_prob = []
        for index in action1Set:
            action1_prob.append(action1_value[index])
        prob_max = np.max(action1_prob, axis=0)
        action1 = np.argmax(action1_prob)
        action1 = action1Set[action1]
        real_action = act1Set[action1]
        return real_action, prob_max

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

        #action2  = np.argmax(action2_value[0])
        return action2

    def getAction2_(self, candidate, action1_real, action2_value):
        action2 = example.getLegalAction2(candidate, action1_real)
        action2set = self.action2set(action2)
        action2_prob = []
        for index in list(set(action2set)):
            action2_prob.append(action2_value[0][index])
        prob_action2 = np.max(action2_prob)
        return prob_action2

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

    def choose_action(self, observation, episode, act1Set, candidate):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]
        if np.random.uniform() < self.epsilon:
            actions_value1 = self.sess.run(self.q_eval1, feed_dict={self.s: observation})
            actions_value2 = self.sess.run(self.q_eval2, feed_dict={self.s: observation})
            action1, action1_real = self.getAction1(act1Set, actions_value1)
            action2 = self.getAction2(candidate, action1_real, actions_value2)
        else:
            action1, action1_real = self.getAction1_random(act1Set)
            action2 = self.getAction2_random(candidate, action1_real)
        action = self.getAction(action1, action2)
        action_real = self.getAction(action1_real, action2)
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
        batch_memory = self.memory[sample_index, :]
        act1set = [self.memory_action1[i] for i in sample_index]
        candidate = [self.memory_candidate[i] for i in sample_index]


        q_next1, q_eval1 = self.sess.run(
            [self.q_next1, self.q_eval1],
            feed_dict={
                self.s_: batch_memory[:, -self.n_features:],  # fixed params
                self.s: batch_memory[:, :self.n_features],  # newest params
            })

        q_next2, q_eval2 = self.sess.run(
            [self.q_next2, self.q_eval2],
            feed_dict={
                self.s_: batch_memory[:, -self.n_features:],  # fixed params
                self.s: batch_memory[:, :self.n_features],  # newest params
            })

        # change q_target w.r.t q_eval's action
        q_target1 = q_eval1.copy()
        q_target2 = q_eval2.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index1 = batch_memory[:, self.n_features].astype(int)
        eval_act_index2 = batch_memory[:, self.n_features + 1].astype(int)
        reward = batch_memory[:, self.n_features + 2]

        # effective action1
        action1_real_ = range(len(batch_index))
        q_next1_ = range(len(batch_index))
        for t in batch_index:
            action1_real_[t], q_next1_[t] = self.getAction1_(act1set[t], q_next1[t])

        # effective action2
        q_next2_ = range(len(batch_index))
        for t in batch_index:
            q_next2_[t] = self.getAction2_(candidate[t], action1_real_[t], q_next2)

        q_target1[batch_index, eval_act_index1] = reward + self.gamma * np.array(q_next1_)
        q_target2[batch_index, eval_act_index2] = reward + self.gamma * np.array(q_next2_)

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
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target1: q_target1,
                                                self.q_target2: q_target2})
        self.cost_his.append(self.cost)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max and self.learn_step_counter % 100 == 0 else self.epsilon_max
        self.learn_step_counter += 1





