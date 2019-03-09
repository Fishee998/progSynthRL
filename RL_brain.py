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
from maze_env import Maze
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
            #n_actions2,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=10,
            memory_size=100,
            batch_size=64,
            e_greedy_increment=0.001,
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
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max


        # total learning step
        self.learn_step_counter = 0
        self.rl = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))
        self.memory_bad = np.zeros((self.memory_size, n_features * 2 + 2))
        self.memory_good = np.zeros((self.memory_size, n_features * 2 + 2))

        self.memory_candidate = range(memory_size)
        self.memory_action1 = range(memory_size)
        self.memory_action2 = range(memory_size)

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
        # self.q_target2 = tf.placeholder(tf.float32, [None, self.n_actions2], name='Q_target')  # for calculating loss
        with tf.variable_scope('eval_net'):
            # c_names(collections_names) are the collections to store variables
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 64, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                self.w1 = w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                self.b1 = b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                self.l1 = l1 = tf.nn.sigmoid(tf.matmul(self.s, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2_1'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions1], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions1], initializer=b_initializer, collections=c_names)
                self.q_eval1 = tf.matmul(l1, w2) + b2

            '''
            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2_2'):
                w2_2 = tf.get_variable('w2', [n_l1, self.n_actions2], initializer=w_initializer, collections=c_names)
                b2_2 = tf.get_variable('b2', [1, self.n_actions2], initializer=b_initializer, collections=c_names)
                self.q_eval2 = tf.matmul(l1, w2_2) + b2_2
            '''

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target1, self.q_eval1))
            # self.loss2 = tf.reduce_mean(tf.squared_difference(self.q_target2, self.q_eval2))
            # self.loss = self.loss1 + self.loss2

        with tf.variable_scope('train'):
            self._train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
            # self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # input
        with tf.variable_scope('target_net'):
            # c_names(collections_names) are the collections to store variables
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.sigmoid(tf.matmul(self.s_, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2_1'):
                w2_1 = tf.get_variable('w2', [n_l1, self.n_actions1], initializer=w_initializer, collections=c_names)
                b2_1 = tf.get_variable('b2', [1, self.n_actions1], initializer=b_initializer, collections=c_names)
                self.q_next1 = tf.matmul(l1, w2_1) + b2_1

            '''
            with tf.variable_scope('l2_2'):
                w2_2 = tf.get_variable('w2', [n_l1, self.n_actions2], initializer=w_initializer, collections=c_names)
                b2_2 = tf.get_variable('b2', [1, self.n_actions2], initializer=b_initializer, collections=c_names)
                self.q_next2 = tf.matmul(l1, w2_2) + b2_2
            '''

    '''       
    def store_transition(self, s, a1, a2, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a1, a2, r], s_))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1
    '''

    def store_transition(self, s, a1, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a1, r], s_))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1


    def store_transition_good(self, s, a1, r, s_):
        if not hasattr(self, 'memory_counter_good'):
            self.memory_counter_good = 0

        transition = np.hstack((s, [a1, r], s_))

        # replace the old memory with new memory
        index = self.memory_counter_good % self.memory_size
        self.memory_good[index, :] = transition

        self.memory_counter_good += 1

    def store_transition_bad(self, s, a1, r, s_):
        if not hasattr(self, 'memory_counter_bad'):
            self.memory_counter_bad = 0

        transition = np.hstack((s, [a1, r], s_))

        # replace the old memory with new memory
        index = self.memory_counter_bad % self.memory_size
        self.memory_bad[index, :] = transition

        self.memory_counter_bad += 1


    def store_badtransition1(self, s, a1, r, s_):
        if not hasattr(self, 'memory_counter_bad1'):
            self.memory_counter_bad1 = 0

        transition = np.hstack((s, [a1, r], s_))

        # replace the old memory with new memory
        index = self.memory_counter_bad1 % self.memory_size
        self.memory_bad1[index, :] = transition

        self.memory_counter_bad1 += 1

    def store_badtransition2(self, s, a1, r, s_):
        if not hasattr(self, 'memory_counter_bad2'):
            self.memory_counter_bad2 = 0

        transition = np.hstack((s, [a1, r], s_))

        # replace the old memory with new memory
        index = self.memory_counter_bad2 % self.memory_size
        self.memory_bad2[index, :] = transition

        self.memory_counter_bad2 += 1

    def store_action1(self, act1_index_store):
        # if not hasattr(self, 'action1_counter'):
            # self.action1_counter = 0

        # replace the old memory with new memory
        index = (self.memory_counter - 1) % self.memory_size
        self.memory_action1[index] = act1_index_store

        # self.action1_counter += 1
    def store_candidate(self, candidate):
        # if not hasattr(self, 'candidate_counter'):
            # self.candidate_counter = 0

        index = (self.memory_counter - 1) % self.memory_size
        self.memory_candidate[index] = example.copyProgram(candidate)

    def store_action2(self, action2):
        index = (self.memory_counter - 1) % self.memory_size
        self.memory_action2[index] = action2

    # action1 action2 legal
    def action2set(self, action2s):
        # actionLen = example.getLength(action2s)
        action2Seleced = []
        index = 0
        while example.get_action2(action2s, index) != 100:
            action2Seleced.append(example.get_action2(action2s, index))
            index += 1
        return action2Seleced

    def getLegalAction_prob(self, candidate):
        legalAction_c = example.legalAction(candidate)
        legalAction = []
        for i in range(3):
            legalAction.append(example.vector_i(legalAction_c, i))

        actions = []
        if legalAction[0] != 0:
            for index in range(legalAction[0]):
                actions.append(index)

        if legalAction[1] != 0:
            for index in range(legalAction[1]):
                actions.append(index + 132)

        if legalAction[2] != 0:
            for index in range(legalAction[2]):
                actions.append(index + 142)

        # legalAction = legalAction.flatten()
        return actions

    def getAction1(self,candidate, act1Set, action1, action1_value):
        action1s = np.nonzero(act1Set)[0]
        # act1Set[action1 - 1] != 0 and
        if act1Set[action1 - 1] != -1 and act1Set[action1 - 1] != 0:
            action2set = np.array(self.action2set(example.getLegalAction2(candidate, action1))) + 42
            # actions = action2set
            actions = np.append(action1s, action2set)
        else:
            actions = action1s

        actions = actions.flatten()
        action1_prob = []
        for index in actions:
            action1_prob.append(action1_value[0][index])
        action1 = np.argmax(action1_prob)
        action = actions[action1]
        return action
        # no restrict
        # action1 = np.argmax(action1_value[0])

    def getLegalAction(self, act1Set, action1_value):
        action1Set = np.nonzero(act1Set)[0]
        action1_prob = []
        for index in action1Set:
            action1_prob.append(action1_value[0][index])
        action = np.argmax(action1_prob)
        return action

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

    def getAction_random(self, candidate, act1Set, action1):
        # print("act1Set", act1Set)
        action1s = np.nonzero(act1Set)
        action1 = int(action1)
        # (act1Set[action1-1] != 0)
        if act1Set[action1-1] != 0 and act1Set[action1 - 1] != -1:
            # action2 = np.array(range(42, 92))
            action2set = np.array(self.action2set(example.getLegalAction2(candidate, action1))) + 42
            # actions = action2set
            actions = np.append(action1s, action2set)
        else:
            actions = action1s[0]
        actions = actions.flatten()
        # actions = np.append(action1s, action2set)
        action = random.choice(actions)

        #real_action = act1Set[action1]
        return action

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

    def getAction2_next(self, action2set, action2_value):
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

    def choose_action(self, observation, candidate, action1):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]
        action_value = action1
        if np.random.uniform() < self.epsilon:
            w1 = self.sess.run(self.w1, feed_dict={self.s: observation})
            l1 = self.sess.run(self.l1, feed_dict={self.s: observation})
            actions_value1 = self.sess.run(self.q_eval1, feed_dict={self.s: observation})
            actions = self.getAction1(candidate, observation[0][:-93], int(action1))
            self.rl = 1
        else:
            # print("choose by random")
            action = self.getAction_random(candidate, observation[0][:-93], int(action1))
            self.rl = 0
        action = int(action)
        action_store = action
        if action < 42:
            action = action + 1
            action_value = observation[0][action - 1]
        else:
            action_store = action
            action = action - 42

        return action, action_value, action_store

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            # print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index_random = np.random.choice(self.memory_size, size=self.batch_size - 48, replace=False)
            sample_index_good = np.random.choice(self.memory_size, size=self.batch_size - 16, replace=False)
            sample_index_bad = np.random.choice(self.memory_size, size=self.batch_size - 64, replace=False)
        else:
            #sample_index = np.random.choice(self.memory_counter, size=self.batch_size, replace=False)
            sample_index_random = np.random.choice(self.memory_counter, size=self.batch_size - 48, replace=False)
            sample_index_good = np.random.choice(self.memory_counter, size=self.batch_size - 16, replace=False)
            sample_index_bad = np.random.choice(self.memory_counter, size=self.batch_size - 64, replace=False)

        #batch_memory = self.memory[sample_index, :]

        batch_memory_random = self.memory[sample_index_random , :]
        batch_memory_good = self.memory_good[sample_index_good, :]
        batch_memory_bad = self.memory_bad[sample_index_bad, :]

        batch_memory = np.append(np.append(batch_memory_random, batch_memory_good, axis=0), batch_memory_bad, axis=0)

        q_next1, q_eval1 = self.sess.run(
            [self.q_next1, self.q_eval1],
            feed_dict={
                self.s_: batch_memory[:, -self.n_features:],  # fixed params
                self.s: batch_memory[:, :self.n_features],  # newest params
            })

        # change q_target w.r.t q_eval's action


        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index1 = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        x = np.max(q_next1, axis=1)

        sets = np.nonzero(batch_memory[:, -self.n_features:])
        '''
        if batch_memory[:,  -self.n_features:][eval_act_index1] != 0 and batch_memory[:,  -self.n_features:][eval_act_index1] != -1:
            action2set = np.array(self.action2set(example.getLegalAction2(candidate, action1))) + 42
            # actions = action2set
            actions = np.append(action1s, action2set)
            else:
                actions = action1s
        '''

        q_target1[batch_index, eval_act_index1] = reward + self.gamma * np.max(q_next1, axis=1)

        # train eval network
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target1: q_target1
                                                })
        self.cost_his.append(self.cost)
        # print("loss",self.cost_his[-1] )

        # increasing epsilon

        #self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max and self.learn_step_counter % 50 == 0 else self.epsilon

        if self.epsilon < self.epsilon_max:
            self.epsilon = self.epsilon + self.epsilon_increment if self.learn_step_counter %1 == 0 else self.epsilon
        else:
            self.epsilon = self.epsilon_max
        self.learn_step_counter += 1

        if self.learn_step_counter % 500 == 0:
            print("epsilon", self.epsilon)
            print("learn_step_counter", self.learn_step_counter)

    def one_hot_action1(self,action1):
        one_hot_action1 = np.zeros(40)
        one_hot_action1[action1] = 1
        return one_hot_action1

    def one_hot_action2(self,action2):
        one_hot_action2 = np.zeros(50)
        for i in action2:
            one_hot_action2[i] = 1
        return one_hot_action2

    def obs(self, candidate):
        observation_ = self.getstate(candidate)
        fitness = example.get_fitness(candidate)
        '''
        if observation_[action1 - 1] != 0 and observation_[action1 - 1] != -1:
            action2set = np.array(self.action2set(example.getLegalAction2(info_.candidate, action1)))
        else:
            action2set = []
        '''
        observation_store = np.append(observation_, fitness)
        return observation_store

    def getstate(self, candidate):
        # root = example.getroot(candidate)
        vector = example.genVector(candidate)
        state = []
        for ind in range(40):
            if example.state_i(vector, ind) > 0:
                # print("\n", example.state_i(vector, ind))
                state.append(example.state_i(vector, ind)/ 100.00 )
            else:
                state.append(example.state_i(vector, ind))
        return state