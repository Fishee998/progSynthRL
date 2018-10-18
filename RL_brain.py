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
            e_greedy_increment=0.00025,
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

    # action1 action2 legal
    def action2set(self, action2s):
        actionLen = example.action2Len(action2s)
        action2Seleced = []
        for index in range(actionLen):
            action2Seleced.append(example.get_action2(action2s, index))
        return action2Seleced

    def getAction1(self, act1Set, action1_value):
        action1Set = np.nonzero(act1Set)[0]
        index = tf.Variable(action1Set)
        #action1_value = tf.Variable(action1_value)
        # action1_value = action1_value[0]
        # action1_value = tf.Variable(action1_value)
        # action1_value = tf.gather(action1_value, index)
        # action1_index = tf.argmax(action1_value)
        action1_index = tf.argmax(tf.gather(tf.Variable(action1_value[0]),index))
        # print(action1_value)
        # print(index)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            action1_index = sess.run(action1_index)
        action1 = action1Set[action1_index]
        return action1

    def getAction1_random(self, act1Set):
        act1Set1 = np.nonzero(act1Set)
        action1 = choice(act1Set1[0])
        return action1

    def getAction2(self, candidate, action1, action2_value):
        action2 = example.getLegalAction2(candidate, action1)
        action2set_ = self.action2set(action2)
        action2_ = []
        for index2 in range(50):
            action2_.append(0)
        for index in range(len(action2set_)):
            a = action2set_[index]
            action2_[a] = 1
        action2Set = np.nonzero(action2_)[0]
        index = tf.Variable(action2Set)
        # action1_value = tf.Variable(action1_value)
        # action1_value = action1_value[0]
        # action1_value = tf.Variable(action1_value)
        # action1_value = tf.gather(action1_value, index)
        # action1_index = tf.argmax(action1_value)
        action2_index = tf.argmax(tf.gather(tf.Variable(action2_value[0]), index))
        # print(index)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            action2_index = sess.run(action2_index)
        action2 = action2Set[action2_index]
        return action2

    def getAction2_random(self, candidate, action1):
        action2_ = example.getLegalAction2(candidate, action1)
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
        # self.epsilon = 0.8 * (0.993) ** episode

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value1 = self.sess.run(self.q_eval1, feed_dict={self.s: observation})
            actions_value2 = self.sess.run(self.q_eval2, feed_dict={self.s: observation})
            action1 = self.getAction1(act1Set, actions_value1)
            action2 = self.getAction2(candidate, action1, actions_value2)
            action = self.getAction(action1, action2)
        else:
            action1 = self.getAction1_random(act1Set)
            action2 = self.getAction2_random(candidate, action1)
            action = self.getAction(action1, action2)
        return action

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

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
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        q_target1[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next1, axis=1)
        q_target2[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next2, axis=1)

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
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()


