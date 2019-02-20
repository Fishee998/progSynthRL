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
import example

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
EP_MAX = 10000
EP_LEN = 100
N_WORKER = 1                # parallel workers
GAMMA = 0.8                 # reward discount factor
A_LR = 0.0001               # learning rate for actor
C_LR = 0.0001               # learning rate for critic
MIN_BATCH_SIZE = 64         # minimum batch size for updating PPO
UPDATE_STEP = 30            # loop update operation n-steps
EPSILON = 0.001            # for clipping surrogate objective
GAME = 'CartPole-v0'

env = Maze()
# env = gym.make(GAME)
S_DIM = env.observation_space.shape[0]
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


class PPONet(object):
    def __init__(self):
        config = tf.ConfigProto() 
        config.gpu_options.per_process_gpu_memory_fraction = 0.3
        self.sess = tf.Session(config=config)
        self.tfs = tf.placeholder(tf.float32, [None, S_DIM], 'state')
        self.epsilon = 0
        self.epsilon_max = 0.8
        self.epsilon_increment = 0.001
        self.learn_step_counter = 0
        # critic
        w_init = tf.random_normal_initializer(0., .1)
        lc = tf.layers.dense(self.tfs, 50, tf.nn.softmax, kernel_initializer=w_init, name='lc')
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

    def update(self):
        global GLOBAL_UPDATE_COUNTER
        while not COORD.should_stop():
            if GLOBAL_EP < EP_MAX:
                UPDATE_EVENT.wait()                     # wait until get batch of data
                self.sess.run(self.update_oldpi_op)     # copy pi to old pi
                data = [QUEUE.get() for _ in range(QUEUE.qsize())]      # collect data from all workers
                data = np.vstack(data)
                s, a, r = data[:, :S_DIM], data[:, S_DIM: S_DIM + 1].ravel(), data[:, -1:]
                adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})
                # update actor and critic in a update loop
                [self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in range(UPDATE_STEP)]
                [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(UPDATE_STEP)]
                UPDATE_EVENT.clear()        # updating finished
                GLOBAL_UPDATE_COUNTER = 0   # reset counter
                ROLLING_EVENT.set()         # set roll-out available
                if self.epsilon < self.epsilon_max:
                    self.epsilon = self.epsilon + self.epsilon_increment if self.learn_step_counter % 1 == 0 else self.epsilon
                else:
                    self.epsilon = self.epsilon_max
                self.learn_step_counter += 1

    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):
            l_a = tf.layers.dense(self.tfs, 100, tf.nn.relu, trainable=trainable)
            a_prob = tf.layers.dense(l_a, A_DIM, tf.nn.softmax, trainable=trainable)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return a_prob, params

    def choose_action(self, s, candidate, action1):  # run by a local
        observation = s[np.newaxis, :]
        # print(observation)
        legalAction = RL.getLegalAction_prob(candidate, observation[0][:40], action1)
        # print(legalAction)
        action_value = action1
        if np.random.uniform() < self.epsilon:
            #print(s[None, :])
            prob_weights = self.sess.run(self.pi, feed_dict={self.tfs: s[None, :]})
            a = prob_weights.shape[1]
            b = prob_weights.ravel()

            for prob_index in range(67):
                if prob_index not in legalAction:
                    b[prob_index] = 0
            sum_prob = np.sum(b)
            for prob_index in range(67):
                b[prob_index] = b[prob_index] / sum_prob
            # action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # select action w.r.t the actions prob
            action = np.random.choice(range(prob_weights.shape[1]), p = b)
        else:
            action = np.random.choice(legalAction)
        action_store = action
        if action < 40:
            # action = action + 1
            action_value = s[action]
        else:

            action = action - 40
        assert action_value != 0
        return action, action_value, action_store

    
    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s})[0, 0]


class Worker(object):
    def __init__(self, wid):
        self.wid = wid
        self.env = Maze()
        self.ppo = GLOBAL_PPO

    def work(self):
        global GLOBAL_EP, GLOBAL_RUNNING_R, GLOBAL_UPDATE_COUNTER
        while not COORD.should_stop():
            action1 = 1
            # if GLOBAL_EP / 2 != 0:
            #    info_ = self.env.reset_(self.maxCandidate)
            # observation = state + act1 + legal act2 + fitValue
            # else:

            if GLOBAL_EP == 0:
                info_ = self.env.reset()
            else:
                maxCand = info_.maxCandidate
                info_ = self.env.reset_(maxCand)

            # s = RL.obs(info_, action1)

            ep_r = 0
            buffer_s, buffer_a, buffer_r = [], [], []
            for t in range(EP_LEN):
                if not ROLLING_EVENT.is_set():                  # while global PPO is updating
                    ROLLING_EVENT.wait()                        # wait until PPO is updated
                    buffer_s, buffer_a, buffer_r = [], [], []   # clear history buffer, use new policy to collect data
                for index_ in range(len(info_.candidates)):
                    candidate = info_.candidates[index_]
                    s = RL.obs(candidate, action1)
                    action, action_value, action_store = self.ppo.choose_action(s, candidate, action1)
                    # print('action:{action}'.format(action=action_store))
                    if action_store < 40:
                        action1 = action
                        s_ = RL.obs(info_.candidates[index_], action1)
                        r = -0.1
                        done = False
                    else:
                        action2 = action
                        a = RL.getAction(action1, action2)
                        r, done, info_ = self.env.step(index_, a)
                        info_.candidates[index_] = info_.candidate
                        s_ = RL.obs(info_.candidate, action1)

                    buffer_s.append(s)
                    # a = action_store
                    buffer_a.append(action_store)
                    buffer_r.append(r)
                    # s = s_
                    ep_r += r

                    GLOBAL_UPDATE_COUNTER += 1                      # count to minimum batch size, no need to wait other workers
                    if t == EP_LEN - 1 or GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE or done:
                        if done:
                            v_s_ = 0                                # end of episode
                        else:
                            v_s_ = self.ppo.get_v(s_)

                        discounted_r = []                           # compute discounted reward
                        for r in buffer_r[::-1]:
                            v_s_ = r + GAMMA * v_s_
                            discounted_r.append(v_s_)
                        discounted_r.reverse()

                        bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, None]
                        buffer_s, buffer_a, buffer_r = [], [], []
                        QUEUE.put(np.hstack((bs, ba, br)))          # put data in the queue
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
