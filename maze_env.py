import astEncoder
import prog
import example
from gym import spaces
from gym.utils import seeding
import numpy as np
import math
candidate_num = 64
maxCandidate_num = 32
class Maze(object):

    actionSet = astEncoder.setActSet()

    def __init__(self):
        self.n_features = 2
        # self.action_space = spaces.Tuple((spaces.Discrete(49), spaces.Discrete(50)))
        self.action_space = spaces.Discrete(92)
        self.observation_space = spaces.Box(low= -1.0, high=6501.0, shape=(85,), dtype=np.int)
        self.seed()
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None
        self.numIll = 0
        self.numlegalButwrong = 0
        self.numlega = 0
        self.badaction = 0
        self.fitness = 0
        self.illegal_action = 0
        self.legal_action = 0
        self.maxCandidate = []
        self.maxFitness = 0
        self.minFitness = 0
        self.candidate = None


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action, index):
        assert self.action_space.contains(action[0] + action[1]), "%r (%s) invalid" % (action, type(action))
        # self.fitness = []
        candidate = self.candidate_[index]['prog']

        fitness = self.candidate_[index]['fit']

        candidate_ = prog.mutation(candidate, action[0], action[1])
        self.candidate_[index]['prog'] = example.copyProgram(candidate_)

        illegal = example.illegal(candidate_)
        reward = 0
        if illegal == 1:
            self.illegal_action += 1
            reward = -5
        else:
            self.legal_action += 1
            oldfitnessValue = fitness
            self.candidate_[index]['state'] = self.getstate(candidate_)
            self.steps_beyond_done = None
            self.candidate_[index]['fit'] = example.get_fitness(candidate_)
            newfitnessValue = example.get_fitness(candidate_)

            if len(self.maxCandidate) < maxCandidate_num:
                maxCandidate_ = {}
                maxCandidate_['prog'] = self.candidate_[index]['prog']
                maxCandidate_['fit'] = self.candidate_[index]['fit']
                maxCandidate_['state'] = self.candidate_[index]['state']
                self.maxCandidate.append(maxCandidate_)
                self.maxCandidate.sort(key=lambda k: (k.get('fit', 0)))
                self.minFitness = self.maxCandidate[0]['fit']
            else:
                if newfitnessValue > self.minFitness:
                    self.maxCandidate[0]['prog'] = self.candidate_[index]['prog']
                    self.maxCandidate[0]['fit'] = self.candidate_[index]['fit']
                    self.maxCandidate[0]['state'] = self.candidate_[index]['state']
                    self.maxCandidate.sort(key=lambda k: (k.get('fit', 0)))
                    self.minFitness = self.maxCandidate[0]['fit']


            # x = newfitnessValue - oldfitnessValue
            '''
            reward = 0.01 * x
            '''
            if newfitnessValue > 74:
                reward = 0.5
            else:
                if newfitnessValue > 64:
                    reward = 0.3
                else:
                    reward = 0.1
            '''
            if newfitnessValue < 74:
                reward = 0.
            else:
                if newfitnessValue < 64:
                    reward = 0.03
                else:
                    reward = 0.5
            
            
            if x > 0:
                if newfitnessValue > 74:
                    reward = 0.6 + 0.009 * x
                else:
                    if newfitnessValue > 69:
                        reward = 0.4 + 0.007 * x
                    else:
                        if newfitnessValue > 40:
                            reward = 0.05 + 0.005 * x
                        else:
                            reward = 0.03 + 0.003 * x
            else:
                if x < 0:
                    if newfitnessValue < 30:
                        reward = -0.3 + 0.009 * x
                    else:
                        if newfitnessValue < 40:
                            reward = -0.3 + 0.007 * x
                        else:
                            if newfitnessValue < 50:
                                reward = -0.3 + 0.005 * x
                            else:
                                if newfitnessValue < 60:
                                    reward = 0.003 * x
                                else:
                                    if newfitnessValue < 69:
                                        reward = 0.1 + 0.001 * x
                                    else:
                                        reward = 0.3 + 0.0005 * x
                else:
                    reward = 0
            '''
        spin_reward = 0
        if newfitnessValue > 78.4:
            reward = 1
            print("???")
            if len(self.spin) > 0:
                for stae in self.state_spin:
                    if (self.state == stae).all() == False:
                        self.state_spin.append(self.state)
                        self.candidate_spin.append(example.copyProgram(self.candidate))
                    else:
                        reward = reward - 1
            self.spin_used = 1
            spin_reward = example.spin_(candidate_)
            if spin_reward == 5:
                print("liveness")
            else:
                if spin_reward == 10:
                    print("safety")
                else:
                    if spin_reward == 20:
                        reward = 2
                        done = True

        done = bool(spin_reward == 20)
        if done:
            reward = 2
            print("done")

        # print('action: {action} fitness: {fitness} reward:{reward}'.format(action=action, fitness=newfitnessValue, reward=reward))

        return reward, done, self

    def reset(self):
        self.state = spaces.Box(low=-1.0, high=6501.0, shape=(42,), dtype=np.int)
        # 100 candidates
        self.candidate_ = []
        candidates = prog.initProg()
        for index in range(candidate_num):
            candidate_ = {}
            candidate = example.getCandidate(candidates, index)
            candidate_['prog'] = candidate
            candidate_['state'] = self.getstate(candidate)
            candidate_['fit'] = example.get_fitness(candidate)
            self.candidate_.append(candidate_)

        return self

    def reset_(self, max):
        self.state = spaces.Box(low=-1.0, high=6501.0, shape=(42,), dtype=np.int)
        # 100 candidates
        self.candidate_ = []
        candidates = prog.initProg()
        for index in range(candidate_num - maxCandidate_num):
            candidate_ = {}
            candidate = example.getCandidate(candidates, index)
            candidate_['prog'] = candidate
            candidate_['state'] = self.getstate(candidate)
            candidate_['fit'] = example.get_fitness(candidate)
            self.candidate_.append(candidate_)
        # self.candidate_.extend(max)

        return self


    def reset_1(self, info_, candidate):
        example.freeAll(None, info_.candidate_[0], None, None, None, 2)
        info_.candidate_[0] = example.copyProgram(candidate)
        state = info_.getstate(candidate)
        info_.state_[0] = np.array(state)
        return info_

    def getstate(self, candidate):
        vector = example.genVector(candidate)
        state = []
        for ind in range(42):
            if example.state_i(vector, ind) > 0:
                state.append(example.state_i(vector, ind) / 10000.0000)
            else:
                state.append(example.state_i(vector, ind))
        return np.array(state)
