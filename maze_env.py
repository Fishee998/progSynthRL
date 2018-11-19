import astEncoder
import prog
import example
from gym import spaces
from gym.utils import seeding
import numpy as np
import math
candidate_num = 1
class Maze(object):

    actionSet = astEncoder.setActSet()

    def __init__(self):
        self.n_features = 2
        # self.action_space = spaces.Tuple((spaces.Discrete(49), spaces.Discrete(50)))
        self.action_space = spaces.Discrete(92)
        self.observation_space = spaces.Box(low= -1.0, high=6501.0, shape=(44,), dtype=np.int)
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
        self.maxCandidate_ = []
        self.maxFitness = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action, index):
        assert self.action_space.contains(action[0] + action[1]), "%r (%s) invalid" % (action, type(action))
        self.fitness_ = []
        candidate = self.candidate_[index]

        fitness = example.get_fitness(candidate)

        candidate_ = prog.mutation(candidate, action[0], action[1])
        self.candidate_[index] = example.copyProgram(candidate_)

        illegal = example.illegal(candidate_)
        reward = 0
        if illegal == 1:
            self.illegal_action += 1
            reward = -5
        else:
            self.legal_action += 1
            oldfitnessValue = fitness
            state_0 = self.getstate(candidate_)
            self.state_[index] = np.array(tuple(state_0))
            self.steps_beyond_done = None
            newfitnessValue = example.get_fitness(candidate_)

            '''
            if self.maxFitness < newfitnessValue:
                self.maxFitness = newfitnessValue
                if example.isNUll(self.maxCandidate) == 1:
                    example.freeAll(None, self.maxCandidate, None, None, None, 2)
                maxCandidate = example.copyProgram(candidate_)
                self.maxCandidate_.append(maxCandidate)
            '''

            self.fitness_.append(newfitnessValue)

            x = newfitnessValue - oldfitnessValue

            '''
            reward = x * 0.05
            if self.maxFitness < 60:
                if newfitnessValue > 60:
                    reward = 1 + x * 0.05
            else:
                if newfitnessValue > 70:
                    reward = 1 + x * 0.05

            '''
            if x > 0:
                if newfitnessValue > 69:
                    reward = 0.9 * x
                else:
                    if newfitnessValue > 60:
                        reward = 0.7 * x
                    else:
                        if newfitnessValue > 50:
                            reward = 0.5 * x
                        else:
                            if newfitnessValue > 40:
                                reward = 0.3 * x
                            else:
                                if newfitnessValue > 30:
                                    reward = 0.1 * x
                                else:
                                    reward = 0.05 * x
            else :
                if x < 0:
                    if newfitnessValue < 30:
                        reward = 0.9 * x
                    else:
                        if newfitnessValue < 40:
                            reward = 0.7 * x
                        else:
                            if newfitnessValue < 50:
                                reward = 0.5 * x
                            else:
                                if newfitnessValue < 60:
                                    reward = 0.3 * x
                                else:
                                    if newfitnessValue < 69:
                                        reward = 0.1 * x
                                    else:
                                        reward = 0.05 * x
                else:
                    reward = 0

        if reward > 0:
            reward = math.log(reward/10.0000)
        else:
            if reward < 0:
                reward = -math.log(-reward/10.000)


        if newfitnessValue > 78.4:
            print("???")
        done = bool(newfitnessValue > 78.4)
        if done:
            reward = 10
            print("done")

        '''
        if not done:
            reward = reward - 0.1
        '''

        # if illegal != 1:
        if reward > 0:
            reward = math.log(reward)
        else:
            if reward < 0:
                reward = -math.log(-reward)

        # print("action", action, "reward", reward)
        # print(newfitnessValue, "-", oldfitnessValue)
        return reward, done, self

    def reset(self):
        self.state = spaces.Box(low=-1.0, high=6501.0, shape=(42,), dtype=np.int)
        # self.state = spaces.Box(low=0, high=20, shape=(300,), dtype=int)
        # 100 candidates
        self.candidate_ = []
        self.state_ = []
        candidates = prog.initProg()
        for index in range(candidate_num):
            candidate = example.getCandidate(candidates, index)
            self.candidate_.append(candidate)
            state = self.getstate(candidate)
            self.state_.append(np.array(state))
            # self.astActNodes_.append(astActNodes)

        return self

    def reset_(self, candidate):
        self.state = spaces.Box(low=-1.0, high=6501.0, shape=(42,), dtype=np.int)
        # self.state = spaces.Box(low=0, high=20, shape=(300,), dtype=int)
        # 100 candidates
        example.freeAll(None, self.candidate_[0], None, None, None, 2)
        self.state_ = []
        self.candidate_[0] = example.copyProgram(candidate)
        state = self.getstate(candidate)
        self.state_.append(np.array(state))
        return self

    def getstate(self, candidate):
        vector = example.genVector(candidate)
        state = []
        for ind in range(42):
            if example.state_i(vector, ind) > 0:
                state.append(example.state_i(vector, ind) / 10000.0000)
            else:
                state.append(example.state_i(vector, ind))
        return state
