import astEncoder
import prog
import example
from gym import spaces
from gym.utils import seeding
import numpy as np
candidate_num = 100
class Maze(object):

    actionSet = astEncoder.setActSet()

    def __init__(self):
        self.n_features = 2
        # self.action_space = spaces.Tuple((spaces.Discrete(49), spaces.Discrete(50)))
        self.action_space = spaces.Discrete(92)
        self.observation_space = spaces.Box(low= -1.0, high=6501.0, shape=(43,), dtype=np.int)
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

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action, index):
        assert self.action_space.contains(action[0] + action[1]), "%r (%s) invalid" % (action, type(action))
        self.fitness_ = []
        candidate = self.candidate_[index]
        '''
        root = example.getroot(self.candidate_[index])
        example.printprog(root, 0, self.candidate_[index])
        '''
        root = example.getroot(self.candidate_[index])
        if example.judgeNULL(example.findNode(root, candidate, action[0])) == 1:
            print("null chnoe")
        fitness = example.get_fitness(candidate)
        # print("fitness value:", fitness)
        state_0 = []
        vector = example.genVector(candidate)
        for ind in range(42):
            state_0.append(example.state_i(vector, ind))

        candidate_ = prog.mutation(candidate, action[0], action[1])

        self.candidate_[index] = candidate_

        illegal = example.illegal(candidate_)
        if illegal == 1:
            '''
            state_ = []
            vector = example.genVector(candidate_)
            for ind in range(42):
                state_.append(example.state_i(vector, ind))
            '''
            self.illegal_action += 1
            reward = -5
        else:
            self.legal_action += 1
            oldfitnessValue = fitness
            vector = example.genVector(candidate_)
            state_ = []
            for ind in range(42):
                state_.append(example.state_i(vector, ind))
            self.state_[index] = np.array(tuple(state_))
            self.steps_beyond_done = None
            newfitnessValue = example.get_fitness(candidate_)
            self.fitness_.append(newfitnessValue)

            if newfitnessValue > 69:
                reward = 5
            else:
                if newfitnessValue > 60:
                    reward = 3
                else:
                    if newfitnessValue > 50:
                        reward = 1
                    else:
                        if newfitnessValue > 40:
                            reward = 0.5
                        else:
                            if newfitnessValue > 30:
                                reward = 0.3
                            else:
                                if newfitnessValue > 20:
                                    reward = 0.1
                                else:
                                    reward = 0.05
            '''
            else :
                if newfitnessValue < oldfitnessValue:
                    if newfitnessValue < 30:
                        reward = -1
                    else:
                        if newfitnessValue < 40:
                            reward = -0.8
                        else:
                            if newfitnessValue < 50:
                                reward = -0.6
                            else:
                                if newfitnessValue < 60:
                                    reward = -0.4
                                else:
                                    if newfitnessValue < 69:
                                        reward = -0.2
                                    else:
                                        reward = -0.05
                else:
                    reward = 0
            '''
        done = bool(self.fitness > 78.4)
        if done:
            reward = 2
            print("done")
        '''
        if not done:
            reward = reward - 1
        '''
        # if illegal != 1:
        print("action", action, "reward", reward)
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

            vector = example.genVector(candidate)
            state = []
            for ind in range(42):
                state.append(example.state_i(vector, ind))
            self.state_.append(np.array(state))
            # self.astActNodes_.append(astActNodes)

        return self