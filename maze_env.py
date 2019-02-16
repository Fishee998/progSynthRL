import astEncoder
import prog
import example
from gym import spaces
from gym.utils import seeding
import numpy as np
import math
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
        self.maxCandidate = None
        self.maxFitness = 0
        self.candidate = None
        self.state_spin = []
        self.candidate_spin = []
        self.candidates = []

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, index, action):
        assert self.action_space.contains(action[0] + action[1]), "%r (%s) invalid" % (action, type(action))
        # self.fitness = []
        candidate = self.candidates[index]

        fitness = example.get_fitness(candidate)

        candidate_ = prog.mutation(candidate, action[0], action[1])
        self.candidate = example.copyProgram(candidate_)

        illegal = example.illegal(candidate_)
        reward = 0
        if illegal == 1:
            self.illegal_action += 1
            reward = -5
        else:
            self.legal_action += 1
            oldfitnessValue = fitness
            state_0 = self.getstate(candidate_)
            self.state = np.array(tuple(state_0))
            self.steps_beyond_done = None
            newfitnessValue = example.get_fitness(candidate_)

            if self.maxFitness < newfitnessValue:
                self.maxFitness = newfitnessValue
                if example.isNUll(self.maxCandidate) == 1:
                    example.freeAll(None, self.maxCandidate, None, None, None, 2)
                maxCandidate = example.copyProgram(candidate_)
                self.maxCandidate = maxCandidate
                print('maxfitness:{maxfitness}'.format(maxfitness = self.maxFitness))
            self.fitness = newfitnessValue

            if newfitnessValue > 60:
                reward = 0.1 * (newfitnessValue - fitness)
            else:
                if newfitnessValue > 74:
                    reward = 0.5 * (newfitnessValue - fitness)
                else:
                    reward = -0.1

        spin_reward = 0
        if newfitnessValue > 79:
            reward = 0.1
            print("???")
            self.maxCandidate = None
            print(len(self.state_spin))
            if len(self.state_spin) > 0:
                for stae in self.state_spin:
                    if (self.state == stae).all() == False:
                        self.state_spin.append(self.state)
                        self.candidate_spin.append(example.copyProgram(self.candidate))
                    else:
                        reward = reward - 1
            else:
                self.state_spin.append(self.state)
                self.candidate_spin.append(example.copyProgram(self.candidate))
            self.spin_used = 1

            spin_reward = example.spin_(candidate_)
            if spin_reward == 4:
                print("liveness")
                reward = reward + 0.1
            else:
                if spin_reward == 10:
                    reward = reward + 0.2
                    print("safety")
                else:
                    if spin_reward == 20:
                        reward = 2
                        done = True
                    else:
                        if spin_reward == 8:
                            print("liveness1 liveness2")
                            reward = reward + 0.2
                        else:
                            reward = reward - 0.1
            
        done = bool(spin_reward == 20)
        if done:
            reward = 2
            print("done")

        # print('action: {action} fitness: {fitness}'.format(action=action, fitness=newfitnessValue))

        return reward, done, self

    def reset(self):
        self.state = spaces.Box(low=-1.0, high=6501.0, shape=(42,), dtype=np.int)
        # self.state = spaces.Box(low=0, high=20, shape=(300,), dtype=int)
        # 100 candidates
        # self.candidate_ = []
        # self.state_ = []
        candidates = prog.initProg()
        '''
        for index in range(candidate_num):
            candidate = example.getCandidate(candidates, index)
            self.candidate.append(candidate)
            state = self.getstate(candidate)
            self.state.append(np.array(state))
            # self.astActNodes_.append(astActNodes)
        '''

        candidate = example.getCandidate(candidates, 0)
        self.candidate = candidate
        self.state = np.array(self.getstate(candidate))
        self.fitness = example.get_fitness(candidate)
        self.candidates.append(self.candidate)

        return self

    def reset_(self, candidate):
        self.state = spaces.Box(low=-1.0, high=6501.0, shape=(42,), dtype=np.int)
        # self.state = spaces.Box(low=0, high=20, shape=(300,), dtype=int)
        # 100 candidates
        '''
        example.freeAll(None, self.candidate_[0], None, None, None, 2)
        self.state_ = []
        self.candidate_[0] = example.copyProgram(candidate)
        state = self.getstate(candidate)
        self.state_.append(np.array(state))
        '''
        self.candidates = []
        candidates = prog.initProg()
        candidate = example.getCandidate(candidates, 0)
        self.candidates.append(candidate)
        if self.maxCandidate != None:
            self.candidate = example.copyProgram(candidate)
            self.candidates.append(self.candidate)
        self.candidates.extend(self.candidate_spin)


        return self

    def reset_1(self, candidate):
        #example.freeAll(None, info_.candidate_[0], None, None, None, 2)
        self.candidate = example.copyProgram(candidate)
        self.state = np.array(info_.getstate(candidate))
        self.fitness = example.get_fitness(candidate) 
        #info_.state_[0] = np.array(state)
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
