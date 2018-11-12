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
        self.action_space = spaces.Discrete(85)
        self.observation_space = spaces.Box(low=0.0, high=35.0, shape=(301,), dtype=np.int)
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

    def step(self, real_action, action, index):
        assert self.action_space.contains(action[0] + action[1]), "%r (%s) invalid" % (action, type(action))
        self.fitness_ = []
        candidate = self.candidate_[index]
        fitness = example.get_fitness(candidate)
        # print("fitness value:", fitness)
        candidate_ = prog.mutation(candidate, real_action[0], real_action[1])
        self.candidate_[index] = candidate_
        illegal = example.illegal(candidate_)
        if illegal == 1:
            self.illegal_action += 1
            reward = -1
            # self.numIll =self.numIll + 1
        else:
            self.legal_action += 1
            oldfitnessValue = fitness
            ast = astEncoder.getAstDict()
            state_, astActNodes = astEncoder.astEncoder(ast)
            # print(state_)
            '''
            if state_ == self.state:
                # self.numlegalButwrong = self.numlegalButwrong + 1
                reward = -5
            '''
            self.ast_[index] = ast
            # self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
            self.state_[index] = np.array(tuple(state_))
            self.steps_beyond_done = None
            # self.candidate = newCandidate
            self.astActNodes_[index] = astActNodes
            # self.numlega = self.numlega + 1
            newfitnessValue = example.get_fitness(candidate_)
            self.fitness_.append(newfitnessValue)
            reward = 0

            if newfitnessValue > oldfitnessValue:
                if newfitnessValue > 69:
                    reward = 0.09 * (newfitnessValue - oldfitnessValue)
                else:
                    if newfitnessValue > 60:
                        reward = 0.08 * (newfitnessValue - oldfitnessValue)
                    else:
                        if newfitnessValue > 50:
                            reward = 0.07 * (newfitnessValue - oldfitnessValue)
                        else:
                            if newfitnessValue > 40:
                                reward = 0.06 * (newfitnessValue - oldfitnessValue)
                            else:
                                if newfitnessValue > 30:
                                    reward = 0.05 * (newfitnessValue - oldfitnessValue)
            else:
                if newfitnessValue < 30:
                    reward = 0.009 * (newfitnessValue - oldfitnessValue)
                else:
                    if newfitnessValue < 40:
                        reward = 0.008 * (newfitnessValue - oldfitnessValue)
                    else:
                        if newfitnessValue < 50:
                            reward = 0.007 * (newfitnessValue - oldfitnessValue)
                        else:
                            if newfitnessValue < 60:
                                reward = 0.006 * (newfitnessValue - oldfitnessValue)
                            else:
                                if newfitnessValue < 69:
                                    reward = 0.004 * (newfitnessValue - oldfitnessValue)


        done = bool(self.fitness > 78.4)
        if done:
            reward = 2
            print("done")
        '''
        if not done:
            reward = reward - 1
        '''
        print("action", action, "reward", reward)
        return reward, done, self

    def reset(self):
        self.state = spaces.Box(low=0, high=20, shape=(300,), dtype=int)
        # 100 candidates
        self.candidate_ = []
        self.ast_ = []
        self.state_ = []
        self.astActNodes_ = []
        candidates = prog.initProg()
        for index in range(candidate_num):
            candidate = example.getCandidate(candidates, index)
            self.candidate_.append(candidate)
            example.printAst(candidate)
            ast = astEncoder.getAstDict()
            self.ast_.append(ast)
            state, astActNodes = astEncoder.astEncoder(ast)
            self.state_.append(np.array(state))
            self.astActNodes_.append(astActNodes)

        return self