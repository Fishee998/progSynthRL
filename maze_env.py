import astEncoder
import prog
import example
from gym import spaces
from gym.utils import seeding
import numpy as np

class Maze(object):

    actionSet = astEncoder.setActSet()

    def __init__(self):
        self.n_features = 2
        self.action_space = spaces.Tuple((spaces.Discrete(49), spaces.Discrete(50)))
        self.observation_space = spaces.Box(low=0.0, high=20.0, shape=(300,), dtype=np.int)
        self.seed()
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None
        self.numIll = 0
        self.numlegalButwrong = 0
        self.numlega = 0
        self.badaction = 0
        self.fitness = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action, index):

        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        self.fitness_ = []
        candidate = self.candidate_[index]
        fitness = example.get_fitness(candidate)
        # print("fitness value:", fitness)
        candidate_ = prog.mutation(candidate, action[0], action[1])
        self.candidate_[index] = candidate_
        illegal = example.illegal(candidate_)
        if illegal == 1:
            reward = -10
            # self.numIll =self.numIll + 1
        else:
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
            newfitnessValue = example.get_fitness(candidate)
            self.fitness_.append(newfitnessValue)
            reward = 0

            if newfitnessValue > oldfitnessValue:
                if newfitnessValue > 69:
                    reward = 0.005 * (newfitnessValue - oldfitnessValue)
                else:
                    if newfitnessValue > 60:
                        reward = 0.004 * (newfitnessValue - oldfitnessValue)
                    else:
                        if newfitnessValue > 50:
                            reward = 0.003 * (newfitnessValue - oldfitnessValue)
                        else:
                            if newfitnessValue > 40:
                                reward = 0.002 * (newfitnessValue - oldfitnessValue)
                            else:
                                if newfitnessValue > 30:
                                    reward = 0.001 * (newfitnessValue - oldfitnessValue)
            else:
                if newfitnessValue < 30:
                    reward = 0.004 * (newfitnessValue - oldfitnessValue)
                else:
                    if newfitnessValue < 40:
                        reward = 0.003 * (newfitnessValue - oldfitnessValue)
                    else:
                        if newfitnessValue < 50:
                            reward = 0.002 * (newfitnessValue - oldfitnessValue)
                        else:
                            if newfitnessValue < 60:
                                reward = 0.001 * (newfitnessValue - oldfitnessValue)
                            else:
                                if newfitnessValue < 69:
                                    reward = 0.0001 * (newfitnessValue - oldfitnessValue)

        done = bool(self.fitness > 78.4)
        if done:
            reward = 1
            print("done")
        if not done:
            reward = reward - 0.01

        return reward, done, self

    def reset(self):
        self.state = spaces.Box(low=0, high=20, shape=(300,), dtype=int)
        # 100 candidates
        self.candidate_ = []
        self.ast_ = []
        self.state_ = []
        self.astActNodes_ = []
        candidates = prog.initProg()
        for index in range(100):
            candidate = example.getCandidate(candidates, index)
            self.candidate_.append(candidate)
            example.printAst(candidate)
            ast = astEncoder.getAstDict()
            self.ast_.append(ast)
            state, astActNodes = astEncoder.astEncoder(ast)
            self.state_.append(np.array(state))
            self.astActNodes_.append(astActNodes)

        return self