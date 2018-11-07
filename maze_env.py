import numpy as np
import time
import sys
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

import astEncoder
import prog
import example
import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

class Maze(object):


    actionSet = astEncoder.setActSet()

    def __init__(self):
        # super(Maze, self).__init__()
        # self.action_space = ['u', 'd', 'l', 'r']
        # self.n_actions = len(self.action_space)
        self.n_features = 2
        # self.title('maze')
        # self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_H * UNIT))

        # actionTuple[0] = nodeNum  actionTuple[1] = actionType
        self.action_space = spaces.Tuple((spaces.Discrete(49), spaces.Discrete(50)))
        #self.observation_space = spaces.Box(low=0.0, high=20.0, shape=(300,))
        #self.action_space = spaces.Discrete(85)
        self.observation_space = spaces.Box(low=0.0, high=20.0, shape=(300,), dtype = np.int)
        self.seed()
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None
        self.numIll = 0
        self.numlegalButwrong=0
        self.numlega=0
        self.badaction = 0
        self.fitness = 0
        # self._build_maze()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action, index):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        self.fitness_ = []
        candidate = self.candidate_[index]
        fitness = example.get_fitness(candidate)
        print("fitness value:", fitness)
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
            '''
            if newfitnessValue > 30:
                reward = 0.02 * (self.fitness - oldfitnessValue)
            if newfitnessValue > 40:
                reward = 0.02 * (self.fitness - oldfitnessValue)
            if newfitnessValue > 50:
                reward = 0.02 * (self.fitness - oldfitnessValue)
            if newfitnessValue > 60:
                reward = 0.02 * (self.fitness - oldfitnessValue)
            '''
            if newfitnessValue > 69:
                print(newfitnessValue)
                reward = 0.1 * (self.fitness - oldfitnessValue)

        done = bool(self.fitness > 78.4)
        if done:
            reward = 5
            print("self.badaction", self.badaction)
        if not done:
            reward = reward - 0.05

        return reward, done, self

        # return np.array(self.state), reward, done, self

    def reset(self):
        self.state = spaces.Box(low=0, high=20, shape=(300,),dtype=int)
        # self.state = self.np_random.uniform(low=0, high=20, size=(3,))
        # candidate = prog.initProg()
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

        '''    
        ast = astEncoder.getAstDict()
        state, astActNodes = astEncoder.astEncoder(ast)
        self.ast = ast
        # self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.state = tuple(state)
        self.steps_beyond_done = None
        self.candidate = candidate
        self.astActNodes = astActNodes
        
        
        return np.array(self.state), self
        '''
        return self