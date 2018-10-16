"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import astEncoder
import prog
import example
import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

class CartPoleEnv(gym.Env):

    actionSet = astEncoder.setActSet()
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        # actionTuple[0] = nodeNum  actionTuple[1] = actionType
        self.action_space = spaces.Tuple((spaces.Discrete(35), spaces.Discrete(49)))
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

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        reward = 0
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state
        newfitnessValue = oldfitnessValue = example.get_fitness(self.candidate)
        newCandidate = prog.mutation(self.candidate, action[0], action[1])
        illegal = example.illegal(newCandidate)
        if illegal == 1:
            reward = -10
            # self.numIll =self.numIll + 1
        else:
            ast = astEncoder.getAstDict()
            state_, astActNodes = astEncoder.astEncoder(ast)
            if state_ == self.state:
                # self.numlegalButwrong = self.numlegalButwrong + 1
                reward = -5
            self.ast = ast
            # self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
            self.state =  tuple(state_)
            self.steps_beyond_done = None
            self.candidate = newCandidate
            self.astActNodes = astActNodes
            self.fitness = newfitnessValue
            # self.numlega = self.numlega + 1
            newfitnessValue = example.get_fitness(newCandidate)
            # print(newfitnessValue)
            if newfitnessValue > 80:
                print newfitnessValue
            # reward = newfitnessValue - oldfitnessValue
            '''
            if oldfitnessValue > 20:
                if newfitnessValue > 29:
                    reward = 1
            if oldfitnessValue > 30:
                if newfitnessValue > 40:
                    reward = 1
            if oldfitnessValue > 40:
                if newfitnessValue > 50:
                    reward = 2
            if oldfitnessValue > 50:
                if newfitnessValue > 60:
                    reward = 2
            if newfitnessValue > 30:
                reward = 1
            if newfitnessValue > 40:
                reward = 2
            if newfitnessValue > 50:
                reward = 3
            if newfitnessValue > 59:
                reward = 4
            if newfitnessValue > 70:
                reward = 10
            if newfitnessValue > 75:
                reward = 10
            '''
            if newfitnessValue == oldfitnessValue:
                self.badaction = self.badaction +1
                # print(self.badaction)
            reward = 0
            if newfitnessValue > 30:
                reward = 0.02 * (newfitnessValue - oldfitnessValue)
            if newfitnessValue > 40:
                reward = 0.02 * (newfitnessValue - oldfitnessValue)
            if newfitnessValue > 50:
                reward = 0.02 * (newfitnessValue - oldfitnessValue)
            if newfitnessValue > 60:
                reward = 0.02 * (newfitnessValue - oldfitnessValue)

            #if oldfitnessValue < 40:
             #   if newfitnessValue > 50:
                   # print(newfitnessValue)
              #     reward = reward + 0.1 * (newfitnessValue - oldfitnessValue)
            #if oldfitnessValue < 50:
             #   if newfitnessValue > 60:
              #      reward = reward + 0.1 * (newfitnessValue - oldfitnessValue)
            if newfitnessValue > 69:
                print(newfitnessValue)
                # reward = 10
                reward = 0.1 * (newfitnessValue - oldfitnessValue)

        done = bool(newfitnessValue  > 78.4)
        if done:
            print(self.badaction)
            reward = 5
        if not done :
            reward = reward - 0.05
        return np.array(self.state), reward, done, self

    def reset(self):
        self.state = spaces.Box(low=0, high=20, shape=(300,),dtype=int)
        # self.state = self.np_random.uniform(low=0, high=20, size=(3,))
        candidate = prog.initProg()
        ast = astEncoder.getAstDict()
        state, astActNodes = astEncoder.astEncoder(ast)
        self.ast = ast
        # self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.state = tuple(state)
        self.steps_beyond_done = None
        self.candidate = candidate
        self.astActNodes = astActNodes
        # return self.state, self
        return np.array(self.state), self

