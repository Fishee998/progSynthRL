from maze_env import Maze
from RL_brain import DeepQNetwork
import astEncoder
import example
import StringIO
import time
import os
import numpy as np
from compiler.ast import flatten

os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda-9.0/lib64'
os.environ['CUDA_HOME'] = '/usr/local/cuda-9.0'
os.environ["CUDA_VISIBLE_DEVICES"]= '0'

candidate_num = 100
target_reward = 80
def run_maze():
    buf = StringIO.StringIO()
    illegal_action = 0
    legal_action = 0
    action1 = 1
    action1_real = 0
    step = 0
    done = False
    for episode in range(2000):

        info_ = env.reset()
        # initial observation
        # observation, info_ = env.reset()
        # 100 candidate in actIndex
        # actIndex = astEncoder.setAction1s(info_)
        reward_cum = 0
        reward_cum_bad = 0
        start = time.time()
        for t in range(20):

            # 100 candidates
            for index in range(candidate_num):
                observation_ = info_.state_[index]
                observation = np.append(observation_, action1)

                # RL choose action based on observation
                if observation[-1] < 1:
                    print("observation[-1]", observation[-1])
                action, action_value, action_store = RL.choose_action(observation, info_.candidate_[index])
                if action_store < 42:
                    action1 = action
                    observation_ = observation
                    observation_[-1] = action1
                    reward = 0
                else:
                    action2 = action
                    action_operation = RL.getAction(action1, action2)
                    reward, done, info_ = env.step(action_operation, index)
                    observation_ = info_.state_[index]
                    observation_ = np.append(observation_, action1)

                '''
                action1_illegal_inAction2 = 0

                illegal_action1 = np.where(observation[:-1] == 0)[0]
                for illegal_action1_ in illegal_action1:
                    observation[-1] = illegal_action1_
                    reward = -5
                    RL.store_badtransition1(observation, illegal_action1_, reward, observation)

                if action_store < 42:
                    action1 = action
                    if observation[action1 - 1] == 0:
                        bad_action1 += 1
                        reward = -1
                    else:
                        # action1_real = real_action
                        reward = 0
                    observation_ = observation
                    observation_[-1] = action1
                else:
                    if observation[action1 - 1] == 0:
                        bad_action2 += 1
                        reward = -5
                        observation_ = observation
                        action1_illegal_inAction2 = 1
                    else:
                        action2 = action
                        # real_action = RL.getAction(action1, action2)
                        # RL take action and get next observation and reward
                        action_operation = RL.getAction(action1, action2)
                        reward, done, info_ = env.step(action_operation, index)
                        observation_ = info_.state_[index]
                        observation_ = np.append(observation_, action1)
                '''

                if done:
                    reward = target_reward
                    fitness = example.get_fitness(info_.candidate_[index])
                    print("i_ep: ", episode, " step:", t, " fitness: ", fitness)
                    spin_reward = example.spin_(info_.candidate_[index])
                    print("Spin: ", "score: ", spin_reward)
                    if spin_reward == 20:
                        buf.write("correct program at i_ep %d: step:%d \n" % (episode, t))
                        fo = open("./correctProg.txt", "a+")
                        fo.write(buf.getvalue())
                        fo.close()
                    if spin_reward == 5:
                        print("Spin: liveness")
                    if spin_reward == 10:
                        print("Spin: safety")
                    reward = reward + spin_reward

                '''
                if action1_illegal_inAction2 == 1:
                    RL.store_badtransition1(observation, action_store, reward, observation_)
                else:
                    if reward == -5:
                        reward_cum_bad += reward
                        RL.store_badtransition2(observation, action_store, reward, observation_)
                    else:
                        reward_cum += reward
                '''
                RL.store_transition(observation, action_store, reward, observation_)
                reward_cum += reward
                step += 1

                if (step > 100) and (step % 5 == 0):
                    RL.learn()

                if reward == 100:
                    print("congratulations. Correct program synthesised. ")
                    break

        end = time.time()

        print("episode: ", episode, "reward_cum_total: ", reward_cum , "time: ", end - start)
        # print("legal action reward_cum",  (reward_cum + 5 * (info_.illegal_action - illegal_action) + 1 * bad_action1))
    # end of game
    print('game over')
    # env.destroy()


if __name__ == "__main__":
    # maze game
    env = Maze()
    RL = DeepQNetwork(env.action_space.n,
                      # env.action_space.spaces[1].n,
                      env.observation_space.shape[0],
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=10,
                      memory_size=100,
                      # output_graph=True
                      )
    run_maze()
