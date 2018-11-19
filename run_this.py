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

candidate_num = 1
target_reward = 80
def run_maze():
    buf = StringIO.StringIO()
    illegal_action = 0
    legal_action = 0
    action1 = 1
    step = 0
    done = False
    info_ = env.reset()
    for episode in range(2000):
        # initial observation
        # observation, info_ = env.reset()
        # 100 candidate in actIndex
        reward_cum = 0
        start = time.time()
        info_ = env.reset()
        # if episode > 0:
        #    info_ = env.reset_(info_.maxCandidate)
        for t in range(1000):

            # 100 candidates
            for index in range(candidate_num):
                observation_ = info_.state_[index]

                observation = np.append(observation_, action1 * action1 /10000.0000)

                fitness = example.get_fitness(info_.candidate_[index])

                observation = np.append(observation,  pow(fitness / 100.00, 2))

                # RL choose action based on observation
                action, action_value, action_store = RL.choose_action(observation, info_.candidate_[index], action1)
                if action_store < 42:
                    action1 = action
                    observation_ = observation
                    observation_[-2] = action1 * action1 / 10000.0000
                    reward = 0
                else:
                    action2 = action
                    action_operation = RL.getAction(action1, action2)
                    reward, done, info_ = env.step(action_operation, index)
                    observation_ = info_.state_[index]
                    observation_ = np.append(observation_, action1 * action1 / 10000.0000)
                    fitness = example.get_fitness(info_.candidate_[index])
                    observation_ = np.append(observation_, pow(fitness / 100.00, 2))

                    if done:
                        break

                if example.get_fitness(info_.candidate_[index]) > 78.4:
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
                      reward_decay=0.8,
                      e_greedy=0.9,
                      replace_target_iter=10,
                      memory_size=100,
                      # output_graph=True
                      )
    run_maze()
