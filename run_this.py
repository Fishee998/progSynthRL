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
    # observation, info_ = env.reset()

    illegal_action = 0
    legal_action = 0
    for episode in range(200):
        step = 0
        info_ = env.reset()
        # initial observation
        # observation, info_ = env.reset()
        # 100 candidate in actIndex
        actIndex = astEncoder.setAction1s(info_)
        reward_cum = 0
        start = time.time()
        illegal_action = info_.illegal_action
        legal_action = info_.legal_action
        for t in range(10):

            # 100 candidates
            for index in range(candidate_num):
                observation = info_.state_[index]
                observation = np.append(observation, 0)
                # RL choose action based on observation
                action, real_action = RL.choose_action(observation, episode, actIndex[index], info_.candidate_[index])
                print("real_action", real_action)
                print("action", action)
                observation[-1] = action[0]

                # RL take action and get next observation and reward
                reward, done, info_ = env.step(real_action, index)
                observation_ = info_.state_[index]
                observation_ = np.append(observation_, 0)
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

                act1_index_store = astEncoder.setAction1s_store(info_, index)
                action1s = np.nonzero(act1_index_store)[0]

                action2 = []
                for ind in action1s:
                    action2.append(
                        RL.action2set(example.getLegalAction2(info_.candidate_[index], act1_index_store[ind])))
                action2 = list(set(flatten(flatten(action2))))
                for ind in action1s:
                    observation_[-1] = ind
                    RL.store_transition(observation, action[1], reward, observation_)
                    step += 1
                    # RL.store_action1(act1_index_store)
                    RL.store_action2(action2)

                reward_cum += reward

                if (step > 101) and (step % 10 == 0):
                    RL.learn()

                if reward == 100:
                    print("congratulations. Correct program synthesised. ")
                    break

            actIndex = astEncoder.setAction1s(info_)

        end = time.time()

        print("illegal action", info_.illegal_action - illegal_action, "legal action", info_.legal_action - legal_action)
        print("episode: ", episode, "reward_cum: ", reward_cum, "time: ", end - start)
        print("legal action reward_cum", - (abs(reward_cum) - 10 * (info_.illegal_action - illegal_action)) )
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
