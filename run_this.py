from maze_env import Maze
from RL_brain import DeepQNetwork
import astEncoder
import example
import StringIO
import time
import os
import numpy as np
import prog
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
    step = 0
    step_good = 0
    done = False
    info_ = env.reset()
    for episode in range(2000):
        # initial observation
        # observation, info_ = env.reset()
        # 100 candidate in actIndex
        reward_cum = 0
        start = time.time()
        candidate_max = info_.maxCandidate
        info_ = env.reset()
        if episode > 0:
            info_ = env.reset_1(info_, candidate_max)

        for t in range(100):

            # 100 candidates
            for index in range(candidate_num):
                observation_ = info_.state_[index]

                #observation = np.append(observation_, action1 * action1 /10000.0000)

                fitness = example.get_fitness(info_.candidate_[index])

                #observation = np.append(observation,  pow(fitness / 100.00, 2))
                if observation_[action1 - 1] != 0 and observation_[action1 - 1] != -1:
                    action2set = np.array(RL.action2set(example.getLegalAction2(info_.candidate_[index], action1)))
                else:
                    action2set = []

                one_hot_action2 = RL.one_hot_action2(action2set)
                one_hot_action1 = RL.one_hot_action1(action1)
                observation_store = np.append(np.append(np.append(observation_, one_hot_action1), one_hot_action2),  pow(fitness / 100.00, 2))

                # RL choose action based on observation
                action, action_value, action_store = RL.choose_action(observation_store, info_.candidate_[index], action1)
                if action_store < 42:
                    action1 = action
                    observation_store_ = observation_store
                    #observation_ = observation
                    #observation_[-2] = action1 * action1 / 10000.0000
                    reward = 0
                else:
                    action2 = action
                    '''
                    action2set = np.array(RL.action2set(example.getLegalAction2(info_.candidate_[index], action1)))
                    not_choosed_fit = []

                   
                    for index_ in action2set:
                        action_operation = RL.getAction(action1, action2)
                        candidate = example.copyProgram(info_.candidate_[index])
                        candidate_ = prog.mutation(candidate, action_operation[0], action_operation[1])
                        fitness = example.get_fitness(candidate_)
                        not_choosed_fit.append(fitness)

                    '''
                    action_operation = RL.getAction(action1, action2)
                    reward, done, info_ = env.step(action_operation, index)
                    observation_1 = info_.state_[index]
                    observation_ = np.append(observation_1, action1 * action1 / 10000.0000)
                    one_hot_action1_ = RL.one_hot_action1(action1)

                    if observation_1[action1 -1] != 0 and observation_1[action1 -1] != -1:
                        action2set_ = np.array(RL.action2set(example.getLegalAction2(info_.candidate_[index], action1)))
                    else:
                        action2set_ = []
                    one_hot_action2_ = RL.one_hot_action2(action2set_)


                    fitness = example.get_fitness(info_.candidate_[index])


                    observation_ = np.append(observation_, pow(fitness / 100.00, 2))

                    observation_store_ = np.append(np.append(np.append(observation_1, one_hot_action1_), one_hot_action2_), pow(fitness / 100.00, 2))

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
                    reward = reward - 0.01 * spin_reward

                if reward > 0:
                    RL.store_transition_good(observation_store, action_store, reward, observation_store_)
                    reward_cum += reward
                    step_good += 1
                else:
                    RL.store_transition_bad(observation_store, action_store, reward, observation_store_)
                    reward_cum += reward

                RL.store_transition(observation_store, action_store, reward, observation_store_)
                step += 1

                if (step_good > 80) and (step % 5 == 0):
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
