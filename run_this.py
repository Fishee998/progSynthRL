from maze_env import Maze
from RL_brain import DeepQNetwork
import astEncoder
import os
import example
import time
import StringIO
import pickle
import classifier.tbcnn.sampling as sampling
import numpy as np

os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda-9.0/lib64'
os.environ['CUDA_HOME'] = '/usr/local/cuda-9.0'
os.environ["CUDA_VISIBLE_DEVICES"]= '0'

candidate_num = 10
target_reward = 80
def run_maze():
    step = 0
    action1 = 1
    step_good = 0
    done = False
    buf = StringIO.StringIO()
    for episode in range(30000):
        start = time.time()
        # initial observation

        info_ = env.reset()
        # actIndex = astEncoder.setAction1s(info_)
        reward_cum = 0

        for t in range(300):

            # RL choose action based on observation
            for index in range(candidate_num):
                example.printAst(info_.candidate_[index])
                ast = astEncoder.getAstDict()
                nodes, children = sampling.gen_samples1(ast, embeddings, embed_lookup)
                observation_ = info_.state_[index]

                if observation_[action1 - 1] != 0 and observation_[action1 - 1] != -1:
                    action2set = np.array(RL.action2set(example.getLegalAction2(info_.candidate_[index], action1)))
                else:
                    action2set = []

                fitness = example.get_fitness(info_.candidate_[index])

                one_hot_action2 = RL.one_hot_action2(action2set)
                one_hot_action1 = RL.one_hot_action1(action1)
                # print(one_hot_action1.dtype)
                # print(one_hot_action1.shape)
                observation_store = np.append(np.append(np.append(observation_, one_hot_action1), one_hot_action2),
                                              pow(fitness / 100.00, 2))

                # RL choose action based on observation
                action, action_value, action_store = RL.choose_action(observation_store,  action1, nodes, children,one_hot_action1,info_.candidate_[index])

                if action_store < 42:
                    action1 = action
                    reward = 0
                    nodes_ = nodes
                    children_ = children
                    one_hot_action1_ = one_hot_action1
                else:
                    action2 = action
                    action_operation = RL.getAction(action1, action2)
                    reward, done, info_ = env.step(action_operation, index)
                    observation_1 = info_.state_[index]
                    observation_ = np.append(observation_1, action1 * action1 / 10000.0000)
                    one_hot_action1_ = RL.one_hot_action1(action1)

                    if observation_1[action1 - 1] != 0 and observation_1[action1 - 1] != -1:
                        action2set_ = np.array(RL.action2set(example.getLegalAction2(info_.candidate_[index], action1)))
                    else:
                        action2set_ = []
                    one_hot_action2_ = RL.one_hot_action2(action2set_)

                    fitness = example.get_fitness(info_.candidate_[index])

                    observation_ = np.append(observation_, pow(fitness / 100.00, 2))

                    observation_store_ = np.append(np.append(np.append(observation_1, one_hot_action1_), one_hot_action2_),
                                                   pow(fitness / 100.00, 2))

                    # RL take action and get next observation and reward
                    ast = astEncoder.getAstDict()
                    nodes_, children_ = sampling.gen_samples1(ast, embeddings, embed_lookup)

                    if info_.spin_used == 1:
                        print("i_ep", episode, " step:", t, " fitness", example.get_fitness(info_.candidate), "time", time.time() - start)
                    if done:
                        print("i_ep", episode, " step:", t, " fitness", example.get_fitness(info_.candidate), "time",
                              time.time() - start)
                        print("Congradulations!")
                        break


                if reward > 0:
                    step_good += 1
                    RL.store_transition_good(nodes, children, one_hot_action1, action_store, reward, nodes_, children_, one_hot_action1_)

                RL.store_transition(nodes, children, one_hot_action1, action_store, reward, nodes_, children_,
                                             one_hot_action1_)


                reward_cum += reward


                if (step_good > 100) and (step % 2 == 0):
                    RL.learn()

                # swap observation
                observation = observation_
                nodes = nodes_
                children = children_
                # print(observation)

                # break while loop when end of this episode
                if reward == 100:
                    break
                step += 1
        print("episode", episode, "reward_cum", reward_cum)
    # end of game
    print('game over')
    env.destroy()


if __name__ == "__main__":
    # maze game
    env = Maze()
    with open('./vectors.pkl', 'rb') as fh:
        embeddings, embed_lookup = pickle.load(fh)
        num_feats = len(embeddings[0])
    RL = DeepQNetwork(env.action_space.n,
                      # env.action_space.spaces[1].n,
                      10,
                      learning_rate=0.01,
                      reward_decay=0.8,
                      e_greedy=0.9,
                      replace_target_iter=10,
                      memory_size=100,
                      # output_graph=True
                      )
    run_maze()
