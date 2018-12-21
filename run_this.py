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

os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda-10.0/lib64'
os.environ['CUDA_HOME'] = '/usr/local/cuda-10.0'
os.environ["CUDA_VISIBLE_DEVICES"]= '2'

candidate_num = 2
target_reward = 80
def run_maze():
    step = 0
    action1 = 1
    step_good = 0
    start = time.time()
    info_ = env.reset()
    for episode in range(600):
        # initial observation
        candidate_max = info_.maxCandidate
        candidate_spin = info_.candidate_spin
        state_spin = info_.state_spin
        info_ = env.reset()
        # actIndex = astEncoder.setAction1s(info_)
        reward_cum = 0
        if episode > 0:
            info_ = env.reset_1(info_, candidate_max, candidate_spin, state_spin)
        for t in range(300):
            # RL choose action based on observation
            for index in range(len(info_.state_)):
                example.printAst(info_.candidate_[index])
                ast = astEncoder.getAstDict()
                nodes, children = sampling.gen_samples1(ast, embeddings, embed_lookup)
                observation = info_.state_[index]
                fitness = example.get_fitness(info_.candidate_[index])

                action_choosen = RL.getChoosenActions(info_.candidate_[index], observation, action1)

                one_hot_actionchoosen = RL.one_hot_actionchsen(action_choosen)
                one_hot_action1 = RL.one_hot_action1(action1)

                one_hot_act1chsn = np.concatenate(
                    (np.concatenate((one_hot_action1, one_hot_actionchoosen), axis=0), [fitness / 100.00]), axis=0)

                # RL choose action based on observation
                action, action_store = RL.choose_action(action1, nodes, children,one_hot_act1chsn,info_.candidate_[index])

                if action_store < 42:
                    # action1 number
                    action1 = action
                    reward = 0
                    nodes_ = nodes
                    children_ = children
                    one_hot_act1chsn_ = one_hot_act1chsn
                else:
                    action2 = action
                    action_operation = RL.getAction(action1, action2)
                    reward, done, info_ = env.step(action_operation, index)
                    observation_ = info_.state_[index]

                    one_hot_action1_ = RL.one_hot_action1(action1)

                    action_choosen_ = RL.getChoosenActions(info_.candidate_[index], observation_, action1)

                    one_hot_actionchoosen_ = RL.one_hot_actionchsen(action_choosen_)

                    fitness = example.get_fitness(info_.candidate_[index])

                    one_hot_act1chsn_ = np.concatenate(
                        (np.concatenate((one_hot_action1_, one_hot_actionchoosen_), axis=0), [fitness / 100.00]), axis=0)

                    # RL take action and get next observation and reward
                    ast = astEncoder.getAstDict()
                    nodes_, children_ = sampling.gen_samples1(ast, embeddings, embed_lookup)

                    if info_.spin_used == 1:
                        print("i_ep", episode, " step:", t, " fitness", example.get_fitness(info_.candidate_[index]), "time", time.time() - start)
                    if done:
                        print("i_ep", episode, " step:", t, " fitness", example.get_fitness(info_.candidate_[index]), "time",
                              time.time() - start)
                        print("Congradulations!")
                        break


                if reward > 0:
                    step_good += 1
                    RL.store_transition_good(nodes, children, one_hot_act1chsn, action_store, reward, nodes_, children_, one_hot_act1chsn_)

                RL.store_transition(nodes, children, one_hot_act1chsn, action_store, reward, nodes_, children_,
                                    one_hot_act1chsn_)

                reward_cum += reward

                if (step_good > 100) and (step_good % 2 == 0):
                     RL.learn()

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
                      learning_rate=0.001,
                      reward_decay=0.8,
                      e_greedy=0.8,
                      replace_target_iter=10,
                      memory_size=100,
                      # output_graph=True
                      )
    run_maze()
