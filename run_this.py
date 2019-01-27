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
import operator

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
                candidate = example.copyProgram(info_.candidate_[index])
                #example.printAst(candidate)
                #ast = astEncoder.getAstDict()
                #nodes, children nodes_temp111, children_tempchildren111= sampling.gen_samples1(ast, embeddings, embed_lookup)


                example.printAstint(candidate)
                nodes, children = sampling.gen_samplesint(embeddings)


                '''
                # nodes, children = sampling._pad_batch_(nodes, children)

                for node_i in nodes_temp:
                    inNode = 0
                    for nods_ in nodes:
                        if (nods_ == node_i).all() == True:
                            inNode = 1
                    if inNode == 0:
                        print("???")


                for children_i in children_tempchildren:
                    if children_i not in children:
                        print("???")

                for node_i in nodes:
                    inNode = 0
                    for nods_ in nodes_temp:
                        if (nods_ == node_i).all() == True:
                            inNode = 1
                    if inNode == 0:
                        print("???")

                for children_i in children:
                    if children_i not in children_tempchildren:
                        print("???")
                '''
                '''
                if np.array(nodes).shape != np.array(nodes_temp).shape:
                    print("??")

                if np.array(children_temp).shape != np.array(children).shape:
                    print("????")
                '''
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
                    # nodes_temp0, children_temp0 = nodes_temp, children_tempchildren
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
                    #example.printAst(info_.candidate_[index])
                    #ast = astEncoder.getAstDict()
                    # nodes_, children_ , nodes_temp0, children_temp0 = sampling.gen_samples1(ast, embeddings, embed_lookup)

                    candidate_2 = example.copyProgram(info_.candidate_[index])

                    example.printAstint(info_.candidate_[index])
                    nodes_, children_ = sampling.gen_samplesint(embeddings)
                    '''
                    for node_i_ in nodes_temp0:
                        inNode = 0
                        for node_i_0 in nodes_:
                            if (node_i_0 == node_i_).all() == True:
                                inNode = 1
                        if inNode == 0:
                            print("???")

                    for children_i_ in children_temp0:
                        if children_i_ not in children_:
                            print("???")

                    for node_i_ in nodes_:
                        inNode = 0
                        for node_i_0 in nodes_temp0:
                            if (node_i_0 == node_i_).all() == True:
                                inNode = 1
                        if inNode == 0:
                            print("???")

                    for children_i_ in children_:
                        if children_i_ not in children_temp0:
                            print("???")

                    '''
                    '''
                    if np.array(nodes_temp0).shape != np.array(nodes_).shape:
                        print("qiqiiq")

                    if np.array(children_temp0).shape != np.array(children_).shape:
                        print("qiqiiq2")
                    '''

                    if info_.spin_used == 1:
                        print("i_ep", episode, " step:", t, " fitness", example.get_fitness(info_.candidate_[index]), "time", time.time() - start)
                    if done:
                        print("i_ep", episode, " step:", t, " fitness", example.get_fitness(info_.candidate_[index]), "time",
                              time.time() - start)
                        print("Congradulations!")
                        break

                '''
                if reward > 0:
                    step_good += 1
                    RL.store_transition_good(nodes, children, one_hot_act1chsn, action_store, reward, nodes_, children_, one_hot_act1chsn_)
                '''
                #RL.store_transition_good(nodes_temp, children_tempchildren, one_hot_act1chsn, action_store, reward, nodes_temp0, children_temp0,
                #                         one_hot_act1chsn_)
                RL.store_transition(nodes, children, one_hot_act1chsn, action_store, reward, nodes_, children_,
                                    one_hot_act1chsn_)

                reward_cum += reward

                if (step > 100):
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
