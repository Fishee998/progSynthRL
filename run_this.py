from maze_env import Maze
from RL_brain import DeepQNetwork
import astEncoder
import example
import StringIO
import pickle
import classifier.tbcnn.sampling as sampling

target_reward = 80
def run_maze():
    step = 0
    buf = StringIO.StringIO()

    tfrecord_wrt = RL.make_tfrecord()
    for episode in range(300):
        # initial observation
        observation, info_ = env.reset()
        actIndex = astEncoder.setAction1s(info_)
        reward_cum = 0
        ast = astEncoder.getAstDict()
        # state, astActNodes = astEncoder.astEncoder(ast)
        nodes, children = sampling.gen_samples1(ast, embeddings, embed_lookup)
        nodes, children = RL.reshapeChildNodes(nodes, children)
        # observation = RL.get_observation(nodes, children)
        for t in range(300):
            # fresh env
            # env.render()

            # RL choose action based on observation
            action = RL.choose_action(nodes, children, actIndex, info_.candidate)

            # RL take action and get next observation and reward
            observation_, reward, done, info_ = env.step(action)
            nodes_, children_ = sampling.gen_samples1(info_.ast, embeddings, embed_lookup)
            nodes_, children_ = RL.reshapeChildNodes(nodes_, children_)
            # observation_ = RL.get_observation(nodes, children)
            #if len(observation_) < 1680:
            #    print("error")
            # print((observation - observation_).any())

            if done and example.get_fitness(info_.candidate) > 78.4:
                reward = target_reward
                print(
                "i_ep" + str(episode) + " step:" + str(t) + " fitness" + str(example.get_fitness(info_.candidate)))
                spin_reward = example.spin_(info_.candidate)
                # buf2.write("program at i_ep %d: step:%d  fitnessValue:%d\n" % (i_episode, t, example.get_fitness(info_.candidate)))
                if spin_reward == 20:
                    buf.write("correct program at i_ep %d: step:%d \n" % (episode, t))
                    fo = open("./correctProg.txt", "a+")
                    fo.write(buf.getvalue())
                    fo.close()
                if spin_reward == 5:
                    print("liveness")
                if spin_reward == 10:
                    print("safety")
                reward = reward + spin_reward

            actIndex = astEncoder.setAction1s(info_)

            RL.store_transition(nodes, children, action[0], action[1], reward, nodes_, children_)

            reward_cum += reward


            if (step > 200) and (step % 100 == 0):
                RL.learn()

            # swap observation
            observation = observation_
            nodes = nodes_
            children = children_
            # print(observation)

            # break while loop when end of this episode
            if done:
                break
            step += 1
        print("episode", episode, "reward_cum", reward_cum)
    # end of game
    print('game over')
    env.destroy()


if __name__ == "__main__":
    # maze game
    env = Maze()
    with open('/Users/zhuang/workspace/ast-node-encoding/data/vectors.pkl', 'rb') as fh:
        embeddings, embed_lookup = pickle.load(fh)
        num_feats = len(embeddings[0])
    RL = DeepQNetwork(
                      env.action_space.spaces[0].n,
                      env.action_space.spaces[1].n,
                      num_feats,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      # output_graph=True
                      )
    env.after(100, run_maze)
    env.mainloop()
    RL.plot_cost()