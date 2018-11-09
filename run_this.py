from maze_env import Maze
from RL_brain import DeepQNetwork
import astEncoder
import example
import StringIO
import time
import os

os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda-9.0/lib64'
os.environ['CUDA_HOME'] = '/usr/local/cuda-9.0'
os.environ["CUDA_VISIBLE_DEVICES"]= '0'

target_reward = 80
def run_maze():
    buf = StringIO.StringIO()
    # observation, info_ = env.reset()
    info_ = env.reset()
    for episode in range(3000):
        step = 0
        # initial observation
        # observation, info_ = env.reset()
        # 100 candidate in actIndex
        actIndex = astEncoder.setAction1s(info_)
        reward_cum = 0
        start = time.time()
        for t in range(10):

            # 100 candidates
            for index in range(100):

                observation = info_.state_[index]

                # RL choose action based on observation
                action, real_action = RL.choose_action(observation, episode, actIndex[index], info_.candidate_[index])

                # RL take action and get next observation and reward
                reward, done, info_ = env.step(real_action, index)
                observation_ = info_.state_[index]
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

                RL.store_transition(observation, action[0], action[1], reward, observation_)
                RL.store_action1(act1_index_store)
                RL.store_candidate(info_.candidate_[index])

                reward_cum += reward

                if (step > 100) and (step % 2 == 0):
                    RL.learn()

                if reward == 100:
                    print("congratulations. Correct program synthesised. ")
                    break

                step += 1

            actIndex = astEncoder.setAction1s(info_)

        end = time.time()
        print("episode: ", episode, "reward_cum: ", reward_cum, "time: ", end - start)
    # end of game
    print('game over')
    # env.destroy()


if __name__ == "__main__":
    # maze game
    env = Maze()
    RL = DeepQNetwork(env.action_space.spaces[0].n,
                      env.action_space.spaces[1].n,
                      env.observation_space.shape[0],
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=10,
                      memory_size=100,
                      # output_graph=True
                      )
    run_maze()
