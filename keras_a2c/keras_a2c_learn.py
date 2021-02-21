import gym
import random
from keras_a2c import Agent
import numpy as np
import math

if __name__ == '__main__':
    n_step = 16
    env = gym.make('CartPole-v1')
    #entropy_coef = lambda x: math.exp(-0.005*x)
    agent = Agent(env.observation_space.shape, env.action_space.n, alpha=0.01,
                  beta=0.02, entropy_coef=1e-3)
    episode_num = 0
    solved = False
    total_steps = 0
    total_rewards = []
    while not solved:
        state = env.reset()
        done = False
        total_reward = 0
        train_step = 0
        while not done:
            action = agent(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            #if done or train_step % n_step == n_step - 1:
            #    agent.learn(episode_num=episode_num)
            total_reward += reward
            train_step += 1
        agent.learn()
        #if episode_num % 10 == 0:
                #print('save agent')
                #agent.save_model('actor_CartPole-v1.h5', 'critic_CartPole-v1.h5')
        total_steps += train_step
        episode_num += 1
        total_rewards.append(total_reward)
        mean_reward = np.mean(total_rewards[-100:])
        print('Episode:', episode_num, 'Mean Reward:', mean_reward, f'({total_reward})')
        if mean_reward >= 195:
            solved = True
            print(f'Solved after {episode_num} ({total_steps} steps).')
