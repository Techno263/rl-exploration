import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import tensorflow.keras.backend as K
import gym
import numpy as np

from keras_ppo import Agent

tf.keras.backend.set_floatx('float64')

gamma = 0.999

if __name__ == '__main__':
    n_step = 16
    env = gym.make('CartPole-v1')
    agent = Agent(env.observation_space.shape, env.action_space.n, alpha=0.001,
                  beta=0.005, entropy_coef=1e-3, entropy_decay=0.999)
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
            total_reward += reward
            train_step += 1
            #env.render()
        actor_loss, critic_loss = agent.learn()
        #if episode_num % 10 == 0:
            #print('save agent')
            #agent.save_model('actor.h5', 'critic.h5')
        total_steps += train_step
        episode_num += 1
        total_rewards.append(total_reward)
        mean_reward = np.mean(total_rewards[-100:])
        print(f'Episode: {episode_num}, Mean Reward: {mean_reward:0.3f} ({total_reward:3.0f})'
              f', Actor Loss: {actor_loss:0.3f}, Critic Loss: {critic_loss:0.3f}')
        if mean_reward >= 450:
            solved = True
            agent.save_model('cartpole-v1_actor.h5', 'cartpole-v1_critic.h5')
            print(f'Solved after {episode_num} ({total_steps} steps).')
