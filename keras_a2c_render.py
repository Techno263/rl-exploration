import gym
import tensorflow as tf
import numpy as np
from keras_a2c import Agent

actor_model_file = 'CartPole-v1_actor_model.h5'

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    agent = Agent(actor_file='actor_CartPole-v1.h5', training=False)
    while True:
        done = False
        state = env.reset()
        while not done:
            env.render()
            agent(state)
            #probs = agent.predict(state)
            next_state, reward, done, _ = env.step(np.argmax(probs))
            state = next_state
