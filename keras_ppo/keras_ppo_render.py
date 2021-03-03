import gym
import tensorflow as tf
import numpy as np
from keras_ppo import Agent

if __name__ == '__main__':
    #env_name = 'MountainCar-v0'
    env_name = 'CartPole-v1'
    env = gym.make(env_name)
    agent = Agent(actor_file=f'{env_name.lower()}_actor.h5', training=False)
    while True:
        done = False
        state = env.reset()
        while not done:
            env.render()
            agent(state)
            probs = agent.predict(state)
            next_state, reward, done, _ = env.step(np.argmax(probs))
            state = next_state
