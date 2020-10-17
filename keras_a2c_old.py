import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import numpy as np
import gym
import os.path as path
import gc

tf.keras.backend.set_floatx('float64')

gamma = 0.99
max_steps = 10000
env_name = 'CartPole-v0'
actor_model_file = 'MountainCar-v0_actor_model.h5'
critic_model_file = 'MountainCar-v0_critic_model.h5'
force_rebuild_actor = False
force_rebuild_critic = False

def get_actor(input_shape, num_actions, dense_layers, activation='relu'):
    input_layer = Input(input_shape, name='input')
    dense_layer = input_layer
    for i, units in enumerate(dense_layers):
        dense_layer = Dense(units, activation=activation, name=f'dense_{i}')(dense_layer)
    output_layer = Dense(num_actions, activation='softmax', name='output')(dense_layer)
    return Model(input_layer, output_layer)

def get_critic(input_shape, dense_layers, activation='relu'):
    input_layer = Input(input_shape, name='input')
    dense_layer = input_layer
    for i, units in enumerate(dense_layers):
        dense_layer = Dense(units, activation=activation, name=f'dense_{i}')(dense_layer)
    output_layer = Dense(1, activation='linear', name='output')(dense_layer)
    return Model(input_layer, output_layer)

if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    env.seed(26)
    if not path.exists(actor_model_file) or force_rebuild_actor:
        print('created actor')
        actor = get_actor(env.observation_space.shape, env.action_space.n, [128, 128])
        actor.save(actor_model_file)
        K.clear_session()
        gc.collect()
        del actor
    if not path.exists(critic_model_file) or force_rebuild_critic:
        print('created critic')
        critic = get_critic(env.observation_space.shape, [128, 128])
        critic.save(critic_model_file)
        K.clear_session()
        gc.collect()
        del critic
    actor = load_model(actor_model_file)
    critic = load_model(critic_model_file)
    actor_optimizer = Adam(learning_rate=0.001)
    critic_optimizer = Adam(learning_rate=0.005)
    critic_loss_func = MeanSquaredError()
    episode_num = 1
    solved = False
    while not solved:
        if episode_num % 100 == 0:
            actor.save(actor_model_file)
            critic.save(critic_model_file)
            print('saved model')
        state = env.reset()
        state = tf.convert_to_tensor(state)
        state = tf.expand_dims(state, 0)
        total_reward = 0
        for timestep in range(1, max_steps+1):
            env.render()
            with tf.GradientTape(persistent=True) as tape:
                action_probs = K.flatten(actor(state))
                action = np.random.choice(env.action_space.n, p=action_probs)
                next_state, reward, done, _ = env.step(action)
                next_state = tf.convert_to_tensor(next_state)
                next_state = tf.expand_dims(next_state, 0)
                total_reward += reward
                critic_value = critic(state)
                critic_target = reward + gamma * critic(next_state) * (1-int(done))
                advantage = critic_target - critic_value
                action_prob = action_probs[action]
                a = K.stack((action_probs, K.log(action_probs)), 0)
                entropy = -K.sum(K.prod(a, 0))
                actor_loss = -K.log(action_prob) * advantage - entropy
                critic_loss = critic_loss_func(critic_target, critic_value)
            actor_grads = tape.gradient(actor_loss, actor.trainable_weights)
            actor_optimizer.apply_gradients(zip(actor_grads, actor.trainable_weights))
            critic_grads = tape.gradient(critic_loss, critic.trainable_weights)
            critic_optimizer.apply_gradients(zip(critic_grads, critic.trainable_weights))
            del tape
            state = next_state
            if done:
                print('episode:', episode_num, 'total reward:', total_reward, timestep)
                episode_num += 1
                break
