import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.models import load_model, clone_model
import tensorflow.keras.backend as K
import numpy as np
import random

K.set_floatx('float64')

def get_actor(
    input_shape,
    action_num,
    layer_sizes,
    activation='relu'
):
    input_layer = Input(input_shape, name='state_input')
    dense_layer = input_layer
    for index, size in enumerate(layer_sizes):
        layer_name = f'hidden_{index:02d}'
        dense_layer = Dense(size, activation=activation, name=layer_name)(dense_layer)
    output_layer = Dense(action_num, activation='softmax', name='action_prob_output')(dense_layer)
    return Model(input_layer, output_layer, name='actor_model')

def get_critic(
    input_shape,
    layer_sizes,
    activation='relu'
):
    input_layer = Input(input_shape, name='state_input')
    dense_layer = input_layer
    for index, size in enumerate(layer_sizes):
        layer_name = f'hidden_{index:02d}'
        dense_layer = Dense(size, activation=activation, name=layer_name)(dense_layer)
    output_layer = Dense(1, activation='linear', name='value_output')(dense_layer)
    return Model(input_layer, output_layer, name='critic_model')

def get_icm_features_model(
    state_shape,
    latent_shape,
    layer_sizes,
    activation='relu'
):
    input_layer = Input(state_shape, name='state_input')
    dense_layer = input_layer
    for index, size in enumerate(layer_sizes):
        layer_name = f'hidden_{index:02d}'
        dense_layer = Dense(size, activation=activation, name=layer_name)(dense_layer)
    output_layer = Dense(latent_shape, activation='linear', name='state_feature_output')(dense_layer)
    return Model(input_layer, output_layer, name='features_model')

def get_icm_forward_model(
    latent_shape,
    action_num,
    layer_sizes,
    activation='relu'
):
    state_input_layer = Input(latent_shape, name='state_input')
    action_input_layer = Input(action_num, name='action_input')
    dense_layer = Concatenate(name='concat')([state_input_layer, action_input_layer])
    for index, size in enumerate(layer_sizes):
        layer_name = f'hidden_{index:02d}'
        dense_layer = Dense(size, activation=activation, name=layer_name)(dense_layer)
    output_layer = Dense(latent_shape, activation='linear', name='pred_next_state_output')(dense_layer)
    return Model([state_input_layer, action_input_layer], output_layer, name='forward_model')

def get_icm_inverse_model(
    latent_shape,
    action_num,
    layer_sizes,
    activation='relu'
):
    state_input_layer = Input(latent_shape, name='state_input')
    next_state_input_layer = Input(latent_shape, name='next_state_input')
    dense_layer = Concatenate(name='concat')([state_input_layer, next_state_input_layer])
    for index, size in enumerate(layer_sizes):
        layer_name = f'hidden_{index:02d}'
        dense_layer = Dense(size, activation=activation, name=layer_name)(dense_layer)
    output_layer = Dense(action_num, activation='softmax', name='pred_action_output')(dense_layer)
    return Model([state_input_layer, next_state_input_layer], output_layer, name='inverse_model')

def get_intrinsic_curiosity_module(
    state_shape,
    action_num,
    latent_shape
):
    state_input = Input(state_shape, name='state_input')
    action_input = Input(action_num, name='action_input')
    next_state_input = Input(state_shape, name='next_state_input')
    feature_model = get_icm_features_model(
        state_shape,
        latent_shape,
        [256, 256]
    )
    state_feature = feature_model(state_input)
    next_state_feature = feature_model(next_state_input)
    forward_model = get_icm_forward_model(
        latent_shape, action_num, [256, 256]
    )
    pred_next_state = forward_model([state_feature, action_input])
    inverse_model = get_icm_inverse_model(
        latent_shape, action_num, [256, 256]
    )
    pred_action = inverse_model([state_feature, next_state_feature])
    def forward_loss_func(x):
        true_next_state, pred_next_state = x
        return (
            K.constant(0.5)
            * K.mean(
                K.square(pred_next_state - true_next_state),
                axis=1,
                keepdims=True
            )
        )
    forward_loss = Lambda(forward_loss_func, name='forward_loss_output')([next_state_feature, pred_next_state])
    return Model(
        [state_input, action_input, next_state_input],
        [forward_loss, pred_action]
    )

class Experience:
    def __init__(
        self,
        state,
        action,
        reward,
        next_state,
        done,
    ):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done
    
    def __repr__(self):
        return (
            'Experience('
            f'state={repr(self.state)}, '
            f'action={repr(self.action)}, '
            f'reward={repr(self.reward)}, '
            f'next_state={repr(self.next_state)}, '
            f'done={repr(self.done)})'
        )

    def __iter__(self):
        yield self.state
        yield self.action
        yield self.reward
        yield self.next_state
        yield self.done

class Agent:
    def __init__(
        self,
        input_shape=None,
        action_num=None,
        alpha=1e-4,
        beta=5e-4,
        gamma=0.99,
        eta=50,
        icm_alpha=1e-4,
        icm_beta=0.2,
        entropy_coef=0.1,
        entropy_decay=0.99,
        actor_loss_epsilon=0.2,
        actor_file=None,
        critic_file=None,
        icm_file=None,
        training=True
    ):
        if actor_file == None:
            if input_shape == None or action_num == None:
                raise Exception('input_shape and action_num are required when no actor file is specified.')
            self.actor = get_actor(input_shape, action_num, [256, 256])
        else:
            self.actor = load_model(actor_file)
        if training:
            self.input_shape = input_shape
            self.action_num = action_num
            self.experiences = []
            if icm_file == None:
                self.icm = get_intrinsic_curiosity_module(
                    input_shape,
                    action_num,
                    64
                )
            else:
                self.icm = load_model(icm_file)
            self.gamma = gamma
            self.eta = eta
            self.icm_beta = icm_beta
            self.entropy_coef = entropy_coef
            self.entropy_decay = entropy_decay
            self.actor_loss_epsilon = actor_loss_epsilon
            self.actor_optimizer = Adam(learning_rate=alpha)
            self.critic_optimizer = Adam(learning_rate=beta)
            self.icm_optimizer = Adam(learning_rate=icm_alpha)
            self.critic_loss_func = MeanSquaredError()
            self.icm_loss_func = MeanSquaredError()
            if critic_file == None:
                if input_shape == None:
                    raise Exception('input_shape is required when no critic file is specified.')
                self.critic = get_critic(input_shape, [256, 256])
            else:
                self.critic = load_model(critic_file)
            self.prev_actor = clone_model(self.actor)
            self.prev_actor.set_weights(self.actor.get_weights())

    def step(self, env, state):
        action_probs = K.squeeze(self.actor(np.array([state])), 0)
        action = np.random.choice(len(action_probs), p=action_probs)
        next_state, reward, done, env_info = env.step(action)
        experience = Experience(
            state, action, reward, next_state, done
        )
        self.experiences.append(experience)
        return next_state, reward, done, env_info

    def learn(self):
        states, actions, rewards, next_states, done = zip(*self.experiences)
        with tf.GradientTape(persistent=True) as tape:
            states = K.stack(states)
            actions = K.constant(actions, dtype='int32')
            rewards = K.expand_dims(K.constant(rewards, dtype=K.floatx()), 1)
            next_states = K.stack(next_states)
            done = K.expand_dims(K.constant(done, dtype=K.floatx()), 1)

            action_probs = self.actor(states)
            action_log_probs = K.log(action_probs)

            actions_one_hot = K.one_hot(actions, self.action_num)
            forward_losses, pred_actions = self.icm([
                states,
                actions_one_hot,
                next_states
            ])
            forward_loss = K.mean(forward_losses)
            inverse_loss = self.icm_loss_func(actions_one_hot, pred_actions)
            icm_loss = (
                self.icm_beta * forward_loss
                + (1 - self.icm_beta) * inverse_loss
            )

            rewards = rewards + self.eta * forward_losses
            critic_value = self.critic(states)
            critic_target = rewards + self.gamma * self.critic(next_states) * (1 - done)
            critic_loss = self.critic_loss_func(critic_target, critic_value)

            indices = K.stack((np.arange(len(action_probs)), actions), 1)
            action_log_prob = K.expand_dims(
                tf.gather_nd(action_log_probs, indices),
                1
            )
            prev_action_prob = K.expand_dims(
                tf.gather_nd(self.prev_actor(states), indices),
                1
            )
            prob_ratio = K.exp(action_log_prob - K.log(prev_action_prob + 1e-10))
            advantage = critic_target - critic_value
            entropy = -K.sum(action_probs * action_log_probs, 1, keepdims=True)
            actor_loss = -K.mean(
                K.minimum(
                    prob_ratio * advantage,
                    K.clip(
                        prob_ratio,
                        1 - self.actor_loss_epsilon,
                        1 + self.actor_loss_epsilon
                    ) * advantage
                ) + entropy * self.entropy_coef
            )
        self.prev_actor.set_weights(self.actor.get_weights())
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_weights)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_weights))
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_weights)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_weights))
        icm_grads = tape.gradient(icm_loss, self.icm.trainable_weights)
        self.icm_optimizer.apply_gradients(zip(icm_grads, self.icm.trainable_weights))
        self.entropy_coef = self.entropy_coef * self.entropy_decay
        self.experiences.clear()
        return actor_loss.numpy(), critic_loss.numpy(), icm_loss.numpy()

    def __call__(self, state):
        state = np.expand_dims(state, 0)
        probs = np.squeeze(self.actor(state))
        return np.random.choice(len(probs), p=probs)

    def predict(self, state):
        if len(state.shape) == 1:
            squeeze = True
            state = np.expand_dims(state, 0)
        else:
            squeeze = False
        probs = self.actor.predict(state)
        return np.squeeze(probs) if squeeze else probs

    def run_episode(self, env, init_state=None):
        if init_state == None:
            state = env.reset()
        else:
            state = init_state
        done = False
        total_reward = 0
        train_steps = 0
        while not done:
            action = self(state)
            next_state, reward, done, _ = env.step(action)
            self.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            train_steps += 1
        actor_loss, critic_loss = self.learn()
        return actor_loss, critic_loss, total_reward, train_steps

    def save_model(self, actor_filepath, critic_filepath=None, icm_file=None):
        self.actor.save(actor_filepath, include_optimizer=False)
        if critic_filepath != None:
            self.critic.save(critic_filepath, include_optimizer=False)
        if icm_file != None:
            self.icm.save(icm_file, include_optimizer=False)

    @staticmethod
    def load_model(actor_file, critic_file=None, alpha=0.0001,
                   beta=0.0005, gamma=0.99, entropy_coef=0.1):
        return Agent(alpha=alpha, beta=beta, gamma=gamma,
                     entropy_coef=entropy_coef, actor_file=actor_file,
                     critic_file=critic_file)

if __name__ == '__main__':
    import gym
    env = gym.make('CartPole-v1')
    agent = Agent(env.observation_space.shape, env.action_space.n, alpha=0.001,
                  beta=0.005, entropy_coef=1e-3, entropy_decay=0.999)
    state = env.reset()
    done = False
    running_reward = 0
    while not done:
        state, reward, done, _ = agent.step(env, state)
        running_reward += reward
    agent.learn()
    print()
