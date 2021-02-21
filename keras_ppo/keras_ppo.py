import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.models import load_model, clone_model
import tensorflow.keras.backend as K
import numpy as np

tf.keras.backend.set_floatx('float64')

def ppo_loss(old_policy):
    pass

class Agent:
    def __init__(self, input_shape=None, action_num=None, alpha=0.0001,
                 beta=0.0005, gamma=0.99, entropy_coef=0.1, entropy_decay=0.99,
                 actor_loss_epsilon=0.2, actor_file=None, critic_file=None,
                 training=True):
        if actor_file == None:
            if input_shape == None or action_num == None:
                raise Exception('input_shape and action_num are required when no actor file is specified.')
            self.actor = self._get_actor(input_shape, action_num, [64])
        else:
            self.actor = load_model(actor_file)
        if training:
            self.states = []
            self.actions = []
            self.rewards = []
            self.next_states = []
            self.dones = []
            self.gamma = gamma
            self.entropy_coef = entropy_coef
            self.entropy_decay = entropy_decay
            self.actor_loss_epsilon = actor_loss_epsilon
            self.actor_optimizer = Adam(learning_rate=alpha)
            self.critic_optimizer = Adam(learning_rate=beta)
            self.critic_loss_func = MeanSquaredError()
            if critic_file == None:
                if input_shape == None:
                    raise Exception('input_shape is required when no critic file is specified.')
                self.critic = self._get_critic(input_shape, [64])
            else:
                self.critic = load_model(critic_file)
            self.prev_actor = clone_model(self.actor)
            self.prev_actor.set_weights(self.actor.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        assert (
            len(self.states)
            == len(self.actions)
            == len(self.rewards)
            == len(self.next_states)
            == len(self.dones)
        )

    def learn(self):
        if not hasattr(self, 'critic'):
            raise Exception('Training is not enabled on this agent')
        states = np.array(self.states)
        next_states = np.array(self.next_states)
        dones = np.expand_dims(self.dones, 1).astype(int)
        rewards = np.expand_dims(self.rewards, 1)
        with tf.GradientTape(persistent=True) as tape:
            states = tf.convert_to_tensor(states)
            next_states = tf.convert_to_tensor(next_states)

            critic_value = self.critic(states)
            critic_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
            critic_loss = self.critic_loss_func(critic_target, critic_value)

            action_probs = self.actor(states)
            indices = K.stack((np.arange(len(action_probs)), self.actions), 1)
            action_prob = K.expand_dims(tf.gather_nd(action_probs, indices), 1)
            prev_action_prob = K.expand_dims(tf.gather_nd(self.prev_actor(states), indices), 1)
            advantage = critic_target - critic_value
            prob_ratio = K.exp(K.log(action_prob) - K.log(prev_action_prob + 1e-10))
            entropy = K.expand_dims(-K.sum(action_probs * K.log(action_probs), 1), 1)
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

        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()
        self.entropy_coef = self.entropy_coef * self.entropy_decay

        return actor_loss.numpy(), critic_loss.numpy()

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

    def save_model(self, actor_filepath, critic_filepath=None):
        self.actor.save(actor_filepath, include_optimizer=False)
        if critic_filepath != None:
            self.critic.save(critic_filepath, include_optimizer=False)

    @staticmethod
    def load_model(actor_file, critic_file=None, alpha=0.0001,
                   beta=0.0005, gamma=0.99, entropy_coef=0.1):
        return Agent(alpha=alpha, beta=beta, gamma=gamma,
                     entropy_coef=entropy_coef, actor_file=actor_file,
                     critic_file=critic_file)
        

    @staticmethod
    def _get_actor(input_shape, action_num, layer_sizes, activation='relu'):
        input_layer = Input(input_shape)
        dense_layer = input_layer
        for size in layer_sizes:
            dense_layer = Dense(size, activation=activation)(dense_layer)
        output_layer = Dense(action_num, activation='softmax')(dense_layer)
        return Model(input_layer, output_layer)

    @staticmethod
    def _get_critic(input_shape, layer_sizes, activation='relu'):
        input_layer = Input(input_shape)
        dense_layer = input_layer
        for size in layer_sizes:
            dense_layer = Dense(size, activation=activation)(dense_layer)
        output_layer = Dense(1, activation='linear')(dense_layer)
        return Model(input_layer, output_layer)
