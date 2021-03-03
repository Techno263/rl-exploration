from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import tensorflow.keras.backend as K
import numpy as np

def get_features_model(
    state_shape,
    latent_shape,
    layer_sizes,
    activation='relu'
):
    input_layer = Input(state_shape)
    dense_layer = input_layer
    for size in layer_sizes:
        dense_layer = Dense(size, activation=activation)(dense_layer)
    output_layer = Dense(latent_shape, activation='linear')(dense_layer)
    return Model(input_layer, output_layer)

def get_forward_model(
    latent_shape,
    action_num,
    layer_sizes,
    activation='relu'
):
    state_input_layer = Input(latent_shape)
    action_input_layer = Input(action_num)
    dense_layer = Concatenate()([state_input_layer, action_input_layer])
    for size in layer_sizes:
        dense_layer = Dense(size, activation=activation)(dense_layer)
    output_layer = Dense(latent_shape, activation='linear')(dense_layer)
    return Model([state_input_layer, action_input_layer], output_layer)

def get_inverse_model(
    latent_shape,
    action_num,
    layer_sizes,
    activation='relu'
):
    state_input_layer = Input(latent_shape)
    next_state_input_layer = Input(latent_shape)
    dense_layer = Concatenate()([state_input_layer, next_state_input_layer])
    for size in layer_sizes:
        dense_layer = Dense(size, activation=activation)(dense_layer)
    output_layer = Dense(action_num, activation='softmax')(dense_layer)
    return Model([state_input_layer, next_state_input_layer], output_layer)

def get_intrinsic_curiosity_module(
    state_shape,
    action_num,
    latent_shape
):
    state_input = Input(state_shape)
    action_input = Input(action_num)
    next_state_input = Input(state_shape)
    feature_model = get_features_model(
        state_shape,
        latent_shape,
        [32, 64]
    )
    state_feature = feature_model(state_input)
    next_state_feature = feature_model(next_state_input)
    forward_model = get_forward_model(
        latent_shape, action_num, [64, 32]
    )
    pred_next_state = forward_model([state_feature, action_input])
    inverse_model = get_inverse_model(
        latent_shape, action_num, [64, 32]
    )
    pred_action = inverse_model([state_feature, next_state_feature])
    forward_loss = (
        K.constant(0.5)
        * K.mean(
            K.square(pred_next_state - next_state_feature),
            axis=1,
            keepdims=True
        )
    )
    return Model(
        [state_input, action_input, next_state_input],
        [forward_loss, pred_action]
    )

class IntrinsicCuriosityModule:
    def __init__(self, state_shape, action_num, latent_shape, alpha=1e-4, beta=0.2):
        self.icm = get_intrinsic_curiosity_module(
            state_shape,
            action_num,
            latent_shape
        )
        self.beta = beta
        self.optimizer = Adam(learning_rate=alpha)

    def learn(self, states, actions, next_states):
        with tf.GradientTape() as tape:
            forward_losses, pred_actions = self(states, actions, next_states)
            forward_loss = K.mean(forward_losses)
            inv_loss = MeanSquaredError(actions, pred_actions)
            loss = (
                self.beta * forward_loss
                + (1 - self.beta) * inv_loss
            )
        grads = tape.gradient(loss, self.icm.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.icm.trainable_weights))
        self.states.clear()
        self.actions.clear()
        self.next_states.clear()
        return loss

    def __call__(self, state, action, next_state):
        breakpoint()
        # Expected shapes
        # state: [None, state_shape]
        # action: [None, action]
        # next_state: [None, state_shape]
        forward_loss, _ = self.icm(state, action, next_state)
        return forward_loss

    def save_module(state_features_filepath, forward_model_filepath, inverse_model_filepath):
        pass

    @staticmethod
    def load_module(state_features_filepath, forward_model_filepath, inverse_model_filepath):
        pass
        
if __name__ == '__main__':
    from tensorflow.keras.utils import plot_model
    icm = get_intrinsic_curiosity_module(4, 2, 64)
    print(icm.summary())
    print('input shape', icm.input_shape)
    print('output shape', icm.output_shape)
