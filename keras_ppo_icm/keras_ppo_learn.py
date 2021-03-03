import gym
import numpy as np

from keras_ppo_icm import Agent

if __name__ == '__main__':
    #env_name = 'MountainCar-v0'
    env_name = 'CartPole-v1'
    env = gym.make(env_name)
    agent = Agent(
        input_shape=env.observation_space.shape,
        action_num=env.action_space.n,
        alpha=1e-3,
        beta=5e-3,
        gamma=0.99,
        eta=30,
        icm_alpha=1e-3,
        icm_beta=0.2,
        entropy_coef=.1,
        entropy_decay=0.999)
    from tensorflow.keras.utils import plot_model
    plot_model(agent.actor, to_file='actor_model.png', show_shapes=True)
    plot_model(agent.critic, to_file='critic_model.png', show_shapes=True)
    plot_model(agent.icm, to_file='icm_model.png', show_shapes=True)
    episode_num = 0
    solved = False
    total_steps = 0
    total_rewards = []
    while not solved:
        state = env.reset()
        done = False
        running_reward = 0
        train_step = 0
        while not done:
            state, reward, done, _ = agent.step(env, state)
            running_reward += reward
            train_step += 1
            #env.render()
        actor_loss, critic_loss, icm_loss = agent.learn()
        total_rewards.append(running_reward)
        total_rewards = total_rewards[-100:]
        mean_reward = np.mean(total_rewards)
        print(
            f'Episode: {episode_num: 6d} ({train_step: 3d}), '
            f'Mean Reward: {mean_reward: 4.0f} ({running_reward: 4.0f}), '
            f'Actor Loss: {actor_loss:0.5f}, '
            f'Critic Loss: {critic_loss:0.5f}, '
            f'ICM Loss: {icm_loss:0.5f}'
        )
        if episode_num % 100 == 0:
            agent.save_model(
                f'{env_name.lower()}_actor.h5',
                f'{env_name.lower()}_critic.h5',
                f'{env_name.lower()}_icm.h5'
            )
        if mean_reward >= 500:
            solved = True
            agent.save_model(
                f'{env_name.lower()}_actor.h5',
                f'{env_name.lower()}_critic.h5',
                f'{env_name.lower()}_icm.h5'
            )
            print(f'Solved after {episode_num} ({total_steps} steps)')
        total_steps += train_step
        episode_num += 1
