import gym
import tensorflow as tf
from keras_a2c.keras_a2c import Agent as A2CAgent
from keras_ppo.keras_ppo import Agent as PPOAgent
import tensorflow.keras.backend as K
import numpy as np
import gc
from tqdm import tqdm

#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.25
#K.tensorflow_backend.set_session(tf.Session(config=config))

def solve_env(env, agent, prefix='', episode_timeout=2000):
    solved = False
    episode_num = 0
    total_steps = 0
    total_rewards = []
    prog_bar = tqdm(total=episode_timeout, ncols=140)
    while not solved:
        actor_loss, critic_loss, total_reward, train_steps = agent.run_episode(env)
        episode_num += 1
        total_steps += train_steps
        total_rewards.append(total_reward)
        mean_reward = np.mean(total_rewards[-100:])
        #print(f'Episode: {episode_num:4}, Mean Reward: {mean_reward:3.3f} ({total_reward:3.0f})')
        msg = f'Mean Reward: {mean_reward: 6.3f} ({total_reward: 3.0f})'
        if prefix:
            msg += prefix + ' ' + msg
        
        prog_bar.set_description(msg)
        prog_bar.update(1)
        if mean_reward >= 195:
            solved = True
        if episode_num > episode_timeout:
            break
    prog_bar.close()
    return episode_num, total_steps, total_rewards, solved

def analyze_solves(episode_metrics, title):
    episodes, steps, rewards, solved = zip(*episode_metrics)
    episodes = np.array(episodes)
    steps = np.array(steps)
    solved = np.array(solved).astype(int)

    total_episodes = np.sum(episodes)
    episode_mean = np.sum(episodes) / len(episodes)
    episode_variance = np.sum(((episodes - episode_mean) ** 2) / len(episodes))
    episode_stddev = np.sqrt(episode_variance)

    total_steps = np.sum(steps)
    steps_mean = np.sum(steps) / len(steps)
    steps_variance = np.sum(((steps - steps_mean) ** 2) / len(steps))
    steps_stddev = np.sqrt(steps_variance)
    '''
    total_rewards = np.sum(rewards)
    rewards_mean = np.sum(rewards) / len(rewards)
    rewards_variance = np.sum(((rewards - rewards_mean) ** 2) / len(rewards))
    rewards_stddev = np.sqrt(rewards_variance)
    '''
    solved_count = np.sum(solved)

    print('========================================')
    print(title)
    print(f'''
Solved: {solved_count}/{len(solved)}
Episode: Total: {total_episodes:01.0f}, Mean: {episode_mean:01.2f}, Stddev: {episode_stddev:01.2f}
Steps: Total: {total_steps:01.0f}, Mean: {steps_mean:01.2f}, Stddev {steps_stddev:01.2f}  
    ''')
    print('========================================')

def get_ppo_metrics(alpha, beta, gamma, entropy_coef, entropy_decay):
    env = gym.make('CartPole-v1')
    ppo_agent = PPOAgent(
        env.observation_space.shape, env.action_space.n, alpha=alpha,
        beta=beta, gamma=gamma, entropy_coef=entropy_coef,
        entropy_decay=entropy_decay
    )
    ret_vals = solve_env(env, ppo_agent)
    K.clear_session()
    gc.collect()
    return ret_vals

def get_a2c_metrics(alpha, beta, gamma, entropy_coef, entropy_decay):
    env = gym.make('CartPole-v1')
    a2c_agent = A2CAgent(
        env.observation_space.shape, env.action_space.n, alpha=alpha,
        beta=beta, gamma=gamma, entropy_coef=entropy_coef,
        entropy_decay=entropy_decay
    )
    ret_vals = solve_env(env, a2c_agent)
    K.clear_session()
    gc.collect()
    return ret_vals

def compare_models(alpha, beta, gamma, entropy_coef, entropy_decay):
    env = gym.make('CartPole-v1')
    ppo_agent = PPOAgent(
        env.observation_space.shape, env.action_space.n, alpha=alpha,
        beta=beta, gamma=gamma, entropy_coef=entropy_coef,
        entropy_decay=entropy_decay
    )
    a2c_agent = A2CAgent(
        env.observation_space.shape, env.action_space.n, alpha=alpha,
        beta=beta, gamma=gamma, entropy_coef=entropy_coef,
        entropy_decay=entropy_decay
    )
    a2c_agent.actor.set_weights(ppo_agent.actor.get_weights())
    a2c_agent.critic.set_weights(ppo_agent.critic.get_weights())
    ppo_metrics = solve_env(env, ppo_agent, 'PPO')
    a2c_metrics = solve_env(env, a2c_agent, 'A2C')
    #ppo_metrics = (*ppo_metrics, 'ppo')
    #a2c_metrics = (*a2c_metrics, 'a2c')
    K.clear_session()
    gc.collect()
    return ppo_metrics, a2c_metrics
    

if __name__ == '__main__':
    gpus = tf.config.list_physical_devices('GPU')
    assert len(gpus) == 1
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8)]
    )

    timeout = 2000
    n_solves = 2

    alpha = 0.01
    beta = 0.05
    gamma = 0.99
    entropy_coef = 1e-3
    entropy_decay=0.995

    metrics = [
        compare_models(alpha, beta, gamma, entropy_coef, entropy_decay)
        for _ in range(10)
    ]
    ppo_metrics, a2c_metrics = map(list, zip(*metrics))
    #ppo_metrics, a2c_metrics = compare_models(alpha, beta, gamma, entropy_coef, entropy_decay)

    '''
    ppo_metrics = [
        get_ppo_metrics(
            alpha, beta, gamma, entropy_coef, entropy_decay
        )
        for _ in range(n_solves)
    ]
    K.clear_session()
    gc.collect()
    a2c_metrics = [
        get_a2c_metrics(
            alpha, beta, gamma, entropy_coef, entropy_decay
        )
        for _ in range(n_solves)
    ]
    K.clear_session()
    gc.collect()
    '''
    analyze_solves(ppo_metrics, '--- Proximal Policy Optimization ---')
    analyze_solves(a2c_metrics, '--- Advantage Actor Critic ---')
