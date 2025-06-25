import tensorflow as tf
import numpy as np
np.bool = np.bool_
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn import preprocessing
from tqdm import tqdm
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts
from tf_agents.policies.policy_saver import PolicySaver
from tf_agents.agents.ddpg import actor_network
from tf_agents.agents.ddpg import actor_rnn_network
from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.ddpg import ddpg_agent

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.trajectories import policy_step
from tf_agents.utils import common
import logging

import config
from  environments import CardGameEnv
from utils import *


from tf_agents.networks import network
from tf_agents.networks import nest_map


from keras import layers
from keras.regularizers import l2



tf.compat.v1.enable_v2_behavior()
os.makedirs(config.LOGDIR,exist_ok=True)
os.makedirs(config.MODEL_SAVE,exist_ok=True)
logging.basicConfig(filename=os.path.join(config.LOGDIR,'log.log'), 
level=logging.INFO, 
format='%(asctime)s | %(name)s | %(levelname)s | %(message)s')
if __name__=='__main__':
    train_py_env = CardGameEnv()
    eval_py_env = CardGameEnv()
    
    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
    actor_fc_layers = config.actor_fc_layers
    critic_obs_fc_layers = config.critic_obs_fc_layers
    critic_action_fc_layers = config.critic_action_fc_layers
    critic_joint_fc_layers = config.critic_joint_fc_layers
    ou_stddev = config.ou_stddev
    ou_damping = config.ou_damping
    target_update_tau = config.target_update_tau
    target_update_period = config.target_update_period
    dqda_clipping = config.dqda_clipping
    td_errors_loss_fn = config.td_errors_loss_fn
    gamma = config.gamma
    reward_scale_factor = config.reward_scale_factor
    gradient_clipping = config.gradient_clipping

    actor_learning_rate = config.actor_learning_rate
    critic_learning_rate = config.critic_learning_rate
    debug_summaries = config.debug_summaries
    summarize_grads_and_vars = config.summarize_grads_and_vars
    
    global_step = tf.compat.v1.train.get_or_create_global_step()

    class ActorRNNNetwork(network.Network):
        def __init__(self, input_tensor_spec, output_tensor_spec, lstm_size=64, fc_layer_params=(256, 128), l2_reg=1e-4, name='ActorRNNNetwork'):
            super(ActorRNNNetwork, self).__init__(input_tensor_spec=input_tensor_spec, state_spec=(), name=name)
            
            self._output_tensor_spec = output_tensor_spec
            self._lstm_size = lstm_size
            self._l2_reg = l2_reg

            # Define LSTM layer with layer normalization and L2 regularization
            self._lstm_layer = tf.keras.layers.LSTM(
                self._lstm_size, 
                return_sequences=False, 
                kernel_initializer='glorot_uniform',
                kernel_regularizer=l2(self._l2_reg),
                recurrent_regularizer=l2(self._l2_reg),
                bias_regularizer=l2(self._l2_reg)
            )
            self._layer_norm = tf.keras.layers.LayerNormalization()
        
            self._dense_layers = []

            for layer_size in fc_layer_params:
                self._dense_layers.append(tf.keras.layers.Dense(
                    layer_size, 
                    activation=tf.keras.activations.relu,
                    kernel_regularizer=l2(self._l2_reg),
                    bias_regularizer=l2(self._l2_reg)
                ))
                # Add dropout layer for regularization
                self._dense_layers.append(tf.keras.layers.Dropout(0.2))
            
            self._logits_layer = tf.keras.layers.Dense(
                output_tensor_spec.shape.num_elements(), 
                activation=tf.keras.activations.tanh,
                #kernel_regularizer=l2(self._l2_reg),
                #bias_regularizer=l2(self._l2_reg)
            )

        def call(self, observations, step_type=None, network_state=(), training=False):
            # Ensure observations are 3D: (batch_size, timesteps, input_dim)
            # Assuming the observations are initially 2D: (batch_size, input_dim)
            x = tf.expand_dims(observations, axis=1)  # Add time dimension
            
            x = self._lstm_layer(x)
            x = self._layer_norm(x)

            for layer in self._dense_layers:
                x = layer(x, training=training)
            
            logits = self._logits_layer(x, training=training)
            actions = tf.nn.tanh(logits)  # Assuming the action spec is between -1 and 1

            # Scale actions to match the output tensor spec
            actions = common.scale_to_spec(actions, self._output_tensor_spec)
            
            return actions, network_state

    actor_net = ActorRNNNetwork(
        input_tensor_spec=train_env.time_step_spec().observation,
        output_tensor_spec=train_env.action_spec(),
        lstm_size=64,
        fc_layer_params=actor_fc_layers,
    )

    critic_net_input_specs = (train_env.time_step_spec().observation,
                            train_env.action_spec())

    critic_net = critic_network.CriticNetwork(
        critic_net_input_specs,
        observation_fc_layer_params=critic_obs_fc_layers,
        action_fc_layer_params=critic_action_fc_layers,
        joint_fc_layer_params=critic_joint_fc_layers,
    )

    tf_agent = ddpg_agent.DdpgAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        actor_network=actor_net,
        critic_network=critic_net,
        actor_optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=actor_learning_rate),
        critic_optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=critic_learning_rate),
        ou_stddev=ou_stddev,
        ou_damping=ou_damping,
        target_update_tau=target_update_tau,
        target_update_period=target_update_period,
        dqda_clipping=dqda_clipping,
        td_errors_loss_fn=td_errors_loss_fn,
        gamma=gamma,
        reward_scale_factor=reward_scale_factor,
        gradient_clipping=gradient_clipping,
        debug_summaries=debug_summaries,
        summarize_grads_and_vars=summarize_grads_and_vars,
        train_step_counter=global_step)
    tf_agent.initialize()
    
    random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                    train_env.action_spec())

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=tf_agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=config.REPLAY_BUFFER_MAX_LENGTH)

    collect_data(train_env, random_policy, replay_buffer, steps=100)

    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3, 
        sample_batch_size=config.BATCH_SIZE, 
        num_steps=2).prefetch(3)
    
    my_policy = tf_agent.collect_policy
    saver = PolicySaver(my_policy, batch_size=None)

    iterator = iter(dataset)
    tf_agent.train = common.function(tf_agent.train)

    # Reset the train step
    tf_agent.train_step_counter.assign(0)

    # Evaluate the agent's policy once before training.
    
    avg_return = compute_avg_return(eval_env, tf_agent.policy, \
                                    config.NUM_EVAL_EPISODES)
    returns = [avg_return]
    iterations=[0]
    for _ in tqdm(range(config.NUM_ITERATIONS),total=config.NUM_ITERATIONS):
            # Collect a few steps using collect_policy and save to the replay buffer.
            for _ in range(config.COLLECT_STEPS_PER_ITERATION):
                collect_step(train_env, tf_agent.collect_policy, replay_buffer)

            # Sample a batch of data from the buffer and update the agent's network.
            experience, unused_info = next(iterator)
            train_loss = tf_agent.train(experience).loss

            step = tf_agent.train_step_counter.numpy()

            if step % config.LOG_INTERVAL == 0:
                print('step = {0}: loss = {1}'.format(step, train_loss))

            if step % config.EVAL_INTERVAL == 0:
                avg_return = compute_avg_return(eval_env, tf_agent.policy, \
                                                config.NUM_EVAL_EPISODES)
                print('step = {0}: Average Return = {1}'.format(step, avg_return))
                logging.info('step = {0}: Average Return = {1}'.format(step, avg_return))
                returns.append(avg_return)
                iterations.append(step)

                # Save portfolio weights during evaluation
                eval_weights = []
                for _ in range(config.NUM_EVAL_EPISODES):
                    time_step = eval_env.reset()
                    episode_weights = []
                    while not time_step.is_last():
                        action = tf_agent.policy.action(time_step).action.numpy()  # Get portfolio weights
                        episode_weights.append(action)
                        time_step = eval_env.step(action)
                    eval_weights.append(episode_weights)

                # Aggregate weights (e.g., average weights at the last step across all episodes)
                final_weights = np.mean([w[-1] for w in eval_weights], axis=0)  # Average final-step weights
                print(f'step = {step}: Final Portfolio Weights = {final_weights}')
                logging.info(f'step = {step}: Final Portfolio Weights = {final_weights}')
                
                # Save weights for further analysis
                with open(os.path.join(config.MODEL_SAVE, f'weights_step_{step}.csv'), 'w') as f:
                    f.write('Asset1,Asset2,Asset3\n')  # Assuming a 3-stock portfolio
                    f.write(','.join(map(str, final_weights)) + '\n')

                # Save the model periodically
                if step % config.MODEL_SAVE_FREQ == 0:
                    saver.save(os.path.join(config.MODEL_SAVE, f'policy_step_{step}_gamma.mdl'))
                
                # except:
                #     print("error_skipping")

    # iterations = range(0, num_iteratioens + 1, eval_interval)
    plt.plot(iterations, returns)
    plt.ylabel('Average Return')
    plt.xlabel('Iterations')
    plt.ylim(top=50)
    plt.show()
    plt.savefig("output_img_gamma.png")
    pd.DataFrame({"interations":iterations,"Return":returns}).to_csv(os.path.join(config.LOGDIR,"output_ar_gamma.csv"),index=None)