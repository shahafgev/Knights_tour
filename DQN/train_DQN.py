# Copyright (c) 2021, Paul Almasan [^1]
#
# [^1]: Universitat Politècnica de Catalunya, Computer Architecture
#     department, Barcelona, Spain. Email: felician.paul.almasan@upc.edu

import numpy as np
import gym
import gc
import os
import sys
import gym_environments
import random
import mpnn as gnn
import tensorflow as tf
from collections import deque
import multiprocessing
import time as tt
import glob
from tqdm.auto import tqdm
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import wandb

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


ENV_NAME = 'KnightTour-v1'

SEED = 37
ITERATIONS = 1000 #10000
TRAINING_EPISODES = 20 # 20
EVALUATION_EPISODES = 40
FIRST_WORK_TRAIN_EPISODE = 60 # 60

MULTI_FACTOR_BATCH = 6 # Number of batches used in training
TAU = 0.08 # Only used in soft weights copy

differentiation_str = "sample_DQN_agent"
checkpoint_dir = "./models"+differentiation_str
store_loss = 3 # Store the loss every store_loss batches

os.environ['PYTHONHASHSEED']=str(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Force TensorFlow to use single thread.
# Multiple threads are a potential source of non-reproducible results.
# For further details, see: https://stackoverflow.com/questions/42022950/
# tf.config.threading.set_inter_op_parallelism_threads(1)
# tf.config.threading.set_intra_op_parallelism_threads(1)

tf.random.set_seed(1)

train_dir = "./TensorBoard/"+differentiation_str
# summary_writer = tf.summary.create_file_writer(train_dir)

copy_weights_interval = 50
evaluation_interval = 20
epsilon_start_decay = 70


hparams = {
    'l2': 0.1,
    'dropout_rate': 0.01,
    'input_shape' : 4,
    'nodes_state_dim': 20, # Size of the hidden states feature vector
    'readout_units': 35, # Number of neurons in the hidden layers of the Readout MLP
    'learning_rate': 0.0001,
    'batch_size': 32,
    'T': 4, #  Amount of message passing
}

MAX_QUEUE_SIZE = 4000


class DQNAgent:
    def __init__(self, batch_size):
        self.memory = deque(maxlen=MAX_QUEUE_SIZE) # Replay memory
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0 # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.writer = None

        self.listQValues = None
        self.numbersamples = batch_size
        self.action = None

        self.global_step = 0
        self.primary_network = gnn.myModel(hparams)
        self.primary_network.build()
        self.target_network = gnn.myModel(hparams)
        self.target_network.build()
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=hparams['learning_rate'],momentum=0.9,nesterov=True)

    def act(self, env, state, flagEvaluation=False):
        """
        Selects an action using epsilon-greedy policy for the Knight's Tour environment.
        - env: KnightTourEnv instance
        - state: np.ndarray of shape [num_squares, input_dim]
        - flagEvaluation: if True, always pick best action (no exploration)
        """
        valid_actions = env.get_valid_moves()  # List of valid action indices
        state_tensor = tf.convert_to_tensor(state, dtype=tf.float32) # [num_squares, input_dim]
        q_values = self.primary_network(state_tensor, env.adjacency_list, training=False).numpy()  # [8]

        # Mask out invalid actions
        masked_q_values = np.full_like(q_values, -np.inf)
        masked_q_values[valid_actions] = q_values[valid_actions]

        # Exploitation
        if flagEvaluation or np.random.rand() > self.epsilon:
            action = np.argmax(masked_q_values)
        # Exploration
        else:
            action = np.random.choice(valid_actions)

        return action
    
    @tf.function
    def _forward_pass(self, node_features, adjacency_list, training=True):
        """
        Performs a forward pass through the primary and target networks.
        """
        prediction_state = self.primary_network(node_features, adjacency_list, training=training)
        preds_next_target = tf.stop_gradient(self.target_network(node_features, adjacency_list, training=training))
        return prediction_state, preds_next_target

    def _train_step(self, batch):
        """
        Performs a single training step (mini-batch update) for the DQN agent.

        Args:
            batch: List of tuples, each containing
                (state, adjacency_list, action, reward, done, next_state, next_adjacency_list)
                - state: np.ndarray, current board state [num_nodes, input_dim]
                - adjacency_list: list of lists, graph structure for the state
                - action: int, action taken
                - reward: float, reward received
                - done: bool, whether the episode ended
                - next_state: np.ndarray, next board state
                - next_adjacency_list: list of lists, graph structure for the next state

        Steps:
            - For each sample in the batch:
                - Compute Q-values for the current state and next state using the primary and target networks.
                - Select the Q-value for the action taken.
                - Compute the target Q-value as reward + gamma * max_next_q * (1 - done).
            - Compute the mean squared error loss between predicted and target Q-values.
            - Add L2 regularization loss.
            - Compute, clip, and apply gradients to update the primary network.
        Returns:
            grad: List of gradients for logging/debugging.
            loss: Scalar loss value for this training step.
        """
        # Record operations for automatic differentiation
        with tf.GradientTape() as tape:
            preds_state = []  # List to store predicted Q-values for each sample in the batch [batch_size]
            target = []       # List to store target Q-values for each sample in the batch [batch_size]
            for state, adjacency_list, action, reward, done, next_state, next_adjacency_list in batch:
                # Convert states to tensors
                state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)  # [num_nodes, input_dim]
                next_state_tensor = tf.convert_to_tensor(next_state, dtype=tf.float32)  # [num_nodes, input_dim]

                # Q-values for current state and next state
                q_values = self.primary_network(state_tensor, adjacency_list, training=True)  # [num_actions=8]
                next_q_values = self.target_network(next_state_tensor, next_adjacency_list, training=True)  # [num_actions=8]

                # Q-value for the action taken (scalar)
                q_value = q_values[action]  # []
                # Max Q-value for next state (scalar)
                max_next_q = tf.reduce_max(next_q_values)  # []
                # Target Q-value: reward + gamma * max_next_q (if not done)
                target_q = reward + self.gamma * max_next_q * (1.0 - float(done))  # []

                preds_state.append(q_value)  # Accumulate predicted Q-value
                target.append(target_q)      # Accumulate target Q-value

            preds_state = tf.stack(preds_state)  # [batch_size], predicted Q-values for actions taken
            target = tf.stack(target)            # [batch_size], target Q-values
            # Compute mean squared error loss between predicted and target Q-values
            loss = tf.keras.losses.MSE(target, preds_state)
            # Add L2 regularization loss from the model
            regularization_loss = sum(self.primary_network.losses)
            loss = loss + regularization_loss

        # Computes the gradient of the loss with respect to the model parameters
        grad = tape.gradient(loss, self.primary_network.variables)
        # Clip gradients to the range [-1, 1] to prevent exploding gradients
        gradients = [tf.clip_by_value(gradient, -1., 1.) for gradient in grad]
        # Apply gradients to update the model parameters
        self.optimizer.apply_gradients(zip(gradients, self.primary_network.variables))
        del tape
        return grad, loss

    def replay(self, episode):
        """
        Performs experience replay for the DQN agent.
        - Samples random mini-batches from the replay buffer and trains the agent.
        - Logs loss values and periodically updates the target network.

        Args:
            episode: int, current training episode (used for target network update frequency)

        Steps:
            - For MULTI_FACTOR_BATCH times:
                - Sample a random batch from the replay buffer.
                - Perform a training step using _train_step.
                - Log the loss every store_loss batches.
            - Every copy_weights_interval episodes, update the target network weights.
            - Run garbage collection to free memory.
        """
        for i in range(MULTI_FACTOR_BATCH):
            batch = random.sample(agent.memory, agent.numbersamples)
            grad, loss = agent._train_step(batch)
            wandb.log({"train/loss": loss.numpy(), "train/episode": ep_it})
            #print(f"[Train] Iter {ep_it} | Batch {i} | Loss: {loss.numpy():.6f}")
        # Hard weights update
        if ep_it % copy_weights_interval == 0:
            agent.target_network.set_weights(agent.primary_network.get_weights())
        gc.collect()
    
    def add_sample(self, state, adjacency_list, action, reward, done, next_state, next_adjacency_list):
        """
        Store a transition in the replay buffer for Knight's Tour:
        (state, adjacency_list, action, reward, done, next_state, next_adjacency_list)
        """
        self.memory.append((state, adjacency_list, action, reward, done, next_state, next_adjacency_list))

if __name__ == "__main__":
    # --- Weights & Biases (wandb) Setup ---
    wandb.init(project="knight-tour-dqn-gnn", config={
        "env": ENV_NAME,
        "seed": SEED,
        "iterations": ITERATIONS,
        "training_episodes": TRAINING_EPISODES,
        "evaluation_episodes": EVALUATION_EPISODES,
        "batch_size": hparams['batch_size'],
        "gamma": 0.95,
        "epsilon_start": 1.0,
        "epsilon_min": 0.01,
        "epsilon_decay": 0.995,
        "learning_rate": hparams['learning_rate'],
        "node_state_dim": hparams['nodes_state_dim'],
        "readout_units": hparams['readout_units'],
        "T": hparams['T'],
    })

    # --- Environment and Agent Setup ---
    env_training = gym.make(ENV_NAME)  # Training environment
    np.random.seed(SEED)
    env_training.seed(SEED)
    env_eval = gym.make(ENV_NAME)      # Evaluation environment
    np.random.seed(SEED)
    env_eval.seed(SEED)

    batch_size = hparams['batch_size']
    agent = DQNAgent(batch_size)       # Initialize DQN agent

    eval_ep = 0
    train_ep = 0
    max_reward = 0
    reward_id = 0

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(model=agent.primary_network, optimizer=agent.optimizer)

    rewards_test = np.zeros(EVALUATION_EPISODES)

    # --- Initial Evaluation Before Training ---
    for eps in tqdm(range(EVALUATION_EPISODES)):
        state = env_eval.reset()
        rewardAddTest = 0
        eval_steps = 0
        while True:
            action = agent.act(env_eval, state, True)  # Always exploit during evaluation
            next_state, reward, done, _ = env_eval.step(action)
            rewardAddTest += reward
            state = next_state
            eval_steps += 1
            if done:
                break
        rewards_test[eps] = rewardAddTest
        wandb.log({"initial_evaluation/episode_reward": rewardAddTest})
        wandb.log({"initial_evaluation/epsilon": agent.epsilon})
    #print(f"[Init Eval] Mean Reward: {evalMeanReward:.2f} | Epsilon: {agent.epsilon:.4f}")

    counter_store_model = 1

    # --- Main Training Loop ---
    for ep_it in tqdm(range(ITERATIONS)): # 100 times

        # Use more training episodes at the start to fill replay buffer
        if ep_it == 0:
            train_episodes = FIRST_WORK_TRAIN_EPISODE
        else:
            train_episodes = TRAINING_EPISODES
        for _ in range(train_episodes): # 60 at first, than 20
            tf.random.set_seed(1)  # For reproducibility
            state = env_training.reset()
            episode_reward = 0
            step_count = 0
            while True:
                action = agent.act(env_training, state, False)  # Epsilon-greedy during training
                next_state, reward, done, _ = env_training.step(action)
                # Store transition in replay buffer
                agent.add_sample(state, env_training.adjacency_list, action, reward, done, next_state, env_training.adjacency_list)
                state = next_state
                episode_reward += reward
                step_count += 1
                if done:
                    break
            wandb.log({"train/episode_reward": episode_reward, "train/iterations": ep_it, "train/epsilon": agent.epsilon})
            #print(f"[Train] Episode {ep_it * train_episodes + train_ep} | Reward: {episode_reward:.2f} | Steps: {step_count} | Epsilon: {agent.epsilon:.4f}")

        # Train the agent using experience replay
        agent.replay(ep_it)

        # --- Epsilon Decay (Exploration Rate) ---
        if ep_it > epsilon_start_decay and agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

        # --- Periodic Evaluation and Model Saving ---
        if ep_it % evaluation_interval == 0:
            for eps in range(EVALUATION_EPISODES):
                state = env_eval.reset()
                rewardAddTest = 0
                eval_steps = 0
                while True:
                    action = agent.act(env_eval, state, True)  # Always exploit during evaluation
                    next_state, reward, done, _ = env_eval.step(action)
                    rewardAddTest += reward
                    state = next_state
                    eval_steps += 1
                    if done:
                        break
                rewards_test[eps] = rewardAddTest
                wandb.log({"eval/episode_reward": rewardAddTest})
                wandb.log({"eval/epsilon": agent.epsilon})
            evalMeanReward = np.mean(rewards_test)

            #print(f"[Eval] Iter {ep_it} | Mean Reward: {evalMeanReward:.2f} | Epsilon: {agent.epsilon:.4f}")

            # Track and save the best model
            if evalMeanReward > max_reward:
                max_reward = evalMeanReward
                reward_id = ep_it
            checkpoint.save(checkpoint_prefix)
            wandb.log({"model/max_reward": max_reward})
            #print(f"[Model] Saved checkpoint at iter {ep_it} | Max Reward: {max_reward:.2f} | Model ID: {reward_id}")

