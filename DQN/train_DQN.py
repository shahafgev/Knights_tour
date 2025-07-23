import numpy as np
import gym
import gc
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Only show errors, not warnings

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
import wandb


ENV_NAME = 'KnightTour-v1'
SEED = 37
ITERATIONS = 100
TRAINING_EPISODES = 10 # 20 
EVALUATION_EPISODES = 20 # 40
FIRST_WORK_TRAIN_EPISODE = 30 # 60
COPY_WEIGHTS_INTERVAL = 10 # 20
EVALUATION_INTERVAL = 10 # 20
EPSILON_START_DECAY = 30 # 70
MAX_QUEUE_SIZE = 4000
MULTI_FACTOR_BATCH = 6 # Number of batches used in training

hparams = {
    'l2': 1e-4,                # Lower L2 regularization to prevent overfitting but not too high
    'dropout_rate': 0.1,       # Slightly higher dropout for regularization
    'input_shape': 6,          # 5 features per node (as per your new state)
    'nodes_state_dim': 32,     # Larger hidden state for richer node representation
    'readout_units': 64,       # More units in the readout MLP for better capacity
    'learning_rate': 0.0005,   # Slightly higher learning rate for faster convergence
    'batch_size': 64,          # Larger batch size for more stable updates
    'T': 4,                    # Message passing steps (4 is a good start)
}

gamma = 0.99
epsilon_start = 1.0
epsilon_min = 0.3
epsilon_decay = 0.995


differentiation_str = "sample_DQN_agent"
checkpoint_dir = "./models" + differentiation_str
store_loss = 3 # Store the loss every store_loss batches

os.environ['PYTHONHASHSEED']=str(SEED)
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(1)


class DQNAgent:
    def __init__(self, batch_size):
        self.memory = deque(maxlen=MAX_QUEUE_SIZE) # Replay memory
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0 # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        self.numbersamples = batch_size
        self.action = None
        self.knight_moves = [
            (-2, -1), (-2, 1), (-1, -2), (-1, 2),
            (1, -2), (1, 2), (2, -1), (2, 1)
        ]
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
        state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)
        q_values = []
        action_node_indices = []
        x, y = env.current_pos
        for action in valid_actions:
            dx, dy = env.knight_moves[action]
            target_x, target_y = x + dx, y + dy
            target_node_idx = env.pos_to_ind[(target_x, target_y)]
            q = self.primary_network(state_tensor, env.adjacency_list, target_node_idx, training=False).numpy()
            q_values.append(q)
            action_node_indices.append(target_node_idx)
        q_values = np.array(q_values)
        # Exploration
        if flagEvaluation or np.random.rand() > self.epsilon:
            action_idx = np.argmax(q_values)
        else: # Exploitation
            action_idx = np.random.choice(len(valid_actions))
        return valid_actions[action_idx], action_node_indices[action_idx] # 0-7, 0-Board_size^2
    
    def valid_moves_by_state(self, state):
        """
        Given a state array, returns the list of valid knight move indices (0-7).
        - state: np.ndarray of shape [num_squares, 4]
        """
        board_size = int(np.sqrt(state.shape[0]))
        knight_moves = [
            (-2, -1), (-2, 1), (-1, -2), (-1, 2),
            (1, -2), (1, 2), (2, -1), (2, 1)
        ]
        # Find current position (where state[:, 3] == 1)
        current_idx = np.argmax(state[:, 3])
        x, y = divmod(current_idx, board_size)
        valid_moves = []
        for i, (dx, dy) in enumerate(knight_moves):
            nx, ny = x + dx, y + dy
            if 0 <= nx < board_size and 0 <= ny < board_size:
                next_idx = nx * board_size + ny
                if state[next_idx, 2] == 0:  # Not visited
                    valid_moves.append(i)
        return valid_moves

    def _train_step(self, batch):
        """
        Performs a single training step (mini-batch update) for the DQN agent.
        Args:
            batch: List of tuples, each containing
                (state, adjacency_list, action_node_idx, reward, done, next_state, next_adjacency_list)
        Returns:
            grad: List of gradients for logging/debugging.
            loss: Scalar loss value for this training step.
        """
        # Record operations for automatic differentiation
        with tf.GradientTape() as tape:
            preds_state = []
            target = []
            for state, adjacency_list, action_node_idx, reward, done, next_state, next_adjacency_list in batch:
                state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)
                next_state_tensor = tf.convert_to_tensor(next_state, dtype=tf.float32)
                # Q-value for the (state, action) pair
                q_value = self.primary_network(state_tensor, adjacency_list, action_node_idx, training=True)
                # For next_state, get all valid next actions
                valid_next_actions = self.valid_moves_by_state(next_state)
                max_next_q = 0
                board_size = int(np.sqrt(next_state.shape[0]))
                # Find current position in next_state
                current_idx = np.argmax(next_state[:, 3])
                x, y = divmod(current_idx, board_size)
                for next_action in valid_next_actions:
                    dx, dy = self.knight_moves[next_action]
                    target_x, target_y = x + dx, y + dy
                    
                    target_node_idx = target_x * board_size + target_y
                    q = self.target_network(next_state_tensor, next_adjacency_list, target_node_idx, training=True).numpy()
                    if q > max_next_q:
                        max_next_q = q

                target_q = reward + self.gamma * max_next_q * (1.0 - float(done))
                preds_state.append(q_value)
                target.append(target_q)
            preds_state = tf.stack(preds_state)
            target = tf.convert_to_tensor(target, dtype=tf.float32)
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

    def train(self, iteration):
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
            - Every COPY_WEIGHTS_INTERVAL episodes, update the target network weights.
            - Run garbage collection to free memory.
        """
        for i in range(MULTI_FACTOR_BATCH):
            batch = random.sample(agent.memory, agent.numbersamples)
            grad, loss = agent._train_step(batch)
            wandb.log({"train/loss": loss.numpy()})
            # print(f"\n loss: {loss} \n")
       
        # Hard weights update
        if iteration % COPY_WEIGHTS_INTERVAL == 0:
            agent.target_network.set_weights(agent.primary_network.get_weights())
        gc.collect()
    
    def add_sample(self, state, adjacency_list, action_node_idx, reward, done, next_state, next_adjacency_list):
        """
        Store a transition in the replay buffer for Knight's Tour:
        (state, adjacency_list, action_node_idx, reward, done, next_state, next_adjacency_list)
        """
        self.memory.append((state, adjacency_list, action_node_idx, reward, done, next_state, next_adjacency_list))


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
    train_step = 0
    eval_step = 0

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(model=agent.primary_network, optimizer=agent.optimizer)

    rewards_test = np.zeros(EVALUATION_EPISODES)

    # # --- Initial Evaluation Before Training ---
    # for eps in tqdm(range(EVALUATION_EPISODES)):
    #     state = env_eval.reset()
    #     rewardAddTest = 0
    #     while True:
    #         action = agent.act(env_eval, state, True)  # Always exploit during evaluation
    #         next_state, reward, done, _ = env_eval.step(action)
    #         rewardAddTest += reward
    #         state = next_state
    #         if done:
    #             break
    #     rewards_test[eps] = rewardAddTest
    #     wandb.log({"initial_evaluation/episode_reward": rewardAddTest})
    #     wandb.log({"initial_evaluation/epsilon": agent.epsilon})


    # --- Main Training Loop ---
    for iteration in tqdm(range(ITERATIONS)): 

        # Use more training episodes at the start to fill replay buffer
        if iteration == 0:
            buffer_episode = FIRST_WORK_TRAIN_EPISODE
        else:
            buffer_episode = TRAINING_EPISODES
        for _ in tqdm(range(buffer_episode)): 
            tf.random.set_seed(1)  # For reproducibility
            state = env_training.reset()
            episode_reward = 0
            step_count = 0
            while True:
                action, action_node_idx = agent.act(env_training, state, False)  # Epsilon-greedy during training
                next_state, reward, done, _ = env_training.step(action)
                # Store transition in replay buffer
                agent.add_sample(state, env_training.adjacency_list, action_node_idx, reward, done, next_state, env_training.adjacency_list)
                state = next_state
                episode_reward += reward
                step_count += 1
                if done:
                    break
            train_step += 1    
            wandb.log({"train/episode_reward": episode_reward, "train/train_step": train_step, "train/epsilon": agent.epsilon})

        # Train the agent using experience replay
        agent.train(iteration)

        # --- Epsilon Decay (Exploration Rate) ---
        if iteration > EPSILON_START_DECAY and agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

        # --- Periodic Evaluation and Model Saving ---
        if iteration % EVALUATION_INTERVAL == 0:
            for eps in range(EVALUATION_EPISODES):
                state = env_eval.reset()
                rewardAddTest = 0
                while True:
                    action, _ = agent.act(env_eval, state, True)  # Always exploit during evaluation
                    next_state, reward, done, _ = env_eval.step(action)
                    rewardAddTest += reward
                    state = next_state
                    if done:
                        break
                rewards_test[eps] = rewardAddTest
                eval_step += 1
                wandb.log({"eval/episode_reward": rewardAddTest,"eval/eval_step": eval_step})
            evalMeanReward = np.mean(rewards_test)


            # Track and save the best model
            if evalMeanReward > max_reward:
                max_reward = evalMeanReward
                reward_id = iteration
            checkpoint.save(checkpoint_prefix)
            wandb.log({"model/max_reward": max_reward})
            #print(f"[Model] Saved checkpoint at iter {iteration} | Max Reward: {max_reward:.2f} | Model ID: {reward_id}")

