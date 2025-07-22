# Deep Reinforcement Learning for the Knight's Tour with Graph Neural Networks

## Project Overview
This project applies Deep Q-Learning (DQN) combined with Graph Neural Networks (GNNs) to solve the Knight's Tour problem on a chessboard. The Knight's Tour is a classic combinatorial problem where a knight must visit every square on the board exactly once. By modeling the board as a graph and leveraging GNNs, the agent learns to generalize move strategies across different board configurations.

## Approach
- **Environment:** The chessboard is represented as a graph, where each square is a node and edges represent valid knight moves. The environment is implemented as an OpenAI Gym environment (`KnightTourEnv`).
- **Agent:** The agent uses a DQN architecture, where the Q-network is a GNN that processes the board's graph structure and outputs Q-values for each possible knight move.
- **Learning:** The agent is trained via experience replay and periodically evaluated. The GNN enables the agent to learn spatial and structural patterns, improving generalization.

## Directory Structure
- `DQN/train_DQN.py` — Main training script for the DQN agent.
- `DQN/mpnn.py` — GNN (Message Passing Neural Network) model definition.
- `DQN/gym-environments/gym_environments/envs/knight_tour_env.py` — Knight's Tour Gym environment.
- `modelssample_DQN_agent/` — Directory for model checkpoints.

## Setup Instructions
1. **Install dependencies:**
   - Python 3.6+
   - Install required packages:
     ```bash
     pip install -r DQN/requirements.txt
     ```
2. **Install the custom Gym environment:**
   ```bash
   cd DQN/gym-environments
   pip install -e .
   cd ../..
   ```
3. **(Optional) Set up Weights & Biases (wandb) for experiment tracking.**

## Running Training
To train the DQN agent on the Knight's Tour problem:
```bash
python DQN/train_DQN.py
```

## Evaluation
The script performs periodic evaluation during training. Model checkpoints are saved in `modelssample_DQN_agent/`.

