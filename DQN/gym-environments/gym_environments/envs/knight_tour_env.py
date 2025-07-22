import gym
import numpy as np
import networkx as nx
import random
from gym import error, spaces, utils
from random import choice
import pylab
import json 
import gc
import matplotlib.pyplot as plt



class KnightTourEnv(gym.Env):
    def __init__(self, board_size=5):
        super(KnightTourEnv, self).__init__()
        self.board_size = board_size
        self.num_squares = board_size * board_size
        self.pos_to_ind = {(i, j): i * self.board_size + j for i in range(self.board_size) for j in range(self.board_size)}
        self.ind_to_pos = {i * self.board_size + j: (i, j) for i in range(self.board_size) for j in range(self.board_size)}
        
        # 8 possible knight moves
        self.knight_moves = [
            (-2, -1), (-2, 1), (-1, -2), (-1, 2),
            (1, -2), (1, 2), (2, -1), (2, 1)
        ]

        # Action space: 8 possible moves
        self.action_space = spaces.Discrete(8)

        # Create the knight's move graph and adjacency matrix
        self.graph = self._create_knight_graph()
        self.adjacency_matrix = nx.adjacency_matrix(
            self.graph, nodelist=[self.pos_to_ind[(i, j)] for i in range(self.board_size) for j in range(self.board_size)]
        ).todense().astype(np.float32)
        # Build adjacency list: for each node, list of neighbor indices
        self.adjacency_list = [list(self.graph.neighbors(i)) for i in range(self.num_squares)]

        self.reset()

    def _create_knight_graph(self):
        G = nx.Graph()
        adjacency = {}
        for i in range(self.board_size):
            for j in range(self.board_size):
                G.add_node(i * self.board_size + j) # Add node index (0-24) to nodes list
                # adjacency[(i,j)]=[]
                # Add valid moves from each position as an edge
                for dx, dy in self.knight_moves:
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < self.board_size and 0 <= nj < self.board_size:
                        G.add_edge(i * self.board_size + j, ni * self.board_size + nj)
                        # adjacency[(i,j)].append((ni,nj))
        # print("adjacency dict:")
        # print(adjacency)
        return G

    def reset(self):
        self.visited = set()
        self.move_count = 0
        # Random starting position
        self.current_pos = (
            np.random.randint(0, self.board_size),
            np.random.randint(0, self.board_size)
        )
        self.visited.add(self.current_pos) # Adds the random starting position
        self.move_count = 1
        return self._get_state()

    def step(self, action):
        dx, dy = self.knight_moves[action]
        new_x = self.current_pos[0] + dx
        new_y = self.current_pos[1] + dy
        done = False
        reward = 0.0
        # Check if move is valid
        if (0 <= new_x < self.board_size and
            0 <= new_y < self.board_size and
            (new_x, new_y) not in self.visited):
            self.current_pos = (new_x, new_y)
            self.visited.add(self.current_pos)
            self.move_count += 1
            reward = 1.0 # Moved to an unvisited square

            if self.move_count == self.num_squares:
                done = True
            elif not self.get_valid_moves():
                done = True

        return self._get_state(), reward, done, {}

    def get_valid_moves(self, get_new_pos = False):
        valid_moves = []
        valid_new_positions = []
        x, y = self.current_pos
        for i, (dx, dy) in enumerate(self.knight_moves):
            new_x = x + dx
            new_y = y + dy
            if (0 <= new_x < self.board_size and
                0 <= new_y < self.board_size and
                (new_x, new_y) not in self.visited):
                valid_moves.append(i)
                valid_new_positions.append((new_x, new_y))
        if get_new_pos:
            return valid_new_positions
        return valid_moves

    def _get_state(self):
        # State: [x_norm, y_norm, visited, current_pos, degree_norm, betweenness]
        state = np.zeros((self.num_squares, 6), dtype=np.float32)

        # Calculate betweenness centrality on the subgraph of unvisited nodes
        unvisited_nodes = [self.pos_to_ind[pos] for pos in self.pos_to_ind if pos not in self.visited]
        
        betweenness = {}
        if len(unvisited_nodes) > 2:
            unvisited_graph = self.graph.subgraph(unvisited_nodes)
            betweenness = nx.betweenness_centrality(unvisited_graph, normalized=True, endpoints=False)

        for i in range(self.board_size):
            for j in range(self.board_size):
                idx = i * self.board_size + j
                pos = (i, j)
                state[idx, 0] = i / (self.board_size - 1) if self.board_size > 1 else 0.0
                state[idx, 1] = j / (self.board_size - 1) if self.board_size > 1 else 0.0
                state[idx, 2] = 1.0 if pos in self.visited else 0.0
                state[idx, 3] = 1.0 if pos == self.current_pos else 0.0
                
                # Calculate normalized degree
                degree = 0
                for dx, dy in self.knight_moves:
                    ni, nj = i + dx, j + dy
                    if (0 <= ni < self.board_size and 0 <= nj < self.board_size and (ni, nj) not in self.visited):
                        degree += 1
                state[idx, 4] = degree / 8.0

                # Assign betweenness centrality
                state[idx, 5] = betweenness.get(idx, 0.0)

        return state

    def print_board(self, valid_moves=False):
        """
        Prints the current board as a matrix, where 0 is unvisited and 1 is visited.
        The knight's current position is marked with 'K'.
        If valid_moves=True, valid moves are marked with 'X'.
        """
        board = np.zeros((self.board_size, self.board_size), dtype=object)
        valid_positions = set()
        if valid_moves:
            valid_positions = self.get_valid_moves(get_new_pos=True)
        for i in range(self.board_size):
            for j in range(self.board_size):
                if (i, j) == self.current_pos:
                    board[i, j] = 'K'
                elif (i, j) in valid_positions:
                    board[i, j] = 'X'
                elif (i, j) in self.visited:
                    board[i, j] = 1
                else:
                    board[i, j] = 0
        for row in board:
            print(' '.join(str(cell) for cell in row))


