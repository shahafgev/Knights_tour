import numpy as np
from knight_tour_env import KnightTourEnv

# Instantiate the environment
env = KnightTourEnv(board_size=5)

# # Print the initial board
# env.print_board(valid_moves=True)
# print("Initial knight position:", env.current_pos)
# print("Valid moves from start:", env.get_valid_moves())



# # Print adjacency matrix info
# adj = env.adjacency_list
# print("Adjacency list:\n", adj)


# # Print graph info
# print("Number of nodes in graph:", env.graph.number_of_nodes())
# print("Number of edges in graph:", env.graph.number_of_edges())

# Reset environment and print initial state
state = env.reset()
print("Initial state:\n", state)
print("Initial knight position:", env.current_pos)
print("Valid moves from start:", env.get_valid_moves(get_new_pos=True))

total_reward = 0
done = False

env.print_board(valid_moves=True)
while not done:
    # Take a valid step (if possible)
    valid_moves = env.get_valid_moves()

    action = valid_moves[0]
    next_state, reward, done, _ = env.step(action)
    total_reward += reward
    # print(f"\nAfter taking action {action}:")
    print("Reward:", reward)
    print("Total reward: ", total_reward)
    # print("Done:", done)
    # print("New knight position:", env.current_pos)
    # print("Valid moves:", env.get_valid_moves(get_new_pos=True))
    env.print_board(valid_moves=True)
    print("Updated state:\n", next_state)

    if done:
        print("No valid moves left")


# # Try an invalid move (e.g., move 0 if not valid)
# invalid_action = 0
# if invalid_action not in env.get_valid_moves():
#     state, reward, done, _ = env.step(invalid_action)
#     print(f"\nAfter trying invalid action {invalid_action}:")
#     print("Reward:", reward)
#     print("Done:", done)
#     print("Knight position:", env.current_pos) 