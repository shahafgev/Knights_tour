import tensorflow as tf
from tensorflow import keras
from keras import regularizers
import numpy as np

class myModel(tf.keras.Model):
    def __init__(self, hparams):
        super(myModel, self).__init__()
        self.hparams = hparams

        self.InputProjection = keras.layers.Dense(self.hparams['nodes_state_dim'],activation=tf.nn.selu,name="InputProjection")
        # Message MLP: combines self and neighbor node features
        self.Message = tf.keras.models.Sequential()
        self.Message.add(keras.layers.Dense(self.hparams['nodes_state_dim'], activation=tf.nn.selu, name="MessageLayer"))
        # Update GRUCell for node states
        self.Update = keras.layers.GRUCell(self.hparams['nodes_state_dim'], dtype=tf.float32)
        # Readout MLP: from aggregated node states to Q-values
        self.Readout = keras.models.Sequential()
        self.Readout.add(keras.layers.Dense(self.hparams['readout_units'], activation=tf.nn.selu, kernel_regularizer=regularizers.l2(hparams['l2']), name="Readout1"))
        self.Readout.add(keras.layers.Dropout(rate=hparams['dropout_rate']))
        self.Readout.add(keras.layers.Dense(self.hparams['readout_units'], activation=tf.nn.selu, kernel_regularizer=regularizers.l2(hparams['l2']), name="Readout2"))
        self.Readout.add(keras.layers.Dropout(rate=hparams['dropout_rate']))
        self.Readout.add(keras.layers.Dense(8, kernel_regularizer=regularizers.l2(hparams['l2']), name="QValues"))


    def build(self, input_shape=None):
        """
        Builds the sub-layers of the model with the correct input shapes.
        This method is called to initialize the weights of the sub-layers based on the expected input dimensions.

        Args:
            input_shape: Optional shape of the input (not used directly here).
        """
        self.InputProjection.build(input_shape=tf.TensorShape([None, self.hparams['input_shape']]))
        # Build the Message layer to accept concatenated node features (self + neighbor)
        self.Message.build(input_shape=tf.TensorShape([None, self.hparams['nodes_state_dim'] * 2]))
        # Build the Update GRUCell to accept node hidden states of size nodes_state_dim
        self.Update.build(input_shape=tf.TensorShape([None, self.hparams['nodes_state_dim']]))
        # Build the Readout MLP to accept the aggregated graph embedding of size nodes_state_dim
        self.Readout.build(input_shape=[None, self.hparams['nodes_state_dim']])
        # Mark the model as built
        self.built = True

    @tf.function
    def call(self, node_features, adjacency_list, training=False):
        # node_features: (num_nodes, input_dim)
        # adjacency_list: list of lists, each sublist contains neighbor indices for that node
        # Project input features to hidden dimension
        nodes_state = self.InputProjection(node_features)  # [num_nodes, nodes_state_dim]
        for _ in range(self.hparams['T']):
            messages = []

            for i in range(nodes_state.shape[0]):
                neighbors = adjacency_list[i] # indexes of neighbors
                if tf.size(neighbors) == 0:
                    agg_message = tf.zeros((self.hparams['nodes_state_dim'],), dtype=nodes_state.dtype)
                else:
                    neighbors_state = tf.gather(nodes_state, neighbors) # exatract the featurs of the neighboring nodes
                                                                        # [num_neighbors, num_node_feturs] 
                    self_state = tf.repeat(# reapeats the nodes state * num of neighbors
                                            # epeats the [1, nodes_state_dim] tensor num_neighbors times along axis 0.
                                            tf.expand_dims(nodes_state[i], 0), # [1,nodes_state_dim]
                                            tf.shape(neighbors_state)[0], # [num_neighbors]
                                            axis=0
                                            ) # output shape: [num_neighbors, nodes_state_dim]
                    concat = tf.concat([self_state, neighbors_state], axis=1)# [num_neighbors, 2 * nodes_state_dim]
                    msgs = self.Message(concat) # [num_neighbors, nodes_state_dim]
                    agg_message = tf.reduce_mean(msgs, axis=0) # [nodes_state_dim]
                messages.append(agg_message) 
            messages = tf.stack(messages, axis=0)# [num_nodes, nodes_state_dim]
            # Update node states
            _, nodes_state = self.Update(messages, [nodes_state])  # nodes_state: ([num_nodes, nodes_state_dim],)
            nodes_state = nodes_state[0]  # [num_nodes, nodes_state_dim]
        # Aggregate node states (sum or mean)
        graph_embedding = tf.reduce_sum(nodes_state, axis=0, keepdims=True)  # [1, nodes_state_dim]
        q_values = self.Readout(graph_embedding, training=training)  # [1, 8]
        q_values = tf.squeeze(q_values, axis=0)  # [8]
        return q_values


# if __name__ == "__main__":
#     # Define hyperparameters
#     hparams = {
#         'nodes_state_dim': 8,       # hidden state size for each node
#         'readout_units': 16,       # hidden units in readout MLP
#         'l2': 0.0,
#         'dropout_rate': 0.0,
#         'T': 2,                    # number of message passing steps
#     }

#     # Instantiate the model
#     model = myModel(hparams)
#     model.build()
#     print("Model built successfully.")

#     # Test with actual KnightTourEnv board
#     import sys
#     sys.path.append("./DQN/gym-environments")
#     from gym_environments.envs.knight_tour_env import KnightTourEnv

#     board_size = 5
#     env = KnightTourEnv(board_size=board_size)
#     state = env.reset()  # shape: (num_squares, 3)
#     print("KnightTourEnv initial state shape:", state.shape)

#     adjacency_list = env.adjacency_list

#     node_features_tf = tf.convert_to_tensor(state)
    
#     # Call the model
#     q_values = model(node_features_tf, adjacency_list, training=False)
#     print("Q-values from real KnightTourEnv board:", q_values.numpy())
#     print("Q-values shape:", q_values.shape)
#     assert q_values.shape == (8,)
#     print("KnightTourEnv test passed!")


