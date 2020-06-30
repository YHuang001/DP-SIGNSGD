from tensorflow import keras

import federated_learning_utils as fl_utils
import numpy as np
import tensorflow as tf

# Input parameters
tf.keras.backend.set_floatx('float64')
training_epochs = 11
number_of_nodes = 31
number_of_byzantines = 2
labels_per_node = 1
loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model_configs = {
    'data_shape': (28, 28),
    'layers': [(128, 'relu'), (10, 'softmax')]
}
train_datasets, test_datasets = fl_utils.AssignDatasets(number_of_nodes, labels_per_node, use_even_split=True, has_same_num_imgs=True)
attack_strategy = {
    'num_of_byzantines': number_of_byzantines,
    'attack_mode': 'best'
}
defend_strategy = ('error_feedback', 'credit_system')
defend_params = {
    # 'credit_weights': [tf.constant(1.0, dtype='float64') for node in range(number_of_nodes + number_of_byzantines)],
    'weight_decay': 0.6,
}
adv_parameters = {
    'loss_function': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    'learning_rate': 0.005,
    'batch_size': 80000,
    'use_vec_training': False,
    'clipping_value': 4,
    'enable_clipping': False,
    # 'b_value': 0.003,
    'attack_strategy': attack_strategy,
    'defend_strategy': defend_strategy,
    'defend_params': defend_params
}

# Simulation Part
federated_model = fl_utils.FederatedModel(model_configs, train_datasets, test_datasets, adv_parameters)
for _ in range(training_epochs):
    p_z, accuracy = federated_model.OneEpochTrainingSto()
    print(p_z, accuracy)
print(federated_model.TestAccuracy())

#################################################################
# You can set up manual training using functions in fl_utls without
# using the FedereratedModel as well for more advanced set up.
# ############################################################### 
