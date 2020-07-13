from tensorflow import keras

import federated_learning_utils as fl_utils
import numpy as np
import tensorflow as tf

# Input parameters
tf.keras.backend.set_floatx('float64')
training_epochs = 51
number_of_nodes = 31
number_of_byzantines = 3
labels_per_node = 1
loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model_configs = {
    'data_shape': (28, 28),
    'layers': [(128, 'relu'), (10, 'softmax')]
}
train_datasets, test_datasets = fl_utils.AssignDatasets(number_of_nodes, labels_per_node, use_even_split=False, has_same_num_imgs=False)
attack_strategy = {
    'num_of_byzantines': number_of_byzantines,
    'attack_mode': 'best'
}
defend_strategy = ()
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
    'attack_strategy': attack_strategy,
    'defend_strategy': defend_strategy,
    'defend_params': defend_params
}

# Simulation Part
federated_model = fl_utils.FederatedModel(model_configs, train_datasets, test_datasets, adv_parameters)
for epoch in range(training_epochs):
    p_z, accuracy = federated_model.OneEpochTrainingSto()
    if epoch % 10 == 0:
        print(p_z, accuracy)
print(federated_model.TestAccuracy())

#################################################################
# You can set up manual training using functions in fl_utls without
# using the FedereratedModel as well for more advanced set up.
# Example codes are as follows.
# ############################################################### 
# model = fl_utils.CreateModel(model_configs)
# attack_strategy = adv_parameters['attack_strategy']
# defend_strategy = adv_parameters['defend_strategy']
# defend_params = adv_parameters['defend_params']
# loss_function = adv_parameters['loss_function']
# batch_size = adv_parameters['batch_size']
# optimizer = tf.keras.optimizers.SGD(learning_rate=adv_parameters['learning_rate'])
# error_grads = [tf.zeros(model_layer.shape, dtype='float64') for model_layer in model.trainable_variables]
# if 'error_grads' not in defend_params:
#     defend_params['error_grads'] = [tf.zeros(model_layer.shape, dtype='float64')
#                                     for model_layer in model.trainable_variables]
# if 'credit_weights' not in defend_params:
#     defend_params['credit_weights'] = [[tf.ones(model.trainable_variables[layer].shape, dtype='float64') for layer in range(len(model.trainable_variables))]
#                                         for node in range(number_of_nodes + attack_strategy['num_of_byzantines'])]


# for epoch in range(training_epochs):
#     all_grads = fl_utils.CollectGradsNormal(train_datasets, model, batch_size, loss_function)
#     b_value = fl_utils.SetHeteroB(all_grads)
#     p_z, sto_transformed_grads, defend_params = fl_utils.CombineStoGradientsWithPEstimation(
#         all_grads, b_value, attack_strategy, defend_strategy, defend_params
#     )
#     optimizer.apply_gradients(zip(sto_transformed_grads, model.trainable_variables))
#     epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
#     accuracy = sum([float(epoch_accuracy(np.asarray(train_datasets[node][1]),
#                                             model(np.asarray(train_datasets[node][0])))) for node in range(number_of_nodes)]) / number_of_nodes
#     if epoch % 10 == 0:
#         print(p_z, accuracy)
# print(sum([float(epoch_accuracy(np.asarray(test_datasets[node][1]),
#                                 model(np.asarray(test_datasets[node][0])))) for node in range(number_of_nodes)]) / number_of_nodes)
