"""
This file includes all utility funtions needed to perform the simulation on federated learning.
DO NOT write your own main simulation function in this file, IMPORT this file to your simulation function.
If you want to add new functions to this file, please FOLLOW THE SAME FORMAT as the existing functions.
"""
import tensorflow as tf
from tensorflow import keras

from collections import defaultdict
import numpy as np
import functools
import heapq
import math


def SameLabelSplitDataReserve(
    nodes, imgs_by_label, labels_per_node,
    num_imgs_per_node, total_images_with_reserve_by_label):
    """Splits the whole dataset into sub-datasets with the same number of distinct labels using the reserved mechanism.

    Args:
        nodes: Number of nodes to whom the splitted sub-datasets belong.
        imgs_by_label: List of image lists where the index for each image list stands for their label.
        labels_per_node: Number of distinct labels in the sub-dataset for one node.
        num_imgs_per_node: Number of images in the sub-dataset splitted, the final result will be close to
            this value, but not necessary equal to, it depends on the number of nodes.
        total_images_with_reserve_by_label: The total number of images for each label after excluding the reserved images.
            This is needed to ensure that all nodes can be assigned with enough distinct labels.

    Returns:
        A dictionary of image and label lists store all sub-datasets for all nodes and each key value represents the node index.
        E.g., dataset_by_node[1][0] stores all images for node 1 and dataset_by_node[1][1] has all the corresponding labels.
    """
    dataset_by_node = defaultdict(list)
    # Label candidates from which the labels are chosen for each node's sub-dataset.
    label_candidates = list(range(10))
    # The image index for each label such that those images with indexes after it are not assigned yet.
    unassigned_img_ids = [0] * 10
    num_imgs_per_label = num_imgs_per_node // labels_per_node

    for node in range(nodes):
        # Randomly chose {labels_per_node} distinct labels from the label candidates.
        chosen_labels = np.random.choice(label_candidates, labels_per_node, replace=False)
        assert (len(set(chosen_labels)) == labels_per_node), (
            f'There should be {labels_per_node} distinct labels, but got {len(set(chosen_labels))} distinct labels.')
        for label in chosen_labels:
            current_unassigned_img_id = unassigned_img_ids[label]
            # Find out the maximum id for the chosen label.
            current_label_id_ceiling = total_images_with_reserve_by_label[label]
            new_unassigned_img_id = current_unassigned_img_id + num_imgs_per_label
            # If the current unassigned image id is lower than the maximum id and the new unassigned image id is higher than
            # the maximum id, set the maximum id as the new unassigned image id meaning that use all images up to the reserved ones.
            if current_unassigned_img_id < current_label_id_ceiling and new_unassigned_img_id >= current_label_id_ceiling:
                new_unassigned_img_id = current_label_id_ceiling
            # If the current unassigned image id is greater than the maximum id meaning that all images except for the reserved
            # ones are assigned, now use reserved images one by one.
            elif current_unassigned_img_id >= current_label_id_ceiling:
                new_unassigned_img_id = current_unassigned_img_id + 1
            assigned_images = imgs_by_label[label][current_unassigned_img_id: new_unassigned_img_id]
            assigned_labels = [label] * len(assigned_images)
            if len(dataset_by_node[node]) == 0:
                dataset_by_node[node].append(assigned_images)
                dataset_by_node[node].append(assigned_labels)
            else:
                dataset_by_node[node][0] += assigned_images
                dataset_by_node[node][1] += assigned_labels
            unassigned_img_ids[label] = new_unassigned_img_id
    return dataset_by_node

def SameLabelSplitDataEven(
    nodes, imgs_by_label, labels_per_node, num_imgs_per_node, has_same_num_imgs = False):
    """Splits the whole dataset into sub-datasets with the same number of distinct labels as even as possible.

    Splits the dataset in a way such that each node has the same number of distinct labels and the number of images
    in each sub-data are as close as possible or even identical if the corresponding flag is set to true.

    Args:
        nodes: Number of nodes to whom the splitted sub-datasets belong.
        imgs_by_label: List of image lists where the index for each image list stands for their label.
        labels_per_node: Number of distinct labels in the sub-dataset for one node.
        num_imgs_per_node: Number of images in the sub-dataset splitted, the final result will be close to
            this value, but not necessary equal to, it depends on the number of nodes.
        has_same_num_imgs: If set to true, the final splitted sub-datasets will have the same number of images,
            but the number is not necessarily equal to num_imgs_per_node, it depends on the number of nodes.
    
    Returns:
        A dictionary of image and label lists store all sub-datasets for all nodes and each key value represents the node index.
        E.g., dataset_by_node[1][0] stores all images for node 1 and dataset_by_node[1][1] has all the corresponding labels.
    """
    dataset_by_node = defaultdict(list)
    # Number of segments in each label neede for all images to be split evenly to all nodes.
    segments = math.ceil(nodes * labels_per_node / 10)
    num_imgs_by_label = [len(imgs_by_label[label]) for label in range(10)]
    if has_same_num_imgs:
        # If same number of images per node is needed, be conservative in assigning the number of images.
        min_imgs_of_all_labels = min(num_imgs_by_label)
        num_imgs_per_segment = min(min_imgs_of_all_labels // segments, num_imgs_per_node // labels_per_node)
    else:
        num_imgs_per_segment = [num_imgs // segments for num_imgs in num_imgs_by_label]
    # Set up label mask so that the output splitted sub-datasets are different each time using this function
    # with the same inputs.
    label_mask = np.arange(10)
    np.random.shuffle(label_mask)

    # The mechanism is to store the unassigned segment id for all labels, assign the segment of the label with
    # the smallest unassigned segment id (least assigned label), this is implemented using the heap.
    segment_ids = []
    heapq.heapify(segment_ids)
    for label in range(10):
        heapq.heappush(segment_ids, (0, label))
    
    for node in range(nodes):
        used_labels = []
        for _ in range(labels_per_node):
            current_segment_id, label = heapq.heappop(segment_ids)
            actual_label = label_mask[label]
            if has_same_num_imgs:
                imgs_per_segment = num_imgs_per_segment
            else:
                imgs_per_segment = num_imgs_per_segment[actual_label]
            assigned_images = imgs_by_label[actual_label][current_segment_id * imgs_per_segment:
                                                          (current_segment_id + 1) * imgs_per_segment]
            assigned_labels = [actual_label] * len(assigned_images)

            if len(dataset_by_node[node]) == 0:
                dataset_by_node[node].append(assigned_images)
                dataset_by_node[node].append(assigned_labels)
            else:
                dataset_by_node[node][0] += assigned_images
                dataset_by_node[node][1] += assigned_labels
            
            used_labels.append((current_segment_id + 1, label))
        
        for used_label in used_labels:
            heapq.heappush(segment_ids, used_label)
    
    return dataset_by_node

def AssignDatasets(nodes, labels_per_node, use_even_split=False, has_same_num_imgs=False):
    """Assigns the training and testing datasets to all nodes.

    Args:
        nodes: Number of nodes to whom the splitted sub-datasets belong.
        labels_per_node: Number of distinct labels in the sub-dataset for one node.
        use_even_split: If set to true, use the even split way to split the sub-datasets.
        has_same_num_imgs: If set to true, the final splitted sub-datasets will have the same number of images,
            but the number is not necessarily equal to num_imgs_per_node, it depends on the number of nodes.
    
    Returns:
        All the training and testing sub-datasets for all nodes.
    """
    if labels_per_node > 10:
        raise ValueError(f'The input number of labels per node is {labels_per_node}, which exceeds the total number of labels!')
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0

    num_train_imgs_per_node = len(train_images) // nodes
    num_test_imgs_per_node = len(test_images) // nodes

    train_images_by_label = defaultdict(list)
    test_images_by_label = defaultdict(list)

    for i, image in enumerate(train_images):
        train_images_by_label[train_labels[i]].append(image)
    for i, image in enumerate(test_images):
        test_images_by_label[test_labels[i]].append(image)
    
    if use_even_split:
        train_dataset_by_node = SameLabelSplitDataEven(nodes, train_images_by_label,
                                                       labels_per_node, num_train_imgs_per_node, has_same_num_imgs)
        test_dataset_by_node = SameLabelSplitDataEven(nodes, test_images_by_label,
                                                      labels_per_node, num_test_imgs_per_node, has_same_num_imgs)
    else:
        total_train_images_with_reserve_by_label = [len(train_images_by_label[label]) - nodes for label in range(10)]
        total_test_images_with_reserve_by_label = [len(test_images_by_label[label]) - nodes for label in range(10)]
        train_dataset_by_node = SameLabelSplitDataReserve(nodes, train_images_by_label,
                                                          labels_per_node, num_train_imgs_per_node,
                                                          total_train_images_with_reserve_by_label)
        test_dataset_by_node = SameLabelSplitDataReserve(nodes, test_images_by_label,
                                                         labels_per_node, num_test_imgs_per_node,
                                                         total_test_images_with_reserve_by_label)
    
    return train_dataset_by_node, test_dataset_by_node     


# Model and gradient utility functions.
def CreateModel(model_configs):
    """Creates a sequentail model given the model config.

    Args:
        model_configs: The layer configurations including the layer size, activation type, and drop outs.
    
    Returns:
        A sequential model.
    """
    base_model = [keras.layers.Flatten(input_shape=model_configs['data_shape'])]
    for layer in model_configs['layers']:
        if layer[0] == 'dropout':
            base_model.append(keras.layers.Dropout(layer[1]))
        else:
            base_model.append(keras.layers.Dense(layer[0], activation=layer[1]))
    return keras.Sequential(base_model)

def SignGrads(grads):
    """Finds the signs for the gradients computed.

    Args:
        grads: Input gradients.
    
    Returns:
        Sign gradients correspond to the input gradients.
    """
    return [tf.sign(grad) for grad in grads]

def Grad(model, loss_function, inputs, targets):
    """Computes the model gradients.

    Args:
        model: The model for which the gradients are computed.
        loss_function: The loss function used to compute the gradients.
        inputs: The inputs for computing the gradients.
        labels: The targets for computing the gradients.

    Returns:
        Model gradients.
    """
    with tf.GradientTape() as tape:
        loss_value = loss_function(y_true=targets, y_pred=model(inputs))
    return tape.gradient(loss_value, model.trainable_variables)

def CollectGradsNormal(datasets, model, batch_size, loss_function):
    """Collects all gradients without clipping.

    If clipping is not needed, use this function for collecting
    gradients for all nodes for performance consideration.

    Args:
        datasets: Datasets for training.
        model: Model to be trained.
        batch_size: Batch size for the batch training.
        loss_function: The loss function used to compute the gradients.
    
    Returns:
        All gradients for all nodes.
    """
    all_grads = []
    for node in range(len(datasets)):
        data_size = len(datasets[node][0])
        if data_size <= batch_size:
            batched_images = np.asarray(datasets[node][0])
            batched_labels = np.asarray(datasets[node][1])
        else:
            candidate_indexes = np.random.choice(list(range(data_size)), batch_size, replace=False)
            batched_images = np.asarray([datasets[node][0][index] for index in candidate_indexes])
            batched_labels = np.asarray([datasets[node][1][index] for index in candidate_indexes])
        all_grads.append(Grad(model, loss_function, batched_images, batched_labels))
    return all_grads

def CollectGradsVec(datasets, model, batch_size, enable_clipping, clipping_value):
    """Collects all gradients with batch training, clipping function included.

    Args:
        datasets: Datasets for training.
        model: Model to be trained.
        batch_size: Batch size for the batch training.
        enable_clipping: If set to true, clip the gradients to the clipping value.
    
    Returns:
        All gradients for all nodes.
    """
    def BatchedGrads(args):
        inputs, targets = args
        with tf.GradientTape() as tape:
            inputs = tf.expand_dims(inputs, 0)
            targets = tf.one_hot(targets, 10)
            predictions = model(inputs)
            # This is essentially equivalent to tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            # with an additional reduce mean step.
            loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true=targets, y_pred=predictions, from_logits=True))

        grads = tape.gradient(loss, model.trainable_variables)
        if enable_clipping:
            global_norm = tf.math.sqrt(tf.add_n([tf.square(tf.norm(grad)) for grad in grads]))
            return [grad / tf.maximum(tf.constant(1.0, dtype='float64'), global_norm / clipping_value) for grad in grads]
        return grads
    
    all_grads = []
    for node in range(len(datasets)):
        data_size = len(datasets[node][0])
        if data_size <= batch_size:
            batched_images = np.asarray(datasets[node][0])
            batched_labels = np.asarray(datasets[node][1])
        else:
            candidate_indexes = np.random.choice(list(range(data_size)), batch_size, replace=False)
            batched_images = np.asarray([datasets[node][0][index] for index in candidate_indexes])
            batched_labels = np.asarray([datasets[node][1][index] for index in candidate_indexes])
        batched_images = tf.convert_to_tensor(batched_images)
        per_example_gradients = tf.vectorized_map(BatchedGrads, (batched_images, batched_labels))
        all_grads.append([tf.reduce_sum(grad, axis = 0) for grad in per_example_gradients])
    return all_grads        

def CombineGrads(all_grads):
    """Combines all gradients through summation without any transformation.

    Args:
        all_grads: Gradients of all model variables.
    
    Returns:
        The combined gradients.
    """
    return [sum(grad[layer] for grad in all_grads) for layer in range(len(all_grads[0]))]

def SetHeteroB(all_grads):
    """Sets the b values for each gradient element based on the maximum the absolute values of all gradients on this entry.

    Args:
        all_gradients: All gradients collected from all nodes.
    
    Returns:
        Heterogeneous b values which has the same shape as the model gradients.
    """
    max_grads = [functools.reduce(lambda a, b: tf.maximum(tf.abs(a), tf.abs(b)), [grad[layer] for grad in all_grads])
                 for layer in range(len(all_grads[0]))]

    return max_grads

def StoTransformation(grads, b_value):
    """Performs the sto transformation on the gradients.

    Args:
        grads: Input gradients.
        b_value: B values required for the sto transformation.
    
    Returns:
        Gradients after the sto transformation.
    """
    lower_bound = 1e-20
    transformed_grads = []
    for i, grad in enumerate(grads):
        random_tensor = tf.random.uniform(grad.shape, minval=0, maxval=1, dtype='float64')
        if isinstance(b_value, list):
            new_grad = grad / (2 * np.maximum(b_value[i], lower_bound)) + 0.5
        else:
            new_grad = grad / (2 * np.maximum(b_value, lower_bound)) + 0.5
        compare_tensor = tf.math.less(random_tensor, new_grad)
        transformed_grads.append(tf.sign(tf.dtypes.cast(compare_tensor, dtype='float64') - 0.5))
    return transformed_grads

def CombineStoGrads(all_grads, ori_grads_sign, b_value, attack_strategy, defend_strategy, defend_params):
    """Combines all gradients through sto transformation and further applies the defend strategy.

    Args:
        all_grads: Gradients of all model variables.
        ori_grads_sign: Gradients with each element being the sign of the combined gradients without any transformation.
        b_value: B values for the sto transformation.
        attack_strategy: Attack strategies adopted by the attacker, including the number of byzantine attackers and the attack mode.
        defend_strategy: The defend strategy applied. Currently, it can be the error feedback or the credit system.
        defend_params: The defend parameters correspond to the defend strategy applied.
    
    Returns:
        Final combined gradients and updated defend parameters.
    """
    # Report grads from each node after the sto transformation and weights applied (if weights exist).
    credit_weights = defend_params['credit_weights']
    normal_nodes = len(all_grads)
    use_per_entry_weight = isinstance(credit_weights[0], list)
    reported_transformed_grads = []
    weighted_transformed_grads = []
    for node, grads in enumerate(all_grads):
        current_sto_transformed_grads = StoTransformation(grads, b_value)
        reported_transformed_grads.append(current_sto_transformed_grads)
        if use_per_entry_weight:
            weighted_grads = [current_sto_transformed_grads[layer] * credit_weights[node][layer]
                              for layer in range(len(grads))]
        else:
            weighted_grads = [current_sto_transformed_grads[layer] * credit_weights[node]
                              for layer in range(len(grads))]
        weighted_transformed_grads.append(weighted_grads)
    
    # Report grads from attackers based on their attack mode.
    combined_grads = CombineGrads(weighted_transformed_grads)
    num_of_byzantines = attack_strategy['num_of_byzantines']
    attack_mode = attack_strategy['attack_mode']
    for byzantine_id in range(num_of_byzantines):
        attacker_full_grads = []
        current_node = normal_nodes + byzantine_id
        for layer in range(len(combined_grads)):
            if attack_mode == 'best':
                attack_grads = -1.0 * ori_grads_sign[layer]
            elif attack_mode == 'random':
                attack_grads = tf.sign(tf.random.uniform(ori_grads_sign[layer].shape, minval=-1.0, maxval=1.0, dtype='float64'))
            else:
                raise TypeError(f'{attack_mode} is not supported!')
            attacker_full_grads.append(attack_grads)
            weights = credit_weights[current_node][layer] if use_per_entry_weight else credit_weights[current_node]
            weighted_attack_grads = attack_grads * weights
            combined_grads[layer] += weighted_attack_grads
        reported_transformed_grads.append(attacker_full_grads)
    
    # Final grads processing.
    error_grads = defend_params['error_grads']
    total_nodes = num_of_byzantines + normal_nodes
    for layer in range(len(combined_grads)):
        combined_grads[layer] /= total_nodes
        combined_grads[layer] += error_grads[layer]
    
    final_grads = [tf.sign(grad) for grad in combined_grads]
    total_grads = sum([functools.reduce(lambda a, b: a*b, list(grad.shape)) for grad in final_grads])
    updated_defend_params = defend_params.copy()
    
    # Update defend parameters
    if 'error_feedback' in defend_strategy:
        updated_defend_params['error_grads'] = [combined_grads[layer] - final_grads[layer] / total_nodes
                                                for layer in range(len(final_grads))]
    if 'credit_system' in defend_strategy:
        weight_decay = defend_params['weight_decay']
        # updated_normalization_value = weight_decay * defend_params['normalization_value'] + 1.0
        updated_credit_weights = []
        for node in range(total_nodes):
            if use_per_entry_weight:
                # If the sign of the reported gradient sentry is equal to the corresponding entry in the final grads,
                # the corresponding weight is 1. Otherwise, it is set to -1.
                new_credit_weights = [
                    (tf.sign(tf.dtypes.cast(tf.equal(tf.sign(reported_transformed_grads[node][layer]), final_grads[layer]), dtype='float64') - 0.5)
                     + credit_weights[node][layer] * weight_decay)
                    for layer in range(len(final_grads))]
            else:
                num_of_equal_grads = sum([
                    tf.reduce_sum(tf.sign(tf.dtypes.cast(tf.equal(tf.sign(reported_transformed_grads[node][layer]), final_grads[layer]), dtype='float64') - 0.5))
                    for layer in range(len(final_grads))
                ])
                new_credit_weights = credit_weights[node] * weight_decay + num_of_equal_grads / total_grads
            updated_credit_weights.append(new_credit_weights)
        updated_defend_params['credit_weights'] = updated_credit_weights
    return final_grads, updated_defend_params

def CombineStoGradientsWithPEstimation(all_grads, b_value, attack_strategy, defend_strategy, defend_params):
    """Combines all gradients using the sto transformation and estimates the portion of sign gradients no equal to the sign of the true gradients.

    Args:
        all_grads: Gradients of all model variables.
        b_value: B values for the sto transformation.
        attack_strategy: Attack strategies adopted by the attacker, including the number of byzantine attackers and the attack mode.
        defend_strategy: The defend strategy applied in sto transformation.
            Currently, it can be the error feedback or the credit system.
        defend_params: The defend parameters correspond to the defend strategy applied in sto transformation.

    Returns:
        Portion of gradients with different signs compared to the true gradients, gradients after sto transformation,
        and error gradients.
    """
    ori_grads = CombineGrads(all_grads)
    ori_grads_sign = SignGrads(ori_grads)
    sto_transformed_grads, updated_defend_params = CombineStoGrads(all_grads, ori_grads_sign, b_value, attack_strategy,
                                                                   defend_strategy, defend_params)
    total_grads = sum([functools.reduce(lambda a, b: a*b, list(grad.shape)) for grad in sto_transformed_grads])
    non_equal_grads = 0
    for i in range(len(sto_transformed_grads)):
        non_equal_grads += tf.math.reduce_sum(tf.dtypes.cast(tf.math.not_equal(sto_transformed_grads[i], ori_grads_sign[i]), dtype="float64"))
    
    return non_equal_grads / total_grads, sto_transformed_grads, updated_defend_params

# Models used for training and testing.
class FederatedModel(object):
    """A sequential model with dense layers used for ferderated learning."""
    def __init__(self, model_configs, train_datasets, test_datasets, adv_parameters):
        """Initializes model with the necessary components.

        Args:
            model_configs: The layer configuration including the layer size, activation type, and drop outs. This should
                be a dictionary.
                For example, model_configs = {
                    'data_shape': (28, 28),
                    'layers': [(512, 'relu'), ('dropout', 0.2), (10, 'softmax')]}.
            loss_function: The loss function used to compute the loss.
            learning_rate: The learning rate for training the model.
            train_datasets: Datasets used for training the model.
            test_datasets: Datasets used for testing the model.
            adv_parameters: Additional parameters used for advanced features like STO transformation and gradient clipping.
                This should be a dictionary.
                For example, adv_parameters = {
                    'loss_function': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    'learning_rate': 0.01,
                    'clipping_value': 4,
                    'enable_clipping': True,
                    'batch_size': 100
                    ...
                }
        """
        self._model = CreateModel(model_configs)
        self._train_datasets = train_datasets
        self._nodes = len(self._train_datasets)
        self._test_datasets = test_datasets
        self._adv_parameters = adv_parameters
        self._use_vec_training = self._adv_parameters['use_vec_training']
        self._optimizer = tf.keras.optimizers.SGD(learning_rate=self._adv_parameters['learning_rate'])
        self._defend_params = adv_parameters['defend_params']
        total_nodes = self._nodes + self._adv_parameters['attack_strategy']['num_of_byzantines']
        if 'error_grads' not in self._defend_params:
            self._defend_params['error_grads'] = [tf.zeros(model_layer.shape, dtype='float64')
                                                  for model_layer in self._model.trainable_variables]
        if 'credit_weights' not in self._defend_params:
            self._defend_params['credit_weights'] = [[tf.ones(self._model.trainable_variables[layer].shape, dtype='float64') for layer in range(len(self._model.trainable_variables))]
                                                      for node in range(total_nodes)]

    def OneEpochTrainingSto(self):
        """Trains the model with clipping options and sto transformation using the vectorized mapping."""
        batch_size = self._adv_parameters['batch_size']
        loss_function = self._adv_parameters['loss_function']
        clipping_value, enable_clipping = self._adv_parameters['clipping_value'], self._adv_parameters['enable_clipping']
        datasets = self._train_datasets
        nodes = self._nodes
        model = self._model
        
        attack_strategy = self._adv_parameters['attack_strategy']
        defend_strategy = self._adv_parameters['defend_strategy']

        if self._use_vec_training:
            all_grads = CollectGradsVec(datasets, model, batch_size, enable_clipping, clipping_value)
        else:
            all_grads = CollectGradsNormal(datasets, model, batch_size, loss_function)
        b_value = self._adv_parameters.get(
            'b_value',
            SetHeteroB(all_grads)
        )
        p_z, sto_transformed_grads, self._defend_params = CombineStoGradientsWithPEstimation(
            all_grads, b_value, attack_strategy, defend_strategy, self._defend_params)
        self._optimizer.apply_gradients(zip(sto_transformed_grads, model.trainable_variables))
        epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        accuracy = sum([float(epoch_accuracy(np.asarray(datasets[node][1]),
                                             model(np.asarray(datasets[node][0])))) for node in range(nodes)]) / nodes
        return p_z, accuracy
    
    def TestAccuracy(self):
        """Computes the test accuracy for the trained model."""
        datasets = self._test_datasets
        nodes = self._nodes
        model = self._model
        epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        return sum([float(epoch_accuracy(np.asarray(datasets[node][1]),
                                         model(np.asarray(datasets[node][0])))) for node in range(nodes)]) / nodes