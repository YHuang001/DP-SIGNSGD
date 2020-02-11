import tensorflow as tf
from tensorflow import keras

from collections import defaultdict
import numpy as np
import pandas as pd
import random
import math
import time
from sklearn.decomposition import PCA
from itertools import islice 
import heapq

# Split the whole dataset to datasets for all nodes in a way that any two nodes do not share any data sample and all nodes' datasets have the same number of distinct labels
def SameLabelSplitData(nodes, images_by_label, labels_by_node, total_images_by_label_with_reserve, num_imgs_per_label):
    dataset_by_node = defaultdict(list)
    candidate_pool = list(range(10))
    current_img_ids = [0]*10
    for node in range(nodes):
        chosen_labels = np.random.choice(candidate_pool, labels_by_node, replace=True)
        for label in chosen_labels:
            current_img_id = current_img_ids[label]
            current_label_ceiling = total_images_by_label_with_reserve[label]
            new_img_id = current_img_id + num_imgs_per_label
            if current_img_id < current_label_ceiling and new_img_id >= current_label_ceiling:
                new_img_id = current_label_ceiling
            elif current_img_id >= current_label_ceiling:
                new_img_id = current_img_id + 1
            images = images_by_label[label][current_img_id : new_img_id]
            labels = [label]*len(images)
            if len(dataset_by_node[node]) == 0:
                dataset_by_node[node].append(images)
                dataset_by_node[node].append(labels)
            else:
                dataset_by_node[node][0] += images
                dataset_by_node[node][1] += labels
            current_img_ids[label] = new_img_id
    return dataset_by_node

# Assign datasets for all nodes from the MNIST dataset
def AssignDatasets(nodes, min_labels, have_same_label_number=True):
    mnist = keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images, test_images = train_images/255.0, test_images/255.0

    train_dataset = zip(train_images, train_labels)
    test_dataset = zip(test_images, test_labels)

    sorted_train_dataset = sorted(train_dataset, key=lambda x: x[1])
    sorted_test_dataset = sorted(test_dataset, key=lambda x: x[1])

    num_train_images_per_label = int(len(train_images)/nodes/min_labels)
    num_test_images_per_label = int(len(test_images)/nodes/min_labels)

    train_images_by_label = defaultdict(list)
    test_images_by_label = defaultdict(list)

    for i, image in enumerate(train_images):
        train_images_by_label[train_labels[i]].append(image)
    for i, image in enumerate(test_images):
        test_images_by_label[test_labels[i]].append(image)
    
    total_train_images_by_label_with_reserve = [len(train_images_by_label[label]) - nodes for label in range(10)]
    total_test_images_by_label_with_reserve = [len(test_images_by_label[label]) - nodes for label in range(10)]
    
    train_dataset_by_node = defaultdict(list)
    test_dataset_by_node =defaultdict(list)

    if min_labels > 10:
        raise ValueError("Minimum number of labels is {}, which exceeds the total number of labels!".format(min_labels))
    if have_same_label_number:
        train_dataset_by_node = SameLabelSplitData(nodes, train_images_by_label, min_labels, total_train_images_by_label_with_reserve, num_train_images_per_label)
        test_dataset_by_node = SameLabelSplitData(nodes, test_images_by_label, min_labels, total_test_images_by_label_with_reserve, num_test_images_per_label)
    else:
        for label in range(min_labels):
            if label == 0:
                for node in range(nodes):
                    current_set = nodes * label + node
                    train_dataset_by_node[node].append([data[0] for data in sorted_train_dataset[current_set*num_train_images_per_label:(current_set+1)*num_train_images_per_label]])
                    train_dataset_by_node[node].append([data[1] for data in sorted_train_dataset[current_set*num_train_images_per_label:(current_set+1)*num_train_images_per_label]])
                    test_dataset_by_node[node].append([data[0] for data in sorted_test_dataset[current_set*num_test_images_per_label:(current_set+1)*num_test_images_per_label]])
                    test_dataset_by_node[node].append([data[1] for data in sorted_test_dataset[current_set*num_test_images_per_label:(current_set+1)*num_test_images_per_label]])
            else:
                for node in range(nodes):
                    current_set = nodes * label + node
                    train_dataset_by_node[node][0] += [data[0] for data in sorted_train_dataset[current_set*num_train_images_per_label:(current_set+1)*num_train_images_per_label]]
                    train_dataset_by_node[node][1] += [data[1] for data in sorted_train_dataset[current_set*num_train_images_per_label:(current_set+1)*num_train_images_per_label]]
                    test_dataset_by_node[node][0] += [data[0] for data in sorted_test_dataset[current_set*num_test_images_per_label:(current_set+1)*num_test_images_per_label]]
                    test_dataset_by_node[node][1] += [data[1] for data in sorted_test_dataset[current_set*num_test_images_per_label:(current_set+1)*num_test_images_per_label]]

    return train_dataset_by_node, test_dataset_by_node

# Create the deep learning model
def CreateModel(data_shape):
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=data_shape),
        keras.layers.Dense(128, activation='relu', dtype='float64'),
        keras.layers.Dense(10, activation='softmax', dtype='float64')
    ])
    return model

# The loss function which uses sparse categorical entropy
def Loss(model, x, y):
    y_ = model(x)
    return LOSS_OBJECT(y_true=y, y_pred=y_)

# Calculate the gradient based on the inputs, model, and loss function
def Grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = Loss(model, inputs, targets)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)

# Set the optimizer for the learning process
def SetOptimizer(lr):
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    return optimizer 

# Find out the number of all gradients computed
def NumOfGrads(grads):
    count = 0
    for grad in grads:
        current_count = grad.shape[0]
        for dimension in grad.shape[1:]:
            current_count *= dimension
        count += current_count
    return count

# Transform the gradients based on Sto Algorithm
def StoTransformation(grads, b, hetero):
    transformed_grads = []
    for i, grad in enumerate(grads):
        random_tensor = tf.random.uniform(grad.shape, minval=0, maxval=1, dtype="float64")
        if hetero:
            new_grad = tf.math.divide(grad, 2*np.maximum(b[i],1e-20)) + 0.5
        else:
            new_grad = grad/tf.constant(2*np.maximum(b,1e-20), dtype="float64") + 0.5
        compare_tensor = tf.math.less(random_tensor, new_grad)
        transformed_grads.append(tf.sign(tf.dtypes.cast(compare_tensor, dtype="float64") - tf.constant(0.5, dtype="float64")))
    return transformed_grads

# Clip the gradients with the global L2 norm
def ClipGradsL2(grads, C):
    clipped_grads, global_norm = tf.clip_by_global_norm(grads, C)
    if global_norm > C:
        print(global_norm)
        print('global_norm larger than C')
    return clipped_grads

# Collect gradients from all nodes
def CollectGradsAdv(model, batch_size, datasets, C, delta, clipping):
    all_grads = []
    for node in range(len(datasets)):
        data_size = len(datasets[node][0])
        candidate_index = list(range(data_size))
        L = batch_size
        if data_size <= batch_size:
            sampled_index = candidate_index
            L = data_size
        else:
            sampled_index = np.random.choice(candidate_index, batch_size, replace=False)
        
        batched_grads = []
        new_grads = []
        if clipping:
            for index in sampled_index:
                image, label = np.asarray([datasets[node][0][index]]), np.asarray([datasets[node][1][index]])
                _, grads = Grad(model, image, label)
                if delta > 0:
                    grads = ClipGradsL2(grads, C)
                else:
                    raise ValueError("delta is {}, it must be positive!".format(delta))
                    
                if not batched_grads:
                    batched_grads = grads
                else:
                    for i, grad in enumerate(grads):
                        batched_grads[i] = tf.math.add(batched_grads[i], grad)
            new_grads = [grad/L for grad in batched_grads]
        else:
            images = np.asarray([datasets[node][0][index] for index in sampled_index])
            labels = np.asarray([datasets[node][1][index] for index in sampled_index])
            _, new_grads = Grad(model, images, labels)
        all_grads.append(new_grads)

    return all_grads

# Set the b value for each gradient element, which is based on the maximum of the absolute values of gradients computed on this entry over all nodes
def SetHeteroB(all_grads):
    max_grads = [tf.math.abs(grad) for grad in all_grads[0]]

    for grads in all_grads[1:]:
        for j in range(len(grads)):
            max_grads[j] = tf.math.maximum(max_grads[j], tf.math.abs(grads[j]))

    return max_grads

# Perform sto transformation for all gradients collected (including wrong gradients from the byzantine attackers) and combine them together. Error feedback included.
def CombinedStoGrads(all_grads, ori_grads_sign, b, num_of_byzantine, error_gradients, hetero):
    combined_grads = []

    for grads in all_grads:
        new_grads = StoTransformation(grads, b, hetero)
        if combined_grads:
            for j in range(len(combined_grads)):
                combined_grads[j] = tf.math.add(combined_grads[j], new_grads[j])
        else:
            combined_grads = new_grads
    
    for j in range(len(combined_grads)):
        combined_grads[j] = tf.math.add(combined_grads[j], tf.math.multiply(-num_of_byzantine, ori_grads_sign[j]))
        combined_grads[j] = tf.math.divide(combined_grads[j], num_of_byzantine+NODES)
        combined_grads[j] = tf.math.add(combined_grads[j], error_gradients[j])    
    
    sto_grads = [tf.sign(grad) for grad in combined_grads]
    error_grads = combined_grads.copy()
    for j in range(len(error_grads)):
        error_grads[j] = tf.math.add(error_grads[j], tf.math.multiply(-1/(num_of_byzantine+NODES),sto_grads[j]))
    
    return sto_grads,error_grads

# Combined all gradients without any transformation (summation)
def CombinedGrads(all_grads):
    combined_grads = []

    for grads in all_grads:
        if combined_grads:
            for j in range(len(combined_grads)):
                combined_grads[j] = tf.math.add(combined_grads[j], grads[j])
        else:
            combined_grads = grads
    
    return combined_grads

# Find the sign for the gradients computed
def SignGrads(grads):
    return [tf.sign(grad) for grad in grads]

# Same as the CombinedStoGrads function with additional computation of the fraction of sign-incorrectly computed gradients
def CombinedGradientsWithPEstimation(all_grads, b, num_of_byzantine, error_grads, hetero):
    ori_grads = CombinedGrads(all_grads)
    ori_grads_sign = SignGrads(ori_grads)
    sto_grads,error_grads = CombinedStoGrads(all_grads, ori_grads_sign, b, num_of_byzantine, error_grads, hetero)    
    num_of_grads = NumOfGrads(sto_grads)
    non_equal_grads = 0
    for i in range(len(sto_grads)):
        non_equal_grads += tf.math.reduce_sum(tf.dtypes.cast(tf.math.not_equal(sto_grads[i], ori_grads_sign[i]), dtype="float64"))

    return non_equal_grads/num_of_grads, sto_grads, error_grads


NODES = 31
LOSS_OBJECT = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
tf.keras.backend.set_floatx('float64')
np.random.seed(101)
random.seed(101)

if __name__ == '__main__':
    num_of_byzantine = 4
    print("The number of byzantine attackers is {}".format(num_of_byzantine))  
    num_labels_per_node = 1
    print("The number of distinct labels in each node's data is {}".format(num_labels_per_node))
    learning_rate = 0.001
    print("The learning rate is {}".format(learning_rate))
    batch_size = 80000
    b = 0.1
    print("b is {}".format(b))
    train_dataset_by_node, test_dataset_by_node = AssignDatasets(NODES,num_labels_per_node)    
    model = CreateModel((28, 28))
    optimizer = SetOptimizer(learning_rate)
    num_epoches = 101
    error_feedback = False
    print("The error feedback status is {}".format(error_feedback))
    _, error_grads = Grad(model, np.asarray(train_dataset_by_node[0][0]), np.asarray(train_dataset_by_node[0][1]))
    P11 = []
    for j in range(len(error_grads)):
        error_grads[j] = tf.math.add(error_grads[j], -error_grads[j])
                        
    for epoch in range(num_epoches):
        epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

        all_grads = CollectGradsAdv(model, batch_size, train_dataset_by_node, C=1000, delta=0, clipping=False)
        b = SetHeteroB(all_grads)
        hetero = True
        p_z, sto_grads,error_grads = CombinedGradientsWithPEstimation(all_grads, b, num_of_byzantine, error_grads, hetero)
        if error_feedback == False:
            for j in range(len(error_grads)):
                error_grads[j] = tf.math.add(error_grads[j], -error_grads[j])
        
        
        optimizer.apply_gradients(zip(sto_grads, model.trainable_variables))
        
        accuracy = 0
        for node in range(NODES):
            accuracy += float(epoch_accuracy(np.asarray(train_dataset_by_node[node][1]), model(np.asarray(train_dataset_by_node[node][0]))))
        accuracy /= NODES
        
        if epoch % 10 == 0:
            print("P11 is {:.3%}".format(p_z))
            print("Epoch {:03d}: Accuracy: {:.3%}".format(epoch, accuracy))
            P11.append(format(p_z))
    test_accuracy = 0
    for node in range(NODES):
        test_accuracy += float(epoch_accuracy(np.asarray(test_dataset_by_node[node][1]), model(np.asarray(test_dataset_by_node[node][0]))))
    test_accuracy /= NODES
    print(P11)
    print("Test accuracy is: {:.3%}".format(test_accuracy))
