#!/gpfs_common/share03/hdai/richeng/Sim1/venv/bin/python3.6
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


# Split the whole dataset to datasets for all nodes in a way that two nodes can share some data and all nodes' datasets have the same number of distinct labels
def SameLabelSplitDataOverlap(nodes, images_by_label, labels_by_node, images_per_node=2000, overlap=0.5):
    dataset_by_node = defaultdict(list)
    current_ids = [0]*10
    candidate_pool = set(range(10))
    prev_candidates = set([])
    images_per_label = images_per_node // labels_by_node
    for node in range(nodes):
        if len(candidate_pool) <= labels_by_node:
            first_candidates = list(candidate_pool)
            remain_num_candidates = labels_by_node - len(candidate_pool)
            second_candidates = list(np.random.choice(list(prev_candidates), remain_num_candidates, replace=False))
            final_candidates = first_candidates + second_candidates
            candidate_pool = prev_candidates.difference(set(second_candidates))
            prev_candidates = set(final_candidates)
        else:
            final_candidates = list(np.random.choice(list(candidate_pool), labels_by_node, replace=False))
            candidate_pool = candidate_pool.difference(set(final_candidates))
            candidate_pool = candidate_pool.union(prev_candidates)
            prev_candidates = set(final_candidates)
        for label in final_candidates:
            current_id = current_ids[label]
            new_id = current_id + images_per_label
            if new_id > len(images_by_label[label]):
                end_id = new_id%len(images_by_label[label])
                images = images_by_label[label][current_id : new_id] + images_by_label[label][:end_id] 
                current_ids[label] = (current_id + int(overlap * images_per_label))%len(images_by_label[label])
            else:
                images = images_by_label[label][current_id : new_id]
                current_ids[label] = (current_id + int(overlap * images_per_label))
            labels = [label]*len(images)
            if len(dataset_by_node[node]) == 0:
                dataset_by_node[node].append(images)
                dataset_by_node[node].append(labels)
            else:
                dataset_by_node[node][0] += images
                dataset_by_node[node][1] += labels
    return dataset_by_node

# Split the whole dataset to datasets for all nodes in a way that any two nodes do not share any data sample and all nodes' datasets have the same number of distinct labels
def SameLabelSplitData(nodes, images_by_label, labels_by_node, number_of_imgs_by_node, same_num_images_per_node=False):
    dataset_by_node = defaultdict(list)
    segments = math.ceil(nodes*labels_by_node/10)
    num_of_images_by_label = [len(images_by_label[label]) for label in range(10)]
    if same_num_images_per_node:
        min_images_of_all_labels = min(num_of_images_by_label)
        num_imgs_per_segment = min(int(min_images_of_all_labels/segments), number_of_imgs_by_node // labels_by_node)
    else:
        num_imgs_per_segment = [int(num_of_images/segments) for num_of_images in num_of_images_by_label]
    segment_ids = []
    heapq.heapify(segment_ids)
    candidate_labels = list(range(10))
    np.random.shuffle(candidate_labels)
    for label in range(10):
        heapq.heappush(segment_ids, (0, label))
    for node in range(nodes):
        used_labels = []
        for _ in range(labels_by_node):
            current_id, label = heapq.heappop(segment_ids)
            actual_label = candidate_labels[label]
            if same_num_images_per_node:
                images_per_seg_per_label = num_imgs_per_segment
            else:
                images_per_seg_per_label = num_imgs_per_segment[actual_label]
            images = images_by_label[actual_label][current_id*images_per_seg_per_label
                                                   :(current_id+1)*images_per_seg_per_label]
            labels = [actual_label]*len(images)
            if len(dataset_by_node[node]) == 0:
                dataset_by_node[node].append(images)
                dataset_by_node[node].append(labels)
            else:
                dataset_by_node[node][0] += images
                dataset_by_node[node][1] += labels
            used_labels.append((current_id+1, label))
        for used_label in used_labels:
            heapq.heappush(segment_ids, used_label)
    return dataset_by_node

# Assign datasets for all nodes from the MNIST dataset
def AssignDatasets(nodes, min_labels = 1, number_of_imgs_by_node = 2000, have_same_label_number=False, same_num_images_per_node=False, sample_overlap_data=False):
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

    for label in range(10):
        np.random.shuffle(train_images_by_label[label])
        np.random.shuffle(test_images_by_label[label])
    
    train_dataset_by_node = defaultdict(list)
    test_dataset_by_node =defaultdict(list)

    if min_labels > 10:
        raise ValueError("Minimum number of labels is {}, which exceeds the total number of labels!".format(min_labels))
    if have_same_label_number:
        if sample_overlap_data:
            train_dataset_by_node = SameLabelSplitDataOverlap(nodes, train_images_by_label, min_labels)
            test_dataset_by_node = SameLabelSplitDataOverlap(nodes, test_images_by_label, min_labels)
        else:
            train_dataset_by_node = SameLabelSplitData(nodes, train_images_by_label, min_labels, number_of_imgs_by_node=number_of_imgs_by_node, same_num_images_per_node=same_num_images_per_node)
            test_dataset_by_node = SameLabelSplitData(nodes, test_images_by_label, min_labels, number_of_imgs_by_node=number_of_imgs_by_node, same_num_images_per_node=same_num_images_per_node)
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
            new_grad = tf.math.divide(grad, 2*np.maximum(b[i],1e-6)) + 0.5
        else:
            new_grad = grad/tf.constant(2*np.maximum(b,1e-6), dtype="float64") + 0.5
        compare_tensor = tf.math.less(random_tensor, new_grad)
        transformed_grads.append(tf.sign(tf.dtypes.cast(compare_tensor, dtype="float64") - tf.constant(0.5, dtype="float64")))
    return transformed_grads

# Add Gaussian noises to the gradients
def AddGaussianNoise(grads, mean, std):
    grads_with_noise = []
    for grad in grads:
        noise = tf.random.normal(grad.shape, mean, std)
        grad_with_noise = grad + noise
        grads_with_noise.append(grad_with_noise)
    
    return grads_with_noise

# Perform the dp sign transformation over the gradients
def DPSignTransformationGaussian(grads, delta, sigma):
    transformed_grads = []
    for grad in grads:
        random_tensor = tf.random.uniform(grad.shape, minval=0, maxval=1, dtype="float64")
        new_grad = 0.5 + 0.5*tf.math.erf(grad/(sigma*(2**0.5)))
        compare_tensor = tf.math.less(random_tensor, new_grad)
        transformed_grads.append(tf.sign(tf.dtypes.cast(compare_tensor, dtype="float64") - tf.constant(0.5, dtype="float64")))
    return transformed_grads

# Perform the dp sign Laplacian transformation over the gradients
def DPSignTransformationLaplacian(grads, delta, lambda_v):
    transformed_grads = []
    for grad in grads:
        random_tensor = tf.random.uniform(grad.shape, minval=0, maxval=1, dtype="float64")
        new_grad = 0.5 + 0.5*tf.math.multiply(tf.sign(grad), (1-tf.math.exp(-1.0*tf.math.divide(tf.math.abs(grad), lambda_v))))
        compare_tensor = tf.math.less(random_tensor, new_grad)
        transformed_grads.append(tf.sign(tf.dtypes.cast(compare_tensor, dtype="float64") - tf.constant(0.5, dtype="float64")))
    return transformed_grads

# Set the epsilon value
def SetEpsilon(tdelta_1, tdelta_2, delta, sigma, lambda_v):
    if delta == 0:
        return tdelta_1/lambda_v
    elif delta > 0:
        return tdelta_2/sigma*math.sqrt(2*math.log(1.25/delta))
    else:
        raise ValueError("delta is {}, it cannot be negative!".format(delta))

# Set the sigma value
def SetSigma(epsilon, delta, C):
    sigma = math.sqrt(2*math.log(1.25/delta))*C/epsilon
    return sigma

# Set the lambda value
def SetLambda(epsilon, C):
    lambda_v = C/epsilon
    
    return lambda_v

# Collect original gradients from all nodes
def CollectGrads(model, datasets):
   all_grads = []

   for node in range(NODES):
       _, grads = Grad(model, datasets[node][0], datasets[node][1])
       all_grads.append(grads)

   return all_grads

# Clip the gradients with the global L2 norm
def ClipGradsL2(grads, C):
    clipped_grads, global_norm = tf.clip_by_global_norm(grads, C)
    if global_norm > C:
        print(global_norm)
        print('global_norm larger than C')
    return clipped_grads

# Compute the gradients for each data sample in a batch with clipping included
def BatchedGrads(args):
   inputs, targets = args
   with tf.GradientTape() as tape:
       inputs = tf.expand_dims(inputs, 0)
       targets = tf.one_hot(targets, 10, dtype='float64')
       predictions = MODEL(inputs)
       loss = tf.keras.losses.categorical_crossentropy(y_true=targets, y_pred=predictions)

   grads = tape.gradient(loss, MODEL.trainable_variables)
   if CLIPPING:
       if DELTA > 0:
           return [grad / tf.maximum(tf.constant(1, dtype='float64'), tf.norm(grad)/C) for grad in grads]
       elif DELTA == 0:
           return [grad / tf.maximum(tf.constant(1, dtype='float64'), tf.norm(grad, ord=1)/C) for grad in grads]
       else:
           raise ValueError("Delta is {}, it cannot be negative!".format(DELTA))
   return grads

# Parallelly compute the gradients for a batch of samples with per-example clipping included.
def CollectGradsVec(batch_size, datasets):
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

# Compute the gradients for a batch of samples in a tensor way (inspired by the GoodFellow's method) for all nodes with per-example clipping included.
def CollectGradsGoodFellow(model, batch_size, datasets):
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
       models = [model]*batch_size
       with tf.GradientTape() as tape:
           predictions = tf.stack([model(tf.reshape(image, [1, 60])) for model, image in zip(models, batched_images)])
           targets = tf.one_hot(batched_labels, 10)
           losses = tf.keras.losses.categorical_crossentropy(y_true=targets, y_pred=predictions)
           batched_variables = [model.trainable_variables] * batch_size
       batched_grads = tape.gradient(losses, batched_variables)
       if CLIPPING:
           if DELTA > 0:
               batched_grads = [[grad / tf.maximum(tf.constant(1, dtype='float64'), tf.norm(grad)/C) for grad in grads] for grads in batched_grads]
           elif DELTA == 0:
               batched_grads = [[grad / tf.maximum(tf.constant(1, dtype='float64'), tf.norm(grad, ord=1)/C) for grad in grads] for grads in batched_grads]
           else:
               raise ValueError("Delta is {}, it cannot be negative!".format(DELTA))
       sum_grads = CombinedGrads(batched_grads)
       all_grads.append(sum_grads)
   return all_grads

# Collect gradients from all nodes with clipping function included.
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
                    raise ValueError("delta is {}, it cannot be negative!".format(delta))
                    
                if not batched_grads:
                    batched_grads = grads
                else:
                    for i, grad in enumerate(grads):
                        batched_grads[i] = tf.math.add(batched_grads[i], grad)
            new_grads = [grad for grad in batched_grads]
        else:
            images = np.asarray([datasets[node][0][index] for index in sampled_index])
            labels = np.asarray([datasets[node][1][index] for index in sampled_index])
            _, grads = Grad(model, images, labels)
            new_grads = [grad*L for grad in grads]
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
def CombinedStoGrads(all_grads, ori_grads_sign, epsilon, delta, num_of_byzantine, error_gradients, C):
    combined_grads = []

    for grads in all_grads:
        if delta > 0: 
            sigma = SetSigma(epsilon, delta, C)
            new_grads = DPSignTransformationGaussian(grads, delta, sigma)
        elif delta == 0:
            lambda_v = SetLambda(epsilon,C)
            new_grads = DPSignTransformationLaplacian(grads, delta, lambda_v)
        else:
            raise ValueError("delta is {}, it cannot be negative!".format(delta))
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
def CombinedGradientsWithPEstimation(all_grads, epsilon, delta, num_of_byzantine, error_grads, C):         
    ori_grads = CombinedGrads(all_grads)
    ori_grads_sign = SignGrads(ori_grads)
    sto_grads,error_grads = CombinedStoGrads(all_grads, ori_grads_sign, epsilon, delta, num_of_byzantine, error_grads, C)    
    num_of_grads = NumOfGrads(sto_grads)
    non_equal_grads = 0
    for i in range(len(sto_grads)):
        non_equal_grads += tf.math.reduce_sum(tf.dtypes.cast(tf.math.not_equal(sto_grads[i], ori_grads_sign[i]), dtype="float64"))

    return non_equal_grads/num_of_grads, sto_grads, error_grads


LOSS_OBJECT = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
MODEL = CreateModel((28, 28))
NODES = 31
DELTA = 0.00001
CLIPPING = True 
tf.keras.backend.set_floatx('float64')
np.random.seed(101)
random.seed(101)

if __name__ == '__main__':
    num_of_byzantine = 4    
    epsilon = 0.2
    learning_rate = 0.01
    num_labels_per_node = 1
    C = 4
    print("The number of byzantine attackers is {}".format(num_of_byzantine))
    print("The number of distinct labels in each node's data is {}".format(num_labels_per_node))
    print("The learning rate is {}".format(learning_rate))
    print("The clipping norm is {}".format(C))
    print("Epsilon is {}".format(epsilon))
    batch_size = 80000
    train_dataset_by_node, test_dataset_by_node = AssignDatasets(NODES, min_labels = num_labels_per_node,
                                                                 number_of_imgs_by_node = 2000, have_same_label_number=True,
                                                                 same_num_images_per_node=False, sample_overlap_data=False)
    optimizer = SetOptimizer(learning_rate)                  
    num_epoches = 51
    error_feedback = True
    _, error_grads = Grad(MODEL, np.asarray(train_dataset_by_node[0][0]), np.asarray(train_dataset_by_node[0][1]))
    for j in range(len(error_grads)):
        error_grads[j] = tf.math.add(error_grads[j], -error_grads[j])
                    
    for epoch in range(num_epoches):
        epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()        
        all_grads = CollectGradsVec(batch_size, train_dataset_by_node)
        p_z, sto_grads, error_grads = CombinedGradientsWithPEstimation(all_grads, epsilon, DELTA, num_of_byzantine,error_grads,C)
        if error_feedback == False:
            for j in range(len(error_grads)):
                error_grads[j] = tf.math.add(error_grads[j], -error_grads[j])
        
        
        optimizer.apply_gradients(zip(sto_grads, MODEL.trainable_variables))
        
        accuracy = 0
        for node in range(NODES):
            accuracy += float(epoch_accuracy(np.asarray(train_dataset_by_node[node][1]), MODEL(np.asarray(train_dataset_by_node[node][0]))))
        accuracy /= NODES
        
        if epoch % 10 == 0:
            print("Epoch {:03d}: Accuracy: {:.3%}".format(epoch, accuracy))
    test_accuracy = 0
    for node in range(NODES):
        test_accuracy += float(epoch_accuracy(np.asarray(test_dataset_by_node[node][1]), MODEL(np.asarray(test_dataset_by_node[node][0]))))
    test_accuracy /= NODES
    print("Test accuracy is: {:.3%}".format(test_accuracy))