import tensorflow as tf
from tensorflow import keras

from collections import defaultdict
import numpy as np
import pandas as pd
import math
import random
import heapq
import math
import time
from sklearn.decomposition import PCA

NODES = 31
LOSS_OBJECT = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

def load_local_mnist(path):
    path = tf.keras.utils.get_file(
        'mnist.npz',
        origin = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz',
        file_hash = '731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1',
        cache_dir = path
    )

    with np.load(path, allow_pickle=True) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']

        return (x_train, y_train), (x_test, y_test)

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

def ArrayDataset(dataset):
    new_dataset = defaultdict(list)
    for i in range(len(dataset)):
        new_dataset[i].append(np.asarray(dataset[i][0]))
        new_dataset[i].append(np.asarray(dataset[i][1]))

    return new_dataset

def CreateModel(data_shape):
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=data_shape),
        keras.layers.Dense(512, activation='relu', dtype='float64'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(512, activation='relu', dtype='float64'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation='softmax', dtype='float64')
    ])
    return model

def Loss(model, x, y):
    y_ = model(x)
    return LOSS_OBJECT(y_true=y, y_pred=y_)

def SetOptimizer(lr):
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    return optimizer 

def Grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = Loss(model, inputs, targets)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)

def NumOfGrads(grads):
    count = 0
    for grad in grads:
        current_count = grad.shape[0]
        for dimension in grad.shape[1:]:
            current_count *= dimension
        count += current_count
    return count

def StoTransformation(grads, b, hetero=True):
    transformed_grads = []
    for i, grad in enumerate(grads):
        random_tensor = tf.random.uniform(grad.shape, minval=0, maxval=1, dtype="float64")
        if hetero:
            new_grad = tf.math.divide(grad, 2*b[i]) + 0.5
        else:
            new_grad = grad/tf.constant(2*b, dtype="float64") + 0.5
        compare_tensor = tf.math.less(random_tensor, new_grad)
        transformed_grads.append(tf.sign(tf.dtypes.cast(compare_tensor, dtype="float64") - tf.constant(0.5, dtype="float64")))
    return transformed_grads

def AddGaussianNoise(grads, mean, std):
    grads_with_noise = []
    for grad in grads:
        noise = tf.random.normal(grad.shape, mean, std)
        grad_with_noise = grad + noise
        grads_with_noise.append(grad_with_noise)
    
    return grads_with_noise

def DPSignTransformation(grads, delta, sigma, lambda_v):
    transformed_grads = []
    for grad in grads:
        random_tensor = tf.random.uniform(grad.shape, minval=0, maxval=1, dtype="float64")
        if delta > 0:
            new_grad = 0.5 + 0.5*tf.math.erf(grad/sigma/(2**0.5))
        elif delta == 0:
            new_grad = 0.5 + 0.5*tf.math.multiply(tf.sign(grad), (1-tf.math.exp(-1.0*tf.math.divide(tf.math.abs(grad), lambda_v))))
        else:
            raise ValueError("delta is {}, it cannot be negative!".format(delta))
        compare_tensor = tf.math.less(random_tensor, new_grad)
        transformed_grads.append(tf.sign(tf.dtypes.cast(compare_tensor, dtype="float64") - tf.constant(0.5, dtype="float64")))
    return transformed_grads

def SetEpsilon(tdelta_1, tdelta_2, delta, sigma, lambda_v):
    if delta == 0:
        return tdelta_1/lambda_v
    elif delta > 0:
        return tdelta_2/sigma*math.sqrt(2*math.log(1.25/delta))
    else:
        raise ValueError("delta is {}, it cannot be negative!".format(delta))

def ZeroTransformation(grads, min_grad):
    transformed_grads = []
    for grad in grads:
        random_tensor = tf.random.uniform(grad.shape, minval=0, maxval=1, dtype="float64")
        new_grad = tf.constant(0.5, "float64")
        compare_tensor = tf.math.less(random_tensor, new_grad)
        sign_grad = tf.sign(tf.dtypes.cast(compare_tensor, dtype="float64") - tf.constant(0.5, dtype="float64"))
        mask_grad = tf.sign(tf.dtypes.cast(tf.math.equal(grad, 0), dtype="float64"))
        transformed_grads.append(grad + tf.math.multiply(sign_grad, mask_grad)*min_grad)
    return transformed_grads

def CollectGrads(model, batch_id, datasets):
    all_grads = []

    for node in range(len(datasets)):
        _, grads = Grad(model, datasets[node][0][batch_id], datasets[node][1][batch_id])
        all_grads.append(grads)

    return all_grads

def ClipGradsL2(grads, C):
    clipped_grads, _ = tf.clip_by_global_norm(grads, C)
    return clipped_grads

def ClipGradsL1(grads, C):
    norm_l1 = tf.add_n([tf.norm(grad, ord=1) for grad in grads])
    clipped_grads,_ = tf.clip_by_global_norm(grads, C, use_norm=norm_l1)  
    return clipped_grads

def BatchedGrads(args):
    inputs, targets = args
    with tf.GradientTape() as tape:
        inputs = tf.expand_dims(inputs, 0)
        targets = tf.one_hot(targets, 10, dtype='float64')
        predictions = MODEL(inputs)
        loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true=targets, y_pred=predictions, from_logits=True))

    grads = tape.gradient(loss, MODEL.trainable_variables)
    if CLIPPING:
        if DELTA > 0:
            global_norm = tf.math.sqrt(tf.add_n([tf.square(tf.norm(grad)) for grad in grads]))
        elif DELTA == 0:
            global_norm = tf.add_n([tf.norm(grad, ord=1) for grad in grads])
        else:
            raise ValueError("Delta is {}, it cannot be negative!".format(DELTA))
        return [grad / tf.maximum(tf.constant(1, dtype='float64'), global_norm/C) for grad in grads]
    return grads

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
            losses = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true=targets, y_pred=predictions, from_logits=True))
            batched_variables = [model.trainable_variables] * batch_size
        batched_grads = tape.gradient(losses, batched_variables)
        if CLIPPING:
            if DELTA > 0:
                batched_grads = [ClipGradsL2(grads, C) for grads in batched_grads]
            elif DELTA == 0:
                batched_grads = [ClipGradsL1(grads, C) for grads in batched_grads]
            else:
                raise ValueError("Delta is {}, it cannot be negative!".format(DELTA))
        sum_grads = CombinedGrads(batched_grads)
        all_grads.append(sum_grads)
    return all_grads

def CollectGradsAdv(model, batch_size, datasets, C, clipping=False):
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
                image, label = np.asarray(datasets[node][0][index]), np.asarray(datasets[node][1][index])
                _, grads = Grad(model, image, label)
                grads = ClipGradsL2(grads, C)
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

def SetB(all_grads):
    max_grad = 0
    for grads in all_grads:
        current_max_grad = max([tf.math.reduce_max(tf.math.abs(grad)) for grad in grads])
        max_grad = max(max_grad, current_max_grad)
    return max_grad + 1e-6

def SetHeteroB(all_grads, offset=1e-6):
    max_grads = [tf.math.abs(grad) for grad in all_grads[0]]

    for grads in all_grads[1:]:
        for j in range(len(grads)):
            max_grads[j] = tf.math.maximum(max_grads[j], tf.math.abs(grads[j]))

    return max_grads + offset

def FindMinGrad(all_grads):
    min_grad = float('inf')
    for grads in all_grads:
        current_min_grad = max([tf.math.reduce_min(tf.math.abs(grad)) for grad in grads])
        min_grad = min(min_grad, current_min_grad)
    return min_grad/2.0

def CombinedStoGrads(all_grads, b):
    combined_grads = []

    for grads in all_grads:
        new_grads = StoTransformation(grads, b)
        if combined_grads:
            for j in range(len(combined_grads)):
                combined_grads[j] = tf.math.add(combined_grads[j], new_grads[j])
        else:
            combined_grads = new_grads
    
    return [tf.sign(grad) for grad in combined_grads]

def CombinedGaussianGrad(all_grads, mean, std):
    combined_grads = []

    for grads in all_grads:
        new_grads = AddGaussianNoise(grads, mean, std)
        if combined_grads:
            for j in range(len(combined_grads)):
                combined_grads[j] = tf.math.add(combined_grads[j], new_grads[j])
        else:
            combined_grads = new_grads
    
    return combined_grads

def CombinedGrads(all_grads, avoid_zeros=False):
    combined_grads = []
    min_grad = FindMinGrad(all_grads)
    for grads in all_grads:
        if avoid_zeros:
            grads = ZeroTransformation(grads, min_grad)
        if combined_grads:
            for j in range(len(combined_grads)):
                combined_grads[j] = tf.math.add(combined_grads[j], grads[j])
        else:
            combined_grads = grads
    
    return combined_grads

def SignGrads(grads):
    return [tf.sign(grad) for grad in grads]

def CombinedGradientsWithPEstimation(all_grads, b, filter_zeros=False):
    
    sto_grads = CombinedStoGrads(all_grads, b)
    ori_grads = CombinedGrads(all_grads)
    ori_grads = SignGrads(ori_grads)
    num_of_grads = NumOfGrads(sto_grads)
    zero_grads = 0
    non_equal_grads = 0
    for i in range(len(sto_grads)):
        non_equal_grads += tf.math.reduce_sum(tf.dtypes.cast(tf.math.not_equal(sto_grads[i], ori_grads[i]), dtype="float64"))
        zero_grads += tf.math.reduce_sum(tf.dtypes.cast(tf.math.equal(ori_grads[i], 0), dtypes="float64"))
    
    p_z = 0
    if filter_zeros:
        if num_of_grads-zero_grads != 0:
            p_z = (non_equal_grads-zero_grads)/(num_of_grads-zero_grads)
    else:
        p_z = non_equal_grads/num_of_grads

    return p_z, sto_grads

def E18Calculation(grads, b):
    tf_const_M = tf.constant(NODES, dtype="float64")
    tf_const_B = tf.constant(1/b, dtype="float64")
    transformed_grads = []
    for grad in grads:
        numerator_grad = tf.math.exp(-0.5*tf_const_B*grad)
        denominator_grad = tf.math.pow(tf.math.divide(tf_const_M, (tf_const_M+tf_const_B*grad)), tf_const_M/2)
        transformed_grads.append(tf.math.divide(numerator_grad, denominator_grad))
    
    return transformed_grads

def RealE18Calculation(sum_grads, b):
    M = NODES
    return np.exp(-sum_grads/(2*b))/pow((M/(M+sum_grads/b)), M/2)

def StoSign(arr, b):
    arr = np.asarray
    random_arr = np.random.rand(arr.shape[0])
    arr = arr/(2*b) + 0.5
    comp_arr = np.less(random_arr, arr)
    return np.sign(sum([1 if v else -1 for v in comp_arr]))

def EnableGPU():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
    # Restrict TensorFlow to only use the first GPU
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)
