"""
This file includes all utility funtions needed to perform the simulation on federated learning.
DO NOT write your own main simulation function in this file, IMPORT this file to your simulation function.
If you want to add new functions to this file, please FOLLOW THE SAME FORMAT as the existing functions.
"""
import tensorflow as tf
from tensorflow import keras

from collections import defaultdict
import numpy as np
import math
import heapq


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
        assert (len(set(chosen_labels)) == labels_per_node) (
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
            elif current_unassigned_img_id >= current_unassigned_img_id:
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
    # the smallest unassigned segment id (least assigned label).
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

            if len(dataset_by_node) == 0:
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
    """Assign the training and testing datasets to all nodes.

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
        train_dataset_by_node = SameLabelSplitDataEven(nodes, train_images_by_label,
                                                       labels_per_node, num_train_imgs_per_node,
                                                       total_train_images_with_reserve_by_label)
        test_dataset_by_node = SameLabelSplitDataEven(nodes, test_images_by_label,
                                                      labels_per_node, num_test_imgs_per_node,
                                                      total_test_images_with_reserve_by_label)
    
    return train_dataset_by_node, test_dataset_by_node     