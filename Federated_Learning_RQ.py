import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops

from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import random

strategy = tf.distribute.Strategy()
BATCH_SIZE_PER_REPLICA = 64
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
EPOCHS = 10
B = 100

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images/255.0
test_images = test_images/255.0

train_images_by_class = defaultdict(list)
test_images_by_class = defaultdict(list)

for index, image in enumerate(train_images):
    train_images_by_class[train_labels[index]].append(image)

for index, image in enumerate(test_images):
    test_images_by_class[test_labels[index]].append(image)


def create_model():
    '''
    Baseline training model for each worker.
    '''
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    return model

with strategy.scope():
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        reduction=tf.keras.losses.Reduction.NONE)
    
    def compute_loss(labels, predictions):
        per_example_loss = loss_object(labels, predictions)
        return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)

with strategy.scope():
    test_loss = tf.keras.metrics.Mean(name='test_loss')

    train_accuracy = tf.keras.metrics.SparseCategoricalCrossentropy(name='train_accuracy')
    test_accuracy = tf.keras.metrics.SparseCategoricalCrossentropy(name='test_accuracy')

with strategy.scope():
    model = create_model()
    
    optimizer = tf.keras.optimizer.Adam()

with strategy.scope():
    def train_step_sto_signSGD(inputs):
        images, labels = inputs

        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = compute_loss(labels, predictions)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        gradients_list = list(gradients)
        gradient_values = [
            ops.convert_to_sensor(
                g.values if isinstance(g, ops.IndexedSlices) else g,
                name="g_%d" %i)
            if g is not None else g
            for i, g in enumerate(gradients_list)]
        new_gradients_list = []
        for i, g in enumerate(gradient_values):
            if g is None:
                new_gradients_list.append(None)
            else:
                new_value = 1 if random.random() < ((B+g)/2/B) else -1
                with ops.colocate_with(g):
                    new_gradients_list.append(array_ops.identity(new_value, name="g_%d" % i))
        
        new_gradients = [
            ops.IndexedSlices(new_g, g.indices, g.dense_shape)
            if isinstance(g, ops.IndexedSlices)
            else new_g
            for (new_g, g) in zip(new_gradients_list, gradients_list)]
        
        optimizer.apply_gradients(zip(new_gradients, model.trainable_variables))

        train_accuracy.update_state(labels, predictions)
        return loss
    
    def test_step(inputs):
        images, labels = inputs
        
        predictions = model(images, training=False)
        t_loss = loss_object(labels, predictions)

        test_loss.update_state(t_loss)
        test_accuracy.update_state(labels, predictions)

with strategy.scope():
  @tf.function
  def distributed_train_step(dataset_inputs):
    per_replica_losses = strategy.experimental_run_v2(train_step_sto_signSGD,
                                                      args=(dataset_inputs,))
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                           axis=None)
 
  @tf.function
  def distributed_test_step(dataset_inputs):
    return strategy.experimental_run_v2(test_step, args=(dataset_inputs,))

  for epoch in range(EPOCHS):
    # TRAIN LOOP
    total_loss = 0.0
    num_batches = 0
    for c, images in train_images_by_class:
      total_loss += distributed_train_step((images, [c]*len(images)))
      num_batches += 1
    train_loss = total_loss / num_batches

    # TEST LOOP
    for c, images in test_images_by_class:
      distributed_test_step((images, [c]*len(images)))


    template = ("Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, "
                "Test Accuracy: {}")
    print (template.format(epoch+1, train_loss,
                           train_accuracy.result()*100, test_loss.result(),
                           test_accuracy.result()*100))

    test_loss.reset_states()
    train_accuracy.reset_states()
    test_accuracy.reset_states()