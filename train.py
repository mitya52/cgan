import tensorflow as tf
import numpy as np
from time import time

from generator import Generator
from cnn import Model


def compute_accuracy(labels, predictions):
    labels = np.argmax(labels, axis=-1)
    predictions = np.argmax(predictions, axis=-1)
    return float(np.count_nonzero(labels == predictions)) / len(labels)


def train(batch_size: int,
          epochs: int):
    train_generator = Generator(batch_size=batch_size, train=True)
    val_generator = Generator(batch_size=batch_size, train=False)

    assert (train_generator.output_shape() == val_generator.output_shape())
    assert (train_generator.num_classes() == val_generator.num_classes())

    image_shape = train_generator.output_shape()
    num_classes = train_generator.num_classes()

    images_input = tf.placeholder(tf.float32, (None,) + image_shape)
    labels_output = tf.placeholder(tf.float32, (None, num_classes))
    is_training = tf.placeholder(tf.bool)

    model = Model(images_input, labels_output, is_training, num_classes)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = tf.contrib.layers.optimize_loss(
            model.loss,
            tf.train.get_global_step(),
            optimizer='Adam',
            learning_rate=0.0001,
            summaries=['loss'])

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        best_val_loss = 1e15
        best_val_acc = 0
        best_epoch = 0
        for epoch in range(epochs):
            # training phase
            losses = []
            t0 = time()
            for images, labels in train_generator.generate_epoch():
                loss = sess.run([train_op], feed_dict={
                    images_input: images,
                    labels_output: labels,
                    is_training: True,
                })
                losses.append(loss)
            t1 = time()
            print('Epoch {} train loss: {}'.format(epoch, np.mean(losses)))
            print('Elapsed time: {:.2f}'.format(t1 - t0))

            losses = []
            accuracies = []
            for images, labels in val_generator.generate_epoch():
                predictions, loss = sess.run([model.predictions, model.loss], feed_dict={
                    images_input: images,
                    labels_output: labels,
                    is_training: False,
                })
                accuracies.append(compute_accuracy(labels, predictions))
                losses.append(loss)
            val_loss = np.mean(losses)
            val_acc = np.mean(accuracies)

            best_val_loss = min(val_loss, best_val_loss)
            best_val_acc = val_acc
            best_epoch = epoch

            print('Val loss: {}'.format(val_loss))
            print('Val accuracy: {}'.format(val_acc))
            print()

        print('Best val loss: {} (acc {}, epoch {})'.format(best_val_loss, best_val_acc, best_epoch))


if __name__ == '__main__':
    batch_size = 64
    epochs = 100

    train(batch_size, epochs)