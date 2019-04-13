import tensorflow as tf

import numpy as np


class Model:
    def __init__(self,
                 images_input: tf.placeholder,
                 labels_output: tf.placeholder,
                 is_training: tf.placeholder,
                 classes: int):
        x = self._conv(images_input, is_training, filters=16)
        x = self._conv(x, is_training, filters=32, pooling=True)
        x = self._conv(x, is_training, filters=32)
        x = self._conv(x, is_training, filters=64, pooling=True)
        x = self._conv(x, is_training, filters=64)

        x = tf.reduce_mean(x, axis=[1, 2], keep_dims=False)

        x = tf.layers.dense(inputs=x, units=128, activation='relu')
        x = tf.layers.dense(inputs=x, units=128, activation='relu')

        logits = tf.layers.dense(inputs=x, units=classes, activation='relu')

        self.loss = tf.losses.softmax_cross_entropy(labels_output, logits)
        self.predictions = tf.nn.softmax(logits, axis=-1)

    @staticmethod
    def _conv(inputs: tf.Tensor,
              is_training: tf.Tensor,
              filters: int,
              batch_norm: bool = True,
              pooling: bool = False):
        x = tf.layers.conv2d(inputs=inputs,
                             filters=filters,
                             kernel_size=3,
                             padding='same',
                             activation='relu',
                             kernel_initializer='he_normal')
        if batch_norm:
            x = tf.layers.batch_normalization(inputs=x,
                                              training=is_training)
        if pooling:
            x = tf.layers.max_pooling2d(inputs=x,
                                        strides=2,
                                        pool_size=2)
        return x


if __name__ == '__main__':
    image_shape = 32, 32, 3
    classes = 10
    batch_size = 32

    images_input = tf.placeholder(tf.float32, (None,) + image_shape)
    labels_output = tf.placeholder(tf.float32, (None, classes))
    is_training = tf.placeholder(tf.bool)

    model = Model(images_input, labels_output, is_training, classes)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = tf.contrib.layers.optimize_loss(
            model.loss,
            tf.train.get_global_step(),
            optimizer='Adam',
            learning_rate=0.001,
            summaries=['loss'])

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        images = np.zeros((batch_size,) + image_shape, dtype=np.float32)
        labels = np.ones((batch_size, classes), dtype=np.float32)

        # training phase
        predictions, loss = sess.run([model.predictions, train_op], feed_dict={
            images_input: images,
            labels_output: labels,
            is_training: True,
        })

        print(predictions.shape)
        print(loss)

        # validation phase
        predictions = sess.run(model.predictions, feed_dict={
            images_input: images,
            is_training: False,
        })

        print(predictions.shape)
