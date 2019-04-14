import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np

from typing import Optional, Union, Tuple


class Model:
    def __init__(self,
                 image_shape: Tuple[int, int, int],
                 num_classes: int,
                 latent_dim: int,
                 dropout_rate: float = 0.3):
        self._image_shape = image_shape
        self._num_classes = num_classes
        self._dropout_rate = dropout_rate

        assert (self._image_shape[0] % (2 ** 2) == 0)
        assert (self._image_shape[1] % (2 ** 2) == 0)

        self.images_input = tf.placeholder(tf.float32, shape=(None,) + self._image_shape)
        self.labels_input = tf.placeholder(tf.float32, shape=(None, self._num_classes))
        self.z_input = tf.placeholder(tf.float32, shape=(None, latent_dim))
        self.is_training = tf.placeholder(tf.bool)

        self.generated = self._create_generator(self.z_input, self.labels_input)

        d_z = self._create_discriminator(self.generated, self.labels_input)
        d_image = self._create_discriminator(self.images_input, self.labels_input)

        eps = 1e-15
        self.g_loss = -tf.reduce_mean(tf.log(d_z + eps))
        self.d_loss = -tf.reduce_mean(tf.log(d_image + eps) + tf.log(1. - d_z + eps)) / 2.0

    def _create_generator(self,
                          z: tf.Tensor,
                          labels: tf.Tensor):
        h, w, c = self._image_shape
        with tf.variable_scope('generator'):
            h, w = h // (2 ** 2), w // (2 ** 2)
            x = tf.concat([z, labels], axis=-1)
            x = tf.layers.dense(x, units=h*w*64, activation='relu')
            x = tf.layers.dropout(x, rate=self._dropout_rate)
            x = tf.reshape(x, [-1, h, w, 64])

            x = self._conv_g(x, 64, upsample=True, dropout=True)
            x = self._conv_g(x, 32, dropout=True)
            x = self._conv_g(x, 32, upsample=True)
            return tf.layers.conv2d(inputs=x,
                                    filters=c,
                                    kernel_size=3,
                                    padding='same',
                                    activation='sigmoid',
                                    kernel_initializer='he_normal')

    def _create_discriminator(self,
                              images: tf.Tensor,
                              labels: tf.Tensor):
        with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
            x = self._conv_d(images, filters=64, dropout=True, pooling=True, additional_labels=labels)
            x = self._conv_d(x, filters=64, dropout=True)
            x = self._conv_d(x, filters=128, dropout=True, pooling=True)
            x = self._conv_d(x, filters=128, dropout=True)

            x = tf.reduce_mean(x, axis=[1, 2], keep_dims=False)

            x = tf.layers.dense(inputs=x, units=128, activation='relu')
            x = tf.layers.dense(inputs=x, units=128, activation='relu')

            return tf.layers.dense(inputs=x, units=1, activation='sigmoid')

    def _conv_g(self,
                inputs,
                filters: Union[int, tf.Tensor],
                upsample: bool = False,
                dropout: bool = False):
        x = inputs
        if upsample:
            size = x.shape[1] * 2, x.shape[2] * 2
            x = tf.image.resize_images(images=x,
                                       size=size)
        x = tf.layers.conv2d(inputs=x,
                             filters=filters,
                             kernel_size=3,
                             padding='same',
                             activation='relu',
                             kernel_initializer='he_normal')
        if dropout:
            x = tf.layers.dropout(inputs=x,
                                  rate=self._dropout_rate)
        return x

    def _conv_d(self,
                inputs,
                filters: int,
                dropout: bool = False,
                pooling: bool = False,
                additional_labels: Optional[tf.Tensor] = None):
        x = tf.layers.conv2d(inputs=inputs,
                             filters=filters,
                             kernel_size=3,
                             padding='same',
                             kernel_initializer='he_normal')
        if additional_labels is not None:
            x = self._add_labels_to_conv2d(x, additional_labels)
        x = tf.nn.relu(x)
        if dropout:
            x = tf.layers.dropout(x, rate=self._dropout_rate)
        if pooling:
            x = tf.layers.max_pooling2d(inputs=x,
                                        strides=2,
                                        pool_size=2)
        return x

    def _add_labels_to_conv2d(self,
                              conv: tf.Tensor,
                              labels: tf.Tensor):
        h, w = self._image_shape[:2]
        c = self._num_classes
        labels = tf.tile(input=labels, multiples=[1, h * w])
        labels = tf.reshape(labels, [-1, h, w, c])
        return tf.concat([conv, labels], axis=-1)


if __name__ == '__main__':
    image_shape = 32, 32, 3
    num_classes = 10
    latent_dim = 2
    batch_size = 32

    model = Model(image_shape, num_classes, latent_dim)

    g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')
    train_g_op = layers.optimize_loss(
        model.g_loss,
        tf.train.get_global_step(),
        optimizer='Adam',
        learning_rate=0.0001,
        variables=g_vars)

    d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
    train_d_op = layers.optimize_loss(
        model.d_loss,
        tf.train.get_global_step(),
        optimizer='Adam',
        learning_rate=0.0001,
        variables=d_vars)

    # print('\n'.join([str(v) for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]))
    # print()
    # print('\n'.join([str(v) for v in g_vars]))
    # print()
    # print('\n'.join([str(v) for v in d_vars]))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        images = np.zeros((batch_size,) + image_shape, dtype=np.float32)
        labels = np.zeros((batch_size, num_classes), dtype=np.float32)
        z = np.zeros((batch_size, latent_dim), dtype=np.float32)

        loss = sess.run(train_g_op, feed_dict={
            model.z_input: z,
            model.labels_input: labels,
        })
        print('G loss: {}'.format(loss))

        loss = sess.run(train_d_op, feed_dict={
            model.images_input: images,
            model.z_input: z,
            model.labels_input: labels,
        })
        print('D loss: {}'.format(loss))
