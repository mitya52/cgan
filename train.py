import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np
import cv2

from itertools import product

from generator import Generator
from gan import Model


def compute_accuracy(labels, predictions):
    labels = np.argmax(labels, axis=-1)
    predictions = np.argmax(predictions, axis=-1)
    return float(np.count_nonzero(labels == predictions)) / len(labels)


def visualize_images(images, grid, scale):
    bs, h, w, c = images.shape
    canvas = np.zeros((grid[0] * h, grid[1] * w, c), dtype=np.uint8)
    for y, x in product(range(grid[0]), range(grid[1])):
        canvas[y*h:(y+1)*h, x*w:(x+1)*w, :] = (images[y*grid[1]+x] * 256).astype(np.uint8)
    h, w = canvas.shape[:2]
    h, w = int(scale * h), int(scale * w)
    return cv2.resize(canvas, (w, h), interpolation=cv2.INTER_AREA)


def steps_by_loss(loss):
    steps = int(10 * (loss ** 2) + 10)
    return max(min(1, steps), 10)


def train(batch_size: int,
          epochs: int):
    train_generator = Generator(batch_size=batch_size, train=True)
    val_generator = Generator(batch_size=batch_size, train=False)

    assert (train_generator.output_shape() == val_generator.output_shape())
    assert (train_generator.num_classes() == val_generator.num_classes())

    image_shape = train_generator.output_shape()
    num_classes = train_generator.num_classes()
    latent_dim = 2

    model = Model(image_shape, num_classes, latent_dim)

    g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')
    train_g_op = layers.optimize_loss(
        model.g_loss,
        tf.train.get_global_step(),
        optimizer='Adam',
        learning_rate=0.00003,
        variables=g_vars)

    d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
    train_d_op = layers.optimize_loss(
        model.d_loss,
        tf.train.get_global_step(),
        optimizer='Adam',
        learning_rate=0.00001,
        variables=d_vars)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        d_loss = 1e15
        g_loss = 1e15
        for epoch in range(epochs):
            # D training phase
            losses = []
            for d_step, (images, labels) in enumerate(train_generator.generate_epoch()):
                z = np.random.uniform(size=(batch_size, latent_dim))
                loss = sess.run(train_d_op, feed_dict={
                    model.images_input: images,
                    model.z_input: z,
                    model.labels_input: labels,
                })
                losses.append(loss)
                if d_step == steps_by_loss(d_loss):
                    break
            d_loss = np.mean(losses)

            # G training phase
            losses = []
            for _ in range(steps_by_loss(g_loss)):
                z = np.random.uniform(size=(batch_size, latent_dim))
                labels = np.zeros((batch_size, num_classes), dtype=np.float32)
                labels[np.arange(batch_size), np.random.randint(num_classes, size=batch_size)] = 1.0
                loss = sess.run(train_g_op, feed_dict={
                    model.z_input: z,
                    model.labels_input: labels,
                })
                losses.append(loss)
            g_loss = np.mean(losses)

            print('D train new loss: {}'.format(d_loss))
            print('G train last loss: {}'.format(g_loss))
            print()

            # visualize generator output
            samples_per_class = 8
            labels = np.eye(num_classes, num_classes, dtype=np.float32)
            labels = np.concatenate([labels] * samples_per_class)
            z = np.random.uniform(size=(num_classes * samples_per_class, latent_dim))
            generated = sess.run(model.generated, feed_dict={
                model.z_input: z,
                model.labels_input: labels,
            })

            image = visualize_images(generated, grid=(samples_per_class, num_classes), scale=4)
            cv2.imshow('images', image)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break


if __name__ == '__main__':
    batch_size = 64
    epochs = 10000

    train(batch_size, epochs)