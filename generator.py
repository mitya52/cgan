import numpy as np
import cv2

from keras.datasets import mnist
from keras.utils import to_categorical


class Generator:
    def __init__(self,
                 batch_size: int,
                 train: bool = False):
        self._output_shape = 28, 28, 1
        self._batch_size = batch_size
        self._num_classes = 10

        x, y = mnist.load_data()[0 if train else 1]
        x = np.reshape(x, (-1,) + self._output_shape).astype(np.float32)
        y = to_categorical(y, self._num_classes).astype(np.float32)

        self._dataset = x, y

    def generate_epoch(self):
        length = len(self._dataset[0])
        steps = length // self._batch_size
        indexes = np.random.permutation(length)
        self._dataset = self._dataset[0][indexes], self._dataset[1][indexes]
        for step in range(steps):
            idx, idy = step * self._batch_size, (step + 1) * self._batch_size
            yield self._dataset[0][idx: idy], self._dataset[1][idx: idy]


if __name__ == '__main__':
    generator = Generator(batch_size=32)
    for x, y in generator.generate_epoch():
        x, y = np.squeeze(x[0]), np.argmax(y[0])

        print('Label: {}'.format(y))
        x = cv2.resize(x, (256, 256), interpolation=cv2.INTER_AREA)
        cv2.imshow('image', x)
        k = cv2.waitKey(0) & 0xFF
        if k == 27:
            break
