from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv


def plot_image(image):
    plt.imshow(image.reshape(28, 28), cmap='binary')
    plt.show()


def save_array_as_image(index, image):
    image = image.reshape(28, 28)
    img = np.zeros([28, 28, 3])

    img[:, :, 0] = image * 255.0
    img[:, :, 1] = image * 255.0
    img[:, :, 2] = image * 255.0

    cv.imwrite(str(index).zfill(4) + '.jpg', img)


# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

for i in range(10):
    save_array_as_image(i, mnist.train.images[i])
