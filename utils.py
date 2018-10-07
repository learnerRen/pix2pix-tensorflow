from scipy.misc import imsave, imread
import tensorflow as tf
import numpy as np
import os
image_name = os.listdir("./edges2handbags/train")
image_num = len(image_name)


def show_result(value, path):
    imsave(path, value)


def random_crop(x1, x2, num, seed=None):
    #im =tf.image.resize_bicubic(x, [284, 284])
    a1 = []
    a2 = []
    s = np.random.randint(0, 1000000000, [4])
    for i in range(int(num/2)):
        for j in range(int(num/2)):
            a = x1[:, 32*i:32*i+224, 32*j:32*j+224, :]#tf.random_crop(x1, [x1.get_shape().as_list()[0], 224, 224, x1.get_shape().as_list()[3]], seed=s[i])
            b = x2[:, 32*i:32*i+224, 32*j:32*j+224, :]#tf.random_crop(x2, [x2.get_shape().as_list()[0], 224, 224, x2.get_shape().as_list()[3]], seed=s[i])
            a1.append(a)
            a2.append(b)
    return tf.concat(a1, axis=0), tf.concat(a2, axis=0)


def load_data(num, im_size=256, im_channel=3):
    a = np.random.randint(0, image_num, [num], dtype=np.int)
    content_image = np.zeros([num, im_size, im_size, im_channel], dtype=np.float32)
    style_image = np.zeros([num, im_size, im_size, im_channel], dtype=np.float32)
    for i in range(num):
        image = (imread("./edges2handbags/train/" + image_name[a[i]]).astype(np.float32) / 127.5 - 1)
        content_image[i, :, :, :] = image[np.newaxis, :, 0:256, :]
        style_image[i, :, :, :] = image[np.newaxis, :, 256:512, :]
    return content_image, style_image
