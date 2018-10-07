import models
import os
import numpy as np
import time
import tensorflow
from scipy.misc import imread
from models import *
from utils import *
to_restore = False
weights_output_path = 'weights_output'
model_name = 'styleGAN'
image_output = 'image_out'
max_iteration = 300000
reuse = True
times = 2#update discriminator times times, update generator once
lambd = 10
os.environ['CUDA_VISIBLE_DEVICES']=0
if os.path.isdir(image_output):
    pass
else:
    os.mkdir(image_output)
m = models.local_style(batch_size=4)
#content_image = np.load("train_content.npy")
#style_image = np.load("train_style.npy")
#print(style_image[0][np.newaxis, :, :, :].shape)
#image_name = os.listdir("./edges2handbags/train")
#image_num = len(image_name)
#image = (imread("./edges2handbags/train/" + image_name[0]).astype(np.float32)/127.5-1)
#stable_content_image = image[np.newaxis, :, 0:256, :]
stable_content_image, stable_style_image = load_data(m.batch_size)
imsave(os.path.join(image_output, "sample_image", "style_sample.jpg"), np.squeeze(stable_style_image[0 , :, : ,:]))


def train(m):
    z_prior = tf.placeholder(np.float32,
                             [m.batch_size, int(m.image_size), int(m.image_size), m.input_dim])
    x = tf.placeholder(np.float32,
                       [None, m.image_size, m.image_size, m.out_dim])
    #y_content = tf.placeholder(tf.float32,
    #                           [m.batch_size, m.image_size, m.image_size, m.out_dim])
    x_generate = m.generator("gen", z_prior)
    x_generate_, x_ = random_crop(tf.concat([x_generate, z_prior], axis=3), tf.concat([x, z_prior], axis=3), 4)
    print(x_generate_)
    y_real = m.discriminator("dis", x_)
    y_generate = m.discriminator("dis", x_generate_, reuse=True)
    # the loss of WGAN-GP is the same like WGAN
    d_loss = -tf.reduce_mean(y_generate) + tf.reduce_mean(y_real)
    g_loss = tf.reduce_mean(y_generate)
    differences = x_generate_ - x_
    alpha = tf.random_uniform([1, 1, 1, 1], minval=0., maxval=1.)
    interpolates = x_ + alpha * differences
    grad = tf.gradients(m.discriminator("dis", interpolates, reuse=True), [interpolates])[0]
    grad_norm = tf.sqrt(tf.reduce_sum(tf.square(grad), reduction_indices=[1, 2, 3]))
    grad_penality = tf.reduce_mean((grad_norm - 1.) ** 2)
    d_loss = d_loss + lambd * grad_penality
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'dis' in var.name]
    g_vars = [var for var in t_vars if 'gen' in var.name]
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        dstyle_train = tf.train.AdamOptimizer(0.0002, 0.5, 0.999).minimize(d_loss, var_list=d_vars)
        g_train = tf.train.AdamOptimizer(0.0002, 0.5, 0.999).minimize(g_loss, var_list=g_vars)
    with tf.Session() as sess:
        saver = tf.train.Saver()
        if to_restore:
            check_file = tf.train.latest_checkpoint(weights_output_path)
            saver.restore(sess, check_file)
        else:
            if os.path.exists(weights_output_path):
                pass
            else:
                os.mkdir(weights_output_path)
        sess.run(tf.global_variables_initializer())
        start = time.time()
        config=tf.Condition
        for i in range(max_iteration):
            content_image, style_image = load_data(m.batch_size)
            sess.run(dstyle_train, feed_dict={z_prior: content_image,
                                              x: style_image})
            if i % times == 0:
                content_image, style_image = load_data(m.batch_size)
                sess.run(g_train, feed_dict={z_prior: content_image,
                                             x: style_image})
            if i % 10000 == 0:
                end = time.time()
                d_loss_value, grad_norm_value = sess.run([d_loss, grad_norm],
                                                         feed_dict={z_prior: stable_content_image,
                                                                    x: stable_style_image,})
                g_loss_value = sess.run(g_loss, feed_dict={z_prior: stable_content_image,
                                                           x: stable_style_image,})
                if d_loss_value<g_loss_value:
                    time=1
                else:
                    time=2
                #print(style_image.shape)
                print("After {} training times, the d_loss is {}, and the g_loss is {}, the grad_norm is{} using {} seconds".format(i, d_loss_value, g_loss_value, grad_norm_value, end-start))
                start = end
                x_generate_value = sess.run(x_generate, feed_dict={z_prior: content_image})
                x_generate_value = np.squeeze(x_generate_value[0])
                saver.save(sess, os.path.join(weights_output_path, model_name), i)
                show_result(x_generate_value, os.path.join(image_output, "random_image",
                                                           "ramdom"+str(i)+'.jpg'))
                print("random image{} is saved ".format(int(i/10000)))
                imsave(os.path.join(image_output, "random_image", "random_style"+str(i)+".jpg"),
                       np.squeeze(style_image[0, :, :, :]))
                x_sample = sess.run(x_generate, feed_dict={z_prior: stable_content_image})
                x_sample = np.squeeze(x_sample[0])
                show_result(x_sample, os.path.join(image_output, "sample_image", "sample"+str(i)+'.jpg'))
                print("sample image{} is saved ".format(int(i / 10000)))

train(m)
