import tensorflow as tf
from ops import *

class local_style:
    def __init__(self, batch_size=1, gf_num=64, df_num=64, input_dim=3, output_dim=3,
                 image_size=256, patch_num=4, output_size=256, times=5, lambd=10, reuse=False):
        #self.sess = sess
        self.batch_size = batch_size
        self.image_size=image_size
        self.gf_num = gf_num
        self.df_num = df_num
        self.out_dim = output_dim
        self.input_dim = input_dim
        self.patch_num = patch_num
        self.output_size = output_size
        self.times = times
        self.lambd = lambd
        self.reuse = reuse

    def generator(self,  name, image):
        s = self.output_size
        s2, s4, s8, s16, s32, s64, s128 = int(s / 2), int(s / 4), int(s / 8), int(s / 16), int(s / 32), int(s / 64), int(s / 128)
        with tf.variable_scope(name):
            g1 = conv2d("g1_conv_bn", image, self.gf_num)
            g2 = conv2d_bn("g2_conv_bn", lrelu(g1), self.gf_num*2)
            g3 = conv2d_bn("g3_conv_bn", lrelu(g2), self.gf_num * 4)
            g4 = conv2d_bn("g4_conv_bn", lrelu(g3), self.gf_num * 8)
            g5 = conv2d_bn("g5_conv_bn", lrelu(g4), self.gf_num * 8)
            g6 = conv2d_bn("g6_conv_bn", lrelu(g5), self.gf_num * 8)
            g7 = conv2d_bn("g7_conv_bn", lrelu(g6), self.gf_num * 8)
            g8 = conv2d("g8_conv_bn", lrelu(g7), self.gf_num * 8)
            d1 = dconv2d_bn("d1_conv_bn", relu(g8), [self.batch_size, s128, s128, self.gf_num*8])
            d1 = tf.nn.dropout(d1, 0.5)
            d1 = tf.concat([d1, g7], 3)
            d2 = dconv2d_bn("d2_conv_bn", relu(d1), [self.batch_size, s64, s64, self.gf_num * 8])
            d2 = tf.nn.dropout(d2, 0.5)
            d2 = tf.concat([d2, g6], 3)
            d3 = dconv2d_bn("d3_conv_bn", relu(d2), [self.batch_size, s32, s32, self.gf_num * 8])
            d3 = tf.nn.dropout(d3, 0.5)
            d3 = tf.concat([d3, g5], 3)
            d4 = dconv2d_bn("d4_conv_bn", relu(d3), [self.batch_size, s16, s16, self.gf_num * 8])
            d4 = tf.concat([d4, g4], 3)
            d5 = dconv2d_bn("d5_conv_bn", relu(d4), [self.batch_size, s8, s8, self.gf_num*4])
            d5 = tf.concat([d5, g3], 3)
            d6 = dconv2d_bn("d6_conv_bn", relu(d5), [self.batch_size, s4, s4, self.gf_num*2])
            d6 = tf.concat([d6, g2], 3)
            d7 = dconv2d_bn("d7_conv_bn", relu(d6), [self.batch_size, s2, s2, self.gf_num])
            d7 = tf.concat([d7, g1], 3)
            d8 = dconv2d("d8_conv_bn", relu(d7), [self.batch_size, s, s, self.out_dim])
            return tf.nn.tanh(d8)


    def discriminator(self, name, image, reuse=False):
        with tf.variable_scope(name, reuse=self.reuse):
            dis1_1 = conv2d("dis1_1", image, self.df_num, reuse=reuse)
            #dis1_2 = conv2d_bn("dis1_2", lrelu(dis1_1), self.df_num, s=1, reuse=reuse)
            #dis1_3 = conv2d_bn("dis1_3", lrelu(dis1_2), self.df_num, s=1, reuse=reuse)
            #dis1 = tf.concat([dis1_2, dis1_1], 3, name="dis1")
            dis2_1 = conv2d_bn("dis2_1", lrelu(dis1_1), self.df_num*2, reuse=reuse)
            #dis2_2 = conv2d_bn("dis2_2", lrelu(dis2_1), self.df_num*2, s=1, reuse=reuse)
            #dis2_3 = conv2d_bn("dis2_3", lrelu(dis2_2), self.df_num * 2, s=1, reuse=reuse)
            #dis2 = tf.concat([dis2_2, dis2_1], 3, name="dis2")
            dis3_1 = conv2d_bn("dis3_1", lrelu(dis2_1), self.df_num * 4, reuse=reuse)
            #dis3_2 = conv2d_bn("dis3_2", lrelu(dis3_1), self.df_num*4, s=1, reuse=reuse)
            #dis3_3 = conv2d_bn("dis3_3", lrelu(dis3_2), self.df_num * 4, s=1, reuse=reuse)
            #dis3 = tf.concat([dis3_2, dis3_1], 3, name="dis3")
            dis4_1 = conv2d_bn("dis4_1", lrelu(dis3_1), self.df_num * 8, reuse=reuse)
            #dis4_2 = conv2d_bn("dis4_2", lrelu(dis4_1), self.df_num * 8, s=1, reuse=reuse)
            #dis4_3 = conv2d_bn("dis4_3", lrelu(dis4_2), self.df_num * 8, s=1, reuse=reuse)
            #dis4 = tf.concat([dis4_2, dis4_1], 3, name="dis4")
            dis5_1 = conv2d_bn("dis5_1", lrelu(dis4_1), self.df_num * 8, reuse=reuse)
            #dis5_2 = conv2d_bn("dis5_2", lrelu(dis5_1), self.df_num * 8, s=1, reuse=reuse)
            #dis5_3 = conv2d_bn("dis5_3", lrelu(dis5_2), self.df_num * 8, s=1, reuse=reuse)
            #dis5 = tf.concat([dis5_2, dis5_1], 3, name="dis5")
            #dis6_1 = conv2d_bn("dis6_1", lrelu(dis5_1), self.df_num * 8, reuse=reuse)
            #dis7_1 = conv2d_bn("dis7_1", lrelu(dis6_1), self.df_num * 8, reuse=reuse)
            dis8 = conv2d_bn("dis8", lrelu(dis5_1), self.df_num * 8, s=1, reuse=reuse)
            dis9 = linear("dis9", tf.layers.flatten(lrelu(dis8)), 1, reuse=reuse)
            return dis9






















