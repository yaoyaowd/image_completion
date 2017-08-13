from __future__ import division

import os
import time
import numpy as np
import tensorflow as tf

from ops import batch_norm, lrelu, conv2d, dilated_conv2d, conv2d_transpose, linear
from utils import dataset_files, get_image

D_ITERATIONS = 4

class WGANTensor(object):
    def __init__(self,
                 sess,
                 learning_rate=0.0001,
                 beta1=0.0,
                 beta2=0.5,
                 batch_size=32,
                 image_size=64,
                 lam=100,
                 checkpoint_dir=None):
        self.sess = sess
        self.batch_size = batch_size
        self.image_size = image_size
        self.image_shape = [image_size, image_size, 3]
        self.lam = lam
        self.checkpoint_dir = checkpoint_dir
        self.global_step = tf.Variable(0, trainable=False)
        self.d_bns = [batch_norm(name='d_bn{}'.format(i)) for i in range(10)]
        self.g_bns = [batch_norm(name='g_bn{}'.format(i, )) for i in range(15)]

        self.images = tf.placeholder(tf.float32, [batch_size] + self.image_shape, name='images')
        self.masks = tf.placeholder(tf.float32, [batch_size] + self.image_shape, name='masks')
        self.masked_images = tf.multiply(self.images, self.masks)
        self.G = self.generator(self.masked_images)
        self.G_final = self.masked_images + tf.multiply(self.G, (1 - self.masks))

        self.d_loss_real = tf.reduce_mean(self.discriminator(self.images))
        self.d_loss_fake = tf.reduce_mean(self.discriminator(self.G, reuse=True))
        self.d_loss = self.d_loss_fake - self.d_loss_real
        self.g_loss_mse = tf.reduce_mean(tf.contrib.layers.flatten(tf.square(self.G - self.images)))
        self.g_loss = self.lam * self.g_loss_mse - self.d_loss_fake

        self.epsilon = tf.random_uniform([], 0.0, 1.0)
        self.G_epsilon = self.epsilon * self.images + (1 - self.epsilon) * self.G
        self.D_epsilon = self.discriminator(self.G_epsilon, reuse=True)
        self.gradients = tf.gradients(self.D_epsilon, self.G_epsilon)[0]
        self.slopes = tf.sqrt(tf.reduce_sum(tf.square(self.gradients), reduction_indices=[1]))
        self.gradient_penalty = 100.0 * tf.reduce_mean((self.slopes - 1.0) ** 2)
        self.d_loss += self.gradient_penalty

        self.d_loss_real_summary = tf.summary.scalar("d_loss_real", self.d_loss_real)
        self.d_loss_fake_summary = tf.summary.scalar("d_loss_fake", self.d_loss_fake)
        self.gradient_penalty_summary = tf.summary.scalar("d_loss_gp", self.gradient_penalty)
        self.d_loss_summary = tf.summary.scalar("d_loss", self.d_loss)
        self.d_summary = tf.summary.merge([
            self.d_loss_real_summary, self.d_loss_fake_summary, self.gradient_penalty_summary, self.d_loss_summary])

        self.masked_images_summary = tf.summary.image("masked_image", self.masked_images)
        self.G_summary = tf.summary.image("g", self.G)
        self.G_final_summary = tf.summary.image("g_final", self.G_final)
        self.g_loss_mse_summary = tf.summary.scalar("g_mse", self.g_loss_mse)
        self.g_loss_summary = tf.summary.scalar("g", self.g_loss)
        self.g_summary = tf.summary.merge([
            self.G_summary, self.masked_images_summary, self.G_final_summary,
            self.g_loss_mse_summary, self.g_loss_summary])

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]
        self.d_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1, beta2=beta2).minimize(
            self.d_loss, var_list=self.d_vars, global_step=self.global_step)
        self.g_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1, beta2=beta2).minimize(
            self.g_loss, var_list=self.g_vars)

        self.writer = tf.summary.FileWriter(os.path.join(self.checkpoint_dir, "logs"), self.sess.graph)
        self.saver = tf.train.Saver(max_to_keep=10)

    def train(self, config):
        tf.global_variables_initializer().run()
        start_time = time.time()
        if self.load(self.checkpoint_dir):
            print("""Loading existing model""")
        else:
            print("""Initalizing new model""")

        for epoch in range(config.epoch):
            data = dataset_files(config.dataset)
            np.random.shuffle(data)
            batch_idx = min(len(data), config.train_size) // self.batch_size
            for idx in range(batch_idx):
                batch_files = data[idx * config.batch_size: (idx + 1) * config.batch_size]
                batch = [get_image(batch_file, self.image_size) for batch_file in batch_files]
                batch_images = np.array(batch).astype(np.float32)
                masks = np.ones([config.batch_size, config.image_size, config.image_size, 3])
                masks[:, 16:48, 16:48, :] = 0.0

                _, d_loss, summary_str = self.sess.run(
                    [self.d_optim, self.d_loss, self.d_summary],
                    feed_dict={self.images: batch_images, self.masks: masks})
                self.writer.add_summary(summary_str, self.global_step.eval())

                if idx > 0 and idx % D_ITERATIONS == D_ITERATIONS - 1:
                    _, g_loss, summary_str = self.sess.run(
                        [self.g_optim, self.g_loss, self.g_summary],
                        feed_dict={self.images: batch_images, self.masks: masks})
                    self.writer.add_summary(summary_str, self.global_step.eval())
                    print("Epoch: [{:2d}] [{:4d}/{:4d}] time: {:4.4f}, d_loss: {:.8f}, g_loss: {:.8f}".format(
                        epoch, idx, batch_idx, time.time() - start_time, d_loss, g_loss))

                if idx % 100 == 0:
                    self.save(config.checkpoint_dir, self.global_step.eval())

    def generator(self, images):
        with tf.variable_scope("generator"):
            g_h0 = tf.nn.relu(conv2d(images, 64, 5, 5, 1, 1, name='g_h0'))
            g_h1 = tf.nn.relu(self.g_bns[0](conv2d(g_h0, 128, 3, 3, 2, 2, name='g_h1')))
            g_h2 = tf.nn.relu(self.g_bns[1](conv2d(g_h1, 128, 3, 3, 1, 1, name='g_h2')))
            g_h3 = tf.nn.relu(self.g_bns[2](conv2d(g_h2, 256, 3, 3, 2, 2, name='g_h3')))
            g_h4 = tf.nn.relu(self.g_bns[3](conv2d(g_h3, 256, 3, 3, 1, 1, name='g_h4')))
            g_h5 = tf.nn.relu(self.g_bns[4](conv2d(g_h4, 256, 3, 3, 1, 1, name='g_h5')))
            g_h6 = tf.nn.relu(self.g_bns[5](dilated_conv2d(g_h5, 256, 3, 3, 2, name='g_h6')))
            g_h7 = tf.nn.relu(self.g_bns[6](dilated_conv2d(g_h6, 256, 3, 3, 4, name='g_h7')))
            g_h8 = tf.nn.relu(self.g_bns[7](dilated_conv2d(g_h7, 256, 3, 3, 8, name='g_h8')))
            g_h9 = tf.nn.relu(self.g_bns[8](dilated_conv2d(g_h8, 256, 3, 3, 16, name='g_h9')))
            g_h10 = tf.nn.relu(self.g_bns[9](conv2d(g_h9, 256, 3, 3, 1, 1, name='g_h10')))
            g_h11 = tf.nn.relu(self.g_bns[10](conv2d(g_h10, 256, 3, 3, 1, 1, name='g_h11')))
            g_h12 = tf.nn.relu(self.g_bns[11](conv2d_transpose(
                g_h11, [self.batch_size, int(self.image_size/2), int(self.image_size/2), 128], 4, 4, 2, 2, name='g_h12')))
            g_h13 = tf.nn.relu(self.g_bns[12](conv2d(g_h12, 128, 3, 3, 1, 1, name='g_h13')))
            g_h14 = tf.nn.relu(self.g_bns[13](conv2d_transpose(
                g_h13, [self.batch_size, self.image_size, self.image_size, 64], 4, 4, 2, 2, name='g_h14')))
            g_h15 = tf.nn.relu(self.g_bns[14](conv2d(g_h14, 32, 3, 3, 1, 1, name='g_h15')))
            g_h16 = tf.nn.tanh(conv2d(g_h15, 3, 3, 3, 1, 1, name='g_h16'))
            return g_h16

    def discriminator(self, images, reuse=False):
        with tf.variable_scope('discriminator', reuse=reuse):
            gd_h0 = lrelu(conv2d(images, 64, name="d_gd_h0_conv")) # 32*32*64
            gd_h1 = lrelu(self.d_bns[0](conv2d(gd_h0, 128, name='d_gd_h1_conv'))) # 16*16*128
            gd_h2 = lrelu(self.d_bns[1](conv2d(gd_h1, 256, name='d_gd_h2_conv'))) # 8*8*256
            gd_h3 = lrelu(self.d_bns[2](conv2d(gd_h2, 512, name='d_gd_h3_conv'))) # 4*4*512
            gd_fc0 = linear(tf.reshape(gd_h3, [self.batch_size, int(4 * 4 * 512)]), 1024, 'd_gd_fc0')
            gd_fc1 = linear(gd_fc0, 1, 'd_gd_fc1')
            return gd_fc1

    def save(self, checkpoint_dir, step):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, "wgan_tensor"),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            return True
        else:
            return False


flags = tf.app.flags
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0001, "Learning rate of for adam [0.0001]")
flags.DEFINE_float("beta1", 0.0, "Momentum term of adam [0.0]")
flags.DEFINE_float("beta2", 0.9, "Beta 2 of adam [0.9]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [32]")
flags.DEFINE_integer("image_size", 64, "The size of image to use")
flags.DEFINE_string("dataset", "lfw-aligned-64", "Dataset directory.")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
FLAGS = flags.FLAGS

if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
if not os.path.exists(FLAGS.sample_dir):
    os.makedirs(FLAGS.sample_dir)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    model = WGANTensor(
        sess,
        learning_rate=FLAGS.learning_rate,
        beta1=FLAGS.beta1,
        beta2=FLAGS.beta2,
        image_size=FLAGS.image_size,
        batch_size=FLAGS.batch_size,
        checkpoint_dir=FLAGS.checkpoint_dir)
    model.train(FLAGS)