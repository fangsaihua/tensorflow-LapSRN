#!/usr/bin/env python2
# -*- coding: utf-8 -*-


# This is a re-implementation of training code of our paper:
# X. Fu, J. Huang, D. Zeng, Y. Huang, X. Ding and J. Paisley. “Removing Rain from Single Images via a Deep Detail Network”, CVPR, 2017.
# author: Xueyang Fu (fxy@stu.xmu.edu.cn)

import os
import h5py
import re
import numpy as np
import tensorflow as tf
import cv2
from tifffile import imsave

os.environ['CUDA_VISIBLE_DEVICES'] = "0"  # select GPU device

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('num_h5_file', 100,
                            """number of training h5 files.""")
tf.app.flags.DEFINE_integer('num_patches', 160,
                            """number of patches in each h5 file.""")
tf.app.flags.DEFINE_integer('learning_rate', 0.001,
                            """learning rate.""")
tf.app.flags.DEFINE_integer('epoch', 20,
                            """epoch.""")
tf.app.flags.DEFINE_integer('batch_size', 16,
                            """Batch size.""")
tf.app.flags.DEFINE_integer('num_channels', 4,
                            """Number of channels of the input.""")
tf.app.flags.DEFINE_integer('image_size', 256,
                            """Size of the images.""")
tf.app.flags.DEFINE_integer('label_size', 256,
                            """Size of the labels.""")
tf.app.flags.DEFINE_string("data_path", "./data_generation/h5data/", "The path of h5 files")

tf.app.flags.DEFINE_string("save_model_path", "./model/", "The path of saving model")


# read h5 files
def read_data(file):
    with h5py.File(file, 'r') as hf:
        ms = hf.get('data')
        pan_2x = hf.get('pan_x2')
        pan_4x = hf.get('pan_x4')
        label_2x = hf.get('label_x2')
        label_4x = hf.get('label_x4')
        return np.array(ms), np.array(pan_2x), np.array(pan_4x), np.array(label_2x), np.array(label_4x)


# guided filter
def guided_filter(data, num_patches=FLAGS.num_patches, width=FLAGS.image_size, height=FLAGS.image_size,
                  channel=FLAGS.num_channels):
    r = 15
    eps = 1.0
    batch_q = np.zeros((num_patches, height, width, channel))
    for i in range(num_patches):
        for j in range(channel):
            I = data[i, :, :, j]
            p = data[i, :, :, j]
            ones_array = np.ones([height, width])
            N = cv2.boxFilter(ones_array, -1, (2 * r + 1, 2 * r + 1), normalize=False, borderType=0)
            mean_I = cv2.boxFilter(I, -1, (2 * r + 1, 2 * r + 1), normalize=False, borderType=0) / N
            mean_p = cv2.boxFilter(p, -1, (2 * r + 1, 2 * r + 1), normalize=False, borderType=0) / N
            mean_Ip = cv2.boxFilter(I * p, -1, (2 * r + 1, 2 * r + 1), normalize=False, borderType=0) / N
            cov_Ip = mean_Ip - mean_I * mean_p
            mean_II = cv2.boxFilter(I * I, -1, (2 * r + 1, 2 * r + 1), normalize=False, borderType=0) / N
            var_I = mean_II - mean_I * mean_I
            a = cov_Ip / (var_I + eps)
            b = mean_p - a * mean_I
            mean_a = cv2.boxFilter(a, -1, (2 * r + 1, 2 * r + 1), normalize=False, borderType=0) / N
            mean_b = cv2.boxFilter(b, -1, (2 * r + 1, 2 * r + 1), normalize=False, borderType=0) / N
            q = mean_a * I + mean_b
            batch_q[i, :, :, j] = q
    return batch_q


def blur_image(data):
    num_patches, height, width, channel = data.shape
    batch_q = np.zeros((num_patches, height, width, channel))
    for i in range(num_patches):
        for j in range(channel):
            img = data[i, :, :, j]
            img = cv2.blur(img, (11, 11))
            batch_q[i, :, :, j] = img

    return batch_q


# initialize weights
def create_kernel(name, shape, initializer=tf.contrib.layers.xavier_initializer()):
    regularizer = tf.contrib.layers.l2_regularizer(scale=1e-10)

    new_variables = tf.get_variable(name=name, shape=shape, initializer=initializer,
                                    regularizer=regularizer, trainable=True)
    return new_variables


def convt_F(detail, level):
    with tf.variable_scope(level):
        with tf.variable_scope('conv_1'):
            kernel = create_kernel(name='weights_1', shape=[3, 3, FLAGS.num_channels + 1, 32])
            biases = tf.Variable(tf.constant(0.0, shape=[32], dtype=tf.float32), trainable=True, name='biases_1')

            conv = tf.nn.conv2d(detail, kernel, [1, 1, 1, 1], padding='SAME')
            feature = tf.nn.bias_add(conv, biases)

            conv_shortcut = tf.nn.relu(feature)

        for i in range(4):
            with tf.variable_scope('conv_%s' % (i * 2 + 2)):
                kernel = create_kernel(name=('weights_%s' % (i * 2 + 2)), shape=[3, 3, 32, 32])
                biases = tf.Variable(tf.constant(0.0, shape=[32], dtype=tf.float32), trainable=True,
                                     name=('biases_%s' % (i * 2 + 2)))

                conv = tf.nn.conv2d(conv_shortcut, kernel, [1, 1, 1, 1], padding='SAME')
                feature = tf.nn.bias_add(conv, biases)

                feature_relu = tf.nn.relu(feature)

            with tf.variable_scope('conv_%s' % (i * 2 + 3)):
                kernel = create_kernel(name=('weights_%s' % (i * 2 + 3)), shape=[3, 3, 32, 32])
                biases = tf.Variable(tf.constant(0.0, shape=[32], dtype=tf.float32), trainable=True,
                                     name=('biases_%s' % (i * 2 + 3)))

                conv = tf.nn.conv2d(feature_relu, kernel, [1, 1, 1, 1], padding='SAME')
                feature = tf.nn.bias_add(conv, biases)

                feature_relu = tf.nn.relu(feature)

                conv_shortcut = tf.add(conv_shortcut, feature_relu)  # shortcut

        with tf.variable_scope('conv_10'):
            kernel = create_kernel(name='weights_10', shape=[3, 3, 32, FLAGS.num_channels])
            biases = tf.Variable(tf.constant(0.0, shape=[FLAGS.num_channels], dtype=tf.float32), trainable=True,
                                 name='biases_10')

            conv = tf.nn.conv2d(conv_shortcut, kernel, [1, 1, 1, 1], padding='SAME')
            feature = tf.nn.bias_add(conv, biases)

            feature_relu = tf.nn.relu(feature)

    return feature_relu


def convt_I(image, level):
    with tf.variable_scope(level):
        out = tf.layers.conv2d_transpose(image, FLAGS.num_channels, 3, strides=4,
                                         padding='SAME')  ###change up-sample scale

    return out


# network structure
def inference(ms, detail_ms, detail_pan_4x):
    ms_4x = convt_I(ms, 'M1')
    detail_ms_4x = convt_I(detail_ms, 'D1')
    detail_4x = convt_F(tf.concat([detail_ms_4x, detail_pan_4x], 3), 'F1')
    detail_4x_summ = tf.summary.image('detail_4x', detail_4x, max_outputs=4)
    ms_4x = tf.add(detail_4x, ms_4x)

    return ms_4x


if __name__ == '__main__':

    ms = tf.placeholder(tf.float32, shape=(None, FLAGS.image_size / 4, FLAGS.image_size / 4, FLAGS.num_channels))
    detail_ms = tf.placeholder(tf.float32, shape=(None, FLAGS.image_size / 4, FLAGS.image_size / 4, FLAGS.num_channels))
    label_2x = tf.placeholder(tf.float32, shape=(None, FLAGS.image_size / 2, FLAGS.image_size / 2, FLAGS.num_channels))
    detail_pan_2x = tf.placeholder(tf.float32, shape=(None, FLAGS.image_size / 2, FLAGS.image_size / 2, 1))
    label_4x = tf.placeholder(tf.float32, shape=(None, FLAGS.image_size, FLAGS.image_size, FLAGS.num_channels))
    detail_pan_4x = tf.placeholder(tf.float32, shape=(None, FLAGS.image_size, FLAGS.image_size, 1))

    ms_4x = inference(ms, detail_ms, detail_pan_4x)

    loss = tf.reduce_mean(tf.square(label_4x - ms_4x))  # MSE loss
    loss_sum_summ = tf.summary.scalar('loss', loss)

    lr_ = FLAGS.learning_rate
    lr = tf.placeholder(tf.float32, shape=[])
    g_optim = tf.train.AdamOptimizer(lr).minimize(loss)  # Optimization method: Adam

    saver = tf.train.Saver(max_to_keep=5)

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8  # GPU setting
    config.gpu_options.allow_growth = True

    data_path = FLAGS.data_path
    save_path = FLAGS.save_model_path
    epoch = int(FLAGS.epoch)

    with tf.Session(config=config) as sess:
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('logs/b/', sess.graph)

        sess.run(tf.global_variables_initializer())

        validation_data_name = "validation.h5"
        validation_ms, validation_pan_2x, validation_pan_4x, validation_label_2x, validation_label_4x = read_data(
            data_path + validation_data_name)

        validation_ms = np.transpose(validation_ms, (0, 2, 3, 1))  # image data
        validation_pan_2x = np.transpose(validation_pan_2x, (0, 2, 3, 1))  # image data
        validation_pan_4x = np.transpose(validation_pan_4x, (0, 2, 3, 1))  # image data
        validation_label_2x = np.transpose(validation_label_2x, (0, 2, 3, 1))  # image data
        validation_label_4x = np.transpose(validation_label_4x, (0, 2, 3, 1))  # image data

        validation_detail_ms = validation_ms - blur_image(validation_ms)  # detail layer
        validation_detail_pan_2x = validation_pan_2x - blur_image(validation_pan_2x)  # detail layer
        validation_detail_pan_4x = validation_pan_4x - blur_image(validation_pan_4x)  # detail layer

        if tf.train.get_checkpoint_state('./model/'):  # load previous trained model
            ckpt = tf.train.latest_checkpoint('./model/')
            saver.restore(sess, ckpt)
            ckpt_num = re.findall(r"\d", ckpt)
            if len(ckpt_num) == 3:
                start_point = 100 * int(ckpt_num[0]) + 10 * int(ckpt_num[1]) + int(ckpt_num[2])
            elif len(ckpt_num) == 2:
                start_point = 10 * int(ckpt_num[0]) + int(ckpt_num[1])
            else:
                start_point = int(ckpt_num[0])
            print("Load success")

        else:
            print("re-training")
            start_point = 0

        for j in range(start_point, epoch):  # epoch

            if j + 1 > (epoch / 3):  # reduce learning rate
                lr_ = FLAGS.learning_rate * 0.1
            if j + 1 > (2 * epoch / 3):
                lr_ = FLAGS.learning_rate * 0.01

            Training_Loss = 0.

            for num in range(FLAGS.num_h5_file):  # h5 files
                train_data_name = "train" + str(num + 1) + ".h5"
                train_ms, train_pan_2x, train_pan_4x, train_label_2x, train_label_4x = read_data(
                    data_path + train_data_name)

                train_ms = np.transpose(train_ms, (0, 2, 3, 1))  # image data
                train_pan_2x = np.transpose(train_pan_2x, (0, 2, 3, 1))  # image data
                train_pan_4x = np.transpose(train_pan_4x, (0, 2, 3, 1))  # image data
                train_label_2x = np.transpose(train_label_2x, (0, 2, 3, 1))  # image data
                train_label_4x = np.transpose(train_label_4x, (0, 2, 3, 1))  # image data

                train_detail_ms = train_ms - blur_image(train_ms)  # detail layer
                train_detail_pan_2x = train_pan_2x - blur_image(train_pan_2x)  # detail layer
                train_detail_pan_4x = train_pan_4x - blur_image(train_pan_4x)  # detail layer

                data_size = int(FLAGS.num_patches / FLAGS.batch_size)  # the number of batch
                for i in range(data_size):
                  rand_index = np.arange(int(i*FLAGS.batch_size),int((i+1)*FLAGS.batch_size))   # batch
                  batch_ms = train_ms[rand_index,:,:,:]
                  batch_detail_ms = train_detail_ms[rand_index,:,:,:]
                  batch_label_2x = train_label_2x[rand_index,:,:,:]
                  batch_label_4x = train_label_4x[rand_index,:,:,:]
                  batch_detail_pan_2x = train_detail_pan_2x[rand_index,:,:,:]
                  batch_detail_pan_4x = train_detail_pan_4x[rand_index,:,:,:]


                  _,lossvalue,summary_str = sess.run([g_optim,loss,merged], feed_dict={ms: batch_ms, detail_ms: batch_detail_ms, label_2x: batch_label_2x, label_4x: batch_label_4x,
                                                                    detail_pan_2x: batch_detail_pan_2x, detail_pan_4x: batch_detail_pan_4x, lr: lr_})
                  writer.add_summary(summary_str, j*FLAGS.num_h5_file*data_size + num*data_size + i)
                  writer.flush()

                  Training_Loss += lossvalue  # training loss

            Training_Loss /= (data_size * FLAGS.num_h5_file)

            model_name = 'model-epoch'  # save model
            save_path_full = os.path.join(save_path, model_name)
            saver.save(sess, save_path_full, global_step=j + 1)

            Validation_Loss = 0;
            data_size_ = int(FLAGS.num_patches / FLAGS.batch_size)  # the number of batch
            for i in range(data_size_):
                rand_index_ = np.arange(int(i * FLAGS.batch_size), int((i + 1) * FLAGS.batch_size))  # batch
                batch_ms_ = validation_ms[rand_index_, :, :, :]
                batch_detail_ms_ = validation_detail_ms[rand_index_, :, :, :]
                batch_label_2x_ = validation_label_2x[rand_index_, :, :, :]
                batch_label_4x_ = validation_label_4x[rand_index_, :, :, :]
                batch_detail_pan_2x_ = validation_detail_pan_2x[rand_index_, :, :, :]
                batch_detail_pan_4x_ = validation_detail_pan_4x[rand_index_, :, :, :]
                batch_pan_4x = validation_pan_4x[rand_index_, :, :, :]

                lossvalue_ = sess.run(loss, feed_dict={ms: batch_ms_, detail_ms: batch_detail_ms_,
                                                       label_2x: batch_label_2x_, label_4x: batch_label_4x_,
                                                       detail_pan_2x: batch_detail_pan_2x_,
                                                       detail_pan_4x: batch_detail_pan_4x_,
                                                       lr: lr_})  # validation loss

                # if j == epoch - 1:
                #     output1[np.where(output1 < 0.)] = 0
                #     output2[np.where(output2 > 1.)] = 1
                #     os.makedirs('./output/' + str(i))
                #
                #     for k in range(FLAGS.batch_size):
                #         final_output1 = output1[k, :, :, :]
                #         final_output2 = output2[k, :, :, :]
                #         label_output1 = batch_label_2x_[k, :, :, :]
                #         label_output2 = batch_label_4x_[k, :, :, :]
                #         pan = batch_pan_4x[k, :, :, :]
                #         ms1 = batch_ms_[k, :, :, :]
                #         imsave('./output/%d/out_1_%d.TIF' % (i, k), final_output1)
                #         imsave('./output/%d/out_2_%d.TIF' % (i, k), final_output2)
                #         imsave('./output/%d/label_1_%d.TIF' % (i, k), label_output1)
                #         imsave('./output/%d/label_2_%d.TIF' % (i, k), label_output2)
                #         imsave('./output/%d/pan_%d.TIF' % (i, k), pan)
                #         imsave('./output/%d/ms_%d.TIF' % (i, k), ms1)

                Validation_Loss += lossvalue_

            Validation_Loss = Validation_Loss / data_size_

            print ('%d epoch is finished, learning rate = %.8f, Training_Loss = %.8f, Validation_Loss = %.8f' %
                   (j + 1, lr_, Training_Loss, Validation_Loss))

