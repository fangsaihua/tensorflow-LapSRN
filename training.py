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
from PS import PS

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

def cosine(predict, label):
    with tf.variable_scope("cosLoss"):
        product = tf.multiply(predict, label)
        numerator = tf.reduce_sum(product, 3)

        norm1 = tf.norm(predict, axis=3)
        norm2 = tf.norm(label, axis=3)
        denominator = tf.multiply(norm1, norm2)

        c = tf.div(numerator, denominator + 1e-10)
        return c

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
def guided_filter(data, num_patches = FLAGS.num_patches, width = FLAGS.image_size, height = FLAGS.image_size, channel = FLAGS.num_channels):
    r = 15
    eps = 1.0
    batch_q = np.zeros((num_patches, height, width, channel))
    for i in range(num_patches):
        for j in range(channel):
            I = data[i, :, :,j]
            p = data[i, :, :,j]
            ones_array = np.ones([height, width])
            N = cv2.boxFilter(ones_array, -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType = 0)
            mean_I = cv2.boxFilter(I, -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType = 0) / N
            mean_p = cv2.boxFilter(p, -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType = 0) / N
            mean_Ip = cv2.boxFilter(I * p, -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType = 0) / N
            cov_Ip = mean_Ip - mean_I * mean_p
            mean_II = cv2.boxFilter(I * I, -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType = 0) / N
            var_I = mean_II - mean_I * mean_I
            a = cov_Ip / (var_I + eps) 
            b = mean_p - a * mean_I
            mean_a = cv2.boxFilter(a , -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType = 0) / N
            mean_b = cv2.boxFilter(b , -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType = 0) / N
            q = mean_a * I + mean_b 
            batch_q[i, :, :,j] = q 
    return batch_q

def blur_image(data, w):
    num_patches, height, width, channel = data.shape
    batch_q = np.zeros((num_patches, height, width, channel))
    for i in range(num_patches):
        for j in range(channel):
            img = data[i, :, :, j]
            img = cv2.blur(img, w)
            batch_q[i, :, :, j] = img

    return batch_q

# initialize weights
def create_kernel(name, shape, initializer=tf.contrib.layers.xavier_initializer()):
    regularizer = tf.contrib.layers.l2_regularizer(scale = 1e-5)

    new_variables = tf.get_variable(name=name, shape=shape, initializer=initializer,
                                    regularizer=regularizer, trainable=True)
    return new_variables

def convt_F(detail, level):
    with tf.variable_scope(level):
        with tf.variable_scope('conv_1'):
            kernel = create_kernel(name='weights_1', shape=[3, 3, FLAGS.num_channels+1, 32])
            biases = tf.Variable(tf.constant(0.0, shape=[32], dtype=tf.float32), trainable=True, name='biases_1')

            conv = tf.nn.conv2d(detail, kernel, [1, 1, 1, 1], padding='SAME')
            feature = tf.nn.bias_add(conv, biases)

            conv_shortcut = tf.nn.relu(feature)

        for i in range(3):
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

                # feature_relu = tf.nn.relu(feature)

                conv_shortcut = tf.add(conv_shortcut, feature)  # shortcut

        with tf.variable_scope('conv_8'):
            kernel = create_kernel(name='weights_8', shape=[3, 3, 32, FLAGS.num_channels])
            biases = tf.Variable(tf.constant(0.0, shape=[FLAGS.num_channels], dtype=tf.float32), trainable=True,
                                 name='biases_8')

            conv = tf.nn.conv2d(conv_shortcut, kernel, [1, 1, 1, 1], padding='SAME')
            feature = tf.nn.bias_add(conv, biases)

            # feature_relu = tf.nn.relu(feature)

    return feature

def convt_I2(image, level):
    with tf.variable_scope(level):
        kernel = create_kernel(name='weights_1', shape=[3, 3, FLAGS.num_channels, 16])
        biases = tf.Variable(tf.constant(0.0, shape=[16], dtype=tf.float32), trainable=True, name='biases_1')

        conv = tf.nn.atrous_conv2d(image, kernel, 2, padding='SAME')
        feature = tf.nn.bias_add(conv, biases)
        out = tf.nn.relu(feature)

        kernel2 = create_kernel(name='weights_2', shape=[3, 3, 16, 16])
        biases2 = tf.Variable(tf.constant(0.0, shape=[16], dtype=tf.float32), trainable=True,
                              name='biases_2')

        conv2 = tf.nn.atrous_conv2d(out, kernel2, 3, padding='SAME')
        feature = tf.nn.bias_add(conv2, biases2)
        out = tf.nn.relu(feature)

        kernel3 = create_kernel(name='weights_3', shape=[3, 3, 16, FLAGS.num_channels])
        biases3 = tf.Variable(tf.constant(0.0, shape=[FLAGS.num_channels], dtype=tf.float32), trainable=True,
                              name='biases_3')

        conv3 = tf.nn.atrous_conv2d(out, kernel3, 3, padding='SAME')
        feature = tf.nn.bias_add(conv3, biases3)
        out = tf.nn.relu(feature)

        out = tf.layers.conv2d_transpose(out, FLAGS.num_channels, 3, strides=2, padding='SAME') ###change up-sample scale

    return out


def convt_I(image, level):
    with tf.variable_scope(level):
        with tf.variable_scope('conv_1'):
            kernel = create_kernel(name='weights_1', shape=[3, 3, FLAGS.num_channels, 32])
            biases = tf.Variable(tf.constant(0.0, shape=[32], dtype=tf.float32), trainable=True, name='biases_1')

            conv = tf.nn.conv2d(image, kernel, [1, 1, 1, 1], padding='SAME')
            feature = tf.nn.bias_add(conv, biases)
            feature_relu = tf.nn.relu(feature)

        with tf.variable_scope('conv_2'):
            kernel2 = create_kernel(name='weights_2', shape=[3, 3, 32, 32])
            biases2 = tf.Variable(tf.constant(0.0, shape=[32], dtype=tf.float32), trainable=True, name='biases_2')

            conv2 = tf.nn.conv2d(feature_relu, kernel2, [1, 1, 1, 1], padding='SAME')
            feature2 = tf.nn.bias_add(conv2, biases2)
            feature_relu2 = tf.nn.relu(feature2)

        with tf.variable_scope('conv_3'):
            kernel3 = create_kernel(name='weights_3', shape=[3, 3, 32, 32])
            biases3 = tf.Variable(tf.constant(0.0, shape=[32], dtype=tf.float32), trainable=True, name='biases_3')

            conv3 = tf.nn.conv2d(feature_relu2, kernel3, [1, 1, 1, 1], padding='SAME')
            feature3 = tf.nn.bias_add(conv3, biases3)
            feature_relu3 = tf.nn.relu(feature3)

        with tf.variable_scope('conv_4'):
            kernel4 = create_kernel(name='weights_4', shape=[3, 3, 32, 16])
            biases4 = tf.Variable(tf.constant(0.0, shape=[16], dtype=tf.float32), trainable=True, name='biases_4')

            conv4 = tf.nn.conv2d(feature_relu3, kernel4, [1, 1, 1, 1], padding='SAME')
            feature4 = tf.nn.bias_add(conv4, biases4)
            feature_relu4 = tf.nn.relu(feature4)

        out = PS(feature_relu4, 2, color=True)
    return out

def convt_I4(image, level):
    with tf.variable_scope(level):
        out = tf.layers.conv2d_transpose(image, FLAGS.num_channels, 3, strides=2, padding='SAME') ###change up-sample scale
    return out

# network structure
def inference(ms, detail_ms, detail_pan_2x, detail_pan_4x):
    ms_2x_base = convt_I(ms, 'M1')
    detail_ms_2x = convt_I4(detail_ms, 'D1')
    detail_2x = convt_F(tf.concat([detail_ms_2x, detail_pan_2x], 3), 'F1')
    detail_2x_summ = tf.summary.image('detail_2x', detail_2x, max_outputs=4)
    ms_2x = tf.add(detail_2x, ms_2x_base)

    ms_4x_base = convt_I(ms_2x, 'M2')
    detail_ms_4x = convt_I4(detail_2x, 'D2')
    detail_4x = convt_F(tf.concat([detail_ms_4x, detail_pan_4x], 3), 'F2')
    detail_4x_summ = tf.summary.image('detail_4x', detail_4x, max_outputs=4)
    ms_4x = tf.add(detail_4x, ms_4x_base)

    return ms_2x, ms_4x, ms_2x_base, detail_2x, ms_4x_base, detail_4x


if __name__ == '__main__':

  ms = tf.placeholder(tf.float32, shape=(None, FLAGS.image_size / 4, FLAGS.image_size / 4, FLAGS.num_channels))
  detail_ms = tf.placeholder(tf.float32, shape=(None, FLAGS.image_size / 4, FLAGS.image_size / 4, FLAGS.num_channels))
  label_2x = tf.placeholder(tf.float32, shape=(None, FLAGS.image_size / 2, FLAGS.image_size / 2, FLAGS.num_channels))
  detail_pan_2x = tf.placeholder(tf.float32, shape=(None, FLAGS.image_size / 2, FLAGS.image_size / 2, 1))
  label_4x = tf.placeholder(tf.float32, shape=(None, FLAGS.image_size, FLAGS.image_size, FLAGS.num_channels))
  detail_pan_4x = tf.placeholder(tf.float32, shape=(None, FLAGS.image_size, FLAGS.image_size, 1))
  detail_2x_p = tf.placeholder(tf.float32, shape = (None, FLAGS.image_size / 2, FLAGS.image_size / 2, 4))
  ms_2x_base_p = tf.placeholder(tf.float32, shape = (None, FLAGS.image_size / 2, FLAGS.image_size / 2, 4))
  detail_4x_p = tf.placeholder(tf.float32, shape = (None, FLAGS.image_size, FLAGS.image_size, 4))
  ms_4x_base_p = tf.placeholder(tf.float32, shape = (None, FLAGS.image_size, FLAGS.image_size, 4))


  ms_2x, ms_4x, ms_2x_base, detail_2x, ms_4x_base, detail_4x = inference(ms, detail_ms, detail_pan_2x, detail_pan_4x)

  # loss2 = tf.reduce_mean(tf.square(label_4x - ms_4x)) - 0.1 * tf.reduce_mean(tf.log(tf.abs(cosine(label_4x, ms_4x))+1e-10))

  # loss = tf.reduce_mean(tf.square(label_2x - ms_2x)) - 0.1 * tf.reduce_mean(tf.log(tf.abs(cosine(ms_2x, label_2x))+1e-10)) + loss2

  # loss2 = tf.reduce_mean(tf.square(label_4x - ms_4x)) + tf.reduce_mean(tf.square(5*(cosine(label_4x, ms_4x) - 1)))
  loss2 = tf.reduce_mean(tf.square(label_4x - ms_4x)) + 0.001 * tf.reduce_mean(tf.square(ms_4x_base_p - ms_4x_base)) + 0.01 * tf.reduce_mean(tf.square(detail_4x_p - detail_4x))
  loss3 = tf.reduce_mean(tf.square(label_2x - ms_2x)) + 0.001 * tf.reduce_mean(tf.square(ms_2x_base_p - ms_2x_base)) + 0.01 * tf.reduce_mean(tf.square(detail_2x_p - detail_2x))

  # loss = tf.reduce_mean(tf.square(label_2x - ms_2x)) + tf.reduce_mean(tf.square(5*(cosine(label_2x, ms_2x) - 1))) + loss2
  loss = 2 * loss3 + loss2

  # loss = tf.reduce_mean(tf.square(label_2x - ms_2x)) + loss2
  loss_sum_summ = tf.summary.scalar('loss', loss2)
  lr_ = FLAGS.learning_rate
  lr  = tf.placeholder(tf.float32 ,shape = []) 
  g_optim =  tf.train.AdamOptimizer(lr).minimize(loss) # Optimization method: Adam
  # gradients, variables_ = zip(*g_optim.compute_gradients(loss))
  # gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
  # g_optim = g_optim.apply_gradients(zip(gradients, variables_))

  saver = tf.train.Saver(max_to_keep = 5)
  
  config = tf.ConfigProto()
  config.gpu_options.per_process_gpu_memory_fraction = 0.8 # GPU setting
  config.gpu_options.allow_growth = True
  
  
  data_path = FLAGS.data_path
  save_path = FLAGS.save_model_path 
  epoch = int(FLAGS.epoch) 

  with tf.Session(config=config) as sess:
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('logs/a/', sess.graph)

    sess.run(tf.global_variables_initializer())
    

    validation_data_name = "validation.h5"
    validation_ms, validation_pan_2x, validation_pan_4x, validation_label_2x, validation_label_4x = read_data(data_path + validation_data_name)

    validation_ms = np.transpose(validation_ms, (0,2,3,1))   # image data
    validation_pan_2x = np.transpose(validation_pan_2x, (0,2,3,1))   # image data
    validation_pan_4x = np.transpose(validation_pan_4x, (0,2,3,1))   # image data
    validation_label_2x = np.transpose(validation_label_2x, (0,2,3,1))   # image data
    validation_label_4x = np.transpose(validation_label_4x, (0,2,3,1))   # image data

    validation_detail_ms = validation_ms - blur_image(validation_ms, (5, 5))  # detail layer
    validation_detail_pan_2x = validation_pan_2x - blur_image(validation_pan_2x, (11, 11))  # detail layer
    validation_detail_pan_4x = validation_pan_4x - blur_image(validation_pan_4x, (21, 21))  # detail layer
    validation_ms_2x_base = blur_image(validation_label_2x, (11, 11))
    validation_detail_2x = validation_label_2x - validation_ms_2x_base
    validation_ms_4x_base = blur_image(validation_label_4x, (21, 21))
    validation_detail_4x = validation_label_4x - validation_ms_4x_base

    if tf.train.get_checkpoint_state('./model/'):   # load previous trained model
      ckpt = tf.train.latest_checkpoint('./model/')
      saver.restore(sess, ckpt)
      ckpt_num = re.findall(r"\d",ckpt)
      if len(ckpt_num)==3:
        start_point = 100*int(ckpt_num[0])+10*int(ckpt_num[1])+int(ckpt_num[2])
      elif len(ckpt_num)==2:
        start_point = 10*int(ckpt_num[0])+int(ckpt_num[1])
      else:
        start_point = int(ckpt_num[0])
      print("Load success")
      print(start_point)
   
    else:
      print("re-training")
      start_point = 0    


    for j in range(start_point,epoch):   # epoch

      if j+1 >(epoch/3):  # reduce learning rate
        lr_ = FLAGS.learning_rate*0.1
      if j+1 >(2*epoch/3):
        lr_ = FLAGS.learning_rate*0.01

      Training_Loss = 0.
  
      for num in range(FLAGS.num_h5_file):    # h5 files
        train_data_name = "train" + str(num+1) + ".h5"
        train_ms, train_pan_2x, train_pan_4x, train_label_2x, train_label_4x = read_data(data_path + train_data_name)

        train_ms = np.transpose(train_ms, (0,2,3,1))   # image data
        train_pan_2x = np.transpose(train_pan_2x, (0,2,3,1))   # image data
        train_pan_4x = np.transpose(train_pan_4x, (0,2,3,1))   # image data
        train_label_2x = np.transpose(train_label_2x, (0,2,3,1))   # image data
        train_label_4x = np.transpose(train_label_4x, (0,2,3,1))   # image data

        train_detail_ms = train_ms - blur_image(train_ms, (5, 5))  # detail layer
        train_detail_pan_2x = train_pan_2x - blur_image(train_pan_2x, (11, 11))  # detail layer
        train_detail_pan_4x = train_pan_4x - blur_image(train_pan_4x, (21, 21))  # detail layer
        train_ms_2x_base = blur_image(train_label_2x, (11, 11))
        train_detail_2x = train_label_2x - train_ms_2x_base
        train_ms_4x_base = blur_image(train_label_4x, (21, 21))
        train_detail_4x = train_label_4x - train_ms_4x_base


        data_size = int( FLAGS.num_patches / FLAGS.batch_size )  # the number of batch
        for i in range(data_size):
          rand_index = np.arange(int(i*FLAGS.batch_size),int((i+1)*FLAGS.batch_size))   # batch
          batch_ms = train_ms[rand_index,:,:,:]
          batch_detail_ms = train_detail_ms[rand_index,:,:,:]
          batch_label_2x = train_label_2x[rand_index,:,:,:]
          batch_label_4x = train_label_4x[rand_index,:,:,:]
          batch_detail_pan_2x = train_detail_pan_2x[rand_index,:,:,:]
          batch_detail_pan_4x = train_detail_pan_4x[rand_index,:,:,:]
          batch_ms_2x_base = train_ms_2x_base[rand_index,:,:,:]
          batch_detail_2x = train_detail_2x[rand_index,:,:,:]
          batch_ms_4x_base = train_ms_4x_base[rand_index,:,:,:]
          batch_detail_4x = train_detail_4x[rand_index,:,:,:]

          _,lossvalue,summary_str = sess.run([g_optim,loss,merged], feed_dict={ms: batch_ms, detail_ms: batch_detail_ms, label_2x: batch_label_2x, label_4x: batch_label_4x,
                                                            detail_pan_2x: batch_detail_pan_2x, detail_pan_4x: batch_detail_pan_4x, detail_2x_p: batch_detail_2x,
                                                                               ms_2x_base_p: batch_ms_2x_base, detail_4x_p: batch_detail_4x, ms_4x_base_p: batch_ms_4x_base, lr: lr_})
          writer.add_summary(summary_str, j*FLAGS.num_h5_file*data_size + num*data_size + i)
          writer.flush()

          Training_Loss += lossvalue  # training loss
  

      Training_Loss /=  (data_size * FLAGS.num_h5_file)

      model_name = 'model-epoch'   # save model
      save_path_full = os.path.join(save_path, model_name)
      saver.save(sess, save_path_full, global_step = j+1)

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
          batch_ms_2x_base = validation_ms_2x_base[rand_index_, :, :, :]
          batch_detail_2x = validation_detail_2x[rand_index_, :, :, :]
          batch_ms_4x_base = validation_ms_4x_base[rand_index_, :, :, :]
          batch_detail_4x = validation_detail_4x[rand_index_, :, :, :]
          output1, output2, lossvalue_  = sess.run([ms_2x, ms_4x, loss],  feed_dict={ms: batch_ms_, detail_ms: batch_detail_ms_, label_2x: batch_label_2x_, label_4x: batch_label_4x_,
                                                            detail_pan_2x: batch_detail_pan_2x_, detail_pan_4x: batch_detail_pan_4x_,
                                                                                     detail_2x_p: batch_detail_2x,
                                                                                     ms_2x_base_p: batch_ms_2x_base,
                                                                                     detail_4x_p: batch_detail_4x,
                                                                                     ms_4x_base_p: batch_ms_4x_base,lr: lr_})  # validation loss

          if j == epoch - 1:
              output1[np.where(output1 < 0.)] = 0
              output2[np.where(output2 > 1.)] = 1
              os.makedirs('./output/' + str(i))

              for k in range(FLAGS.batch_size):
                  final_output1 = output1[k, :, :, :]
                  final_output2 = output2[k, :, :, :]
                  label_output1 = batch_label_2x_[k, :, :, :]
                  label_output2 = batch_label_4x_[k, :, :, :]
                  pan = batch_pan_4x[k, :, :, :]
                  ms1 = batch_ms_[k, :, :, :]
                  imsave('./output/%d/out_1_%d.TIF' % (i, k), final_output1)
                  imsave('./output/%d/out_2_%d.TIF' % (i, k), final_output2)
                  imsave('./output/%d/label_1_%d.TIF' % (i, k), label_output1)
                  imsave('./output/%d/label_2_%d.TIF' % (i, k), label_output2)
                  imsave('./output/%d/pan_%d.TIF' % (i, k), pan)
                  imsave('./output/%d/ms_%d.TIF' % (i, k), ms1)

          Validation_Loss += lossvalue_

      Validation_Loss = Validation_Loss / data_size_

      print ('%d epoch is finished, learning rate = %.8f, Training_Loss = %.8f, Validation_Loss = %.8f' %
               (j+1, lr_, Training_Loss, Validation_Loss))
  
