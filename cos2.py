import tensorflow as tf
import random
import numpy as np

def cosine(predict, label):
    with tf.variable_scope("cosLoss"):
        product = tf.multiply(predict, label)
        numerator = tf.reduce_sum(product, 2)

        norm1 = tf.norm(predict, axis=2)
        norm2 = tf.norm(label, axis=2)
        denominator = tf.multiply(norm1, norm2)

        c = tf.div(numerator, denominator + 1e-7)
        return c

a = np.array([[[1.,2.,3.,4.,5.,6.],[1.,2.,3.,4.,5.,6.],[1.,2.,3.,4.,5.,6.]],
              [[1., 2., 3., 4., 5.,6.], [1., 2., 3., 4., 5.,6.], [1., 2., 3., 4., 5.,6.]],
              [[1., 2., 3., 4., 5.,6.], [1., 2., 3., 4., 5.,6.], [1., 2., 3., 4., 5.,6.]]])
b = np.array([[[3.,3.,3.,3.,3.,3.],[3.,3.,3.,3.,3.,3.],[3.,3.,3.,3.,3.,3.]],
              [[3., 3., 3., 3., 3.,3.], [3., 3., 3., 3., 3.,3.], [3., 3., 3., 3., 3.,3.]],
              [[3., 3., 3., 3., 3.,3.], [3., 3., 3., 3., 3.,3.], [3., 3., 3., 3., 3.,3.]]])
print b

# p1 = tf.placeholder(tf.float32, shape=(1,3))
p3 = tf.Variable(a, trainable=True)
p2 = tf.constant(b)

x = 1
# loss1 = -tf.reduce_mean(tf.log(cosine(p3, p2)+1e-10))
loss1 = tf.reduce_mean(tf.square(1*(cosine(p3, p2)-1)))
# loss2 = tf.reduce_mean((tf.square(p3 - p2)));
loss2 = tf.reduce_mean(tf.reduce_sum((tf.square(p3 - p2)),2)) / 6;
c =  loss2 + x * loss1

g_optim = tf.train.GradientDescentOptimizer(1).minimize(c)  # Optimization method: Adam

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(1000000):
        _, c_, loss1_, loss2_, p = sess.run([g_optim, c, loss1, loss2, p3]);
        print(p)
        print c_, loss1_, loss2_
        print('------------------------')
        # print c_