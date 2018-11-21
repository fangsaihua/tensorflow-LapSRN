import numpy as np
import tensorflow as tf

def PS(I, r):
    print(I.shape)
    O = np.zeros([I.shape[0]*r, I.shape[1]*r, I.shape[2]/(r*2)])
    for x in range(O.shape[0]):
        for y in range(O.shape[1]):
            for c in range(O.shape[2]):
                c += 1
                a = np.floor(x/r).astype("int")
                b = np.floor(y/r).astype("int")
                d = c*r*(y%r) + c*(x%r)
                print x, y, c-1, "=>", a, b, d
                O[x, y, c-1] = I[a, b, d]
    return O


x = np.arange(16 * 16).reshape(8, 8, 4)
O = PS(x, 2)

with tf.Session() as sess:
    pass