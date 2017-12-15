import numpy as np
import tensorflow as tf
from bilinear_sampler import bilinear_sampler


# code to test bilinear_sampler.py

x = tf.random_uniform([1, 5, 5, 3])
v = tf.ones([1, 5, 5, 2]) * 2.
print('v')

#y = bilinear_sampler(x, v, resize=True, crop=(0,4,0,4))
z = bilinear_sampler(x, v)

with tf.Session() as sess:

  x_,z_= sess.run([x,z])
  print(x_)

  print(z_)

