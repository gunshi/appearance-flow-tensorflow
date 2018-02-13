import numpy as np
import tensorflow as tf
from bilinear_sampler import bilinear_sampler
from tf.contrib.resampler import resampler


# code to test bilinear_sampler.py

x = tf.random_uniform([1, 5, 5, 3])
v = tf.ones([1, 5, 5, 2]) * 2.
print('v')

#y = bilinear_sampler(x, v, resize=True, crop=(0,4,0,4))
z = bilinear_sampler(x, v)

with tf.Session() as sess:

	x_,z_= sess.run([x,z])
  	z2 = resampler(x_,v)

  	#add coords
  	z3 = resampler(x_,)

  

  	print(x_)

  	print(z_)

