import numpy as np
import tensorflow as tf
from bilinear_sampler import bilinear_sampler
from tensorflow.contrib.resampler import resampler

def _get_grid_array(N, H, W, h, w):
    N_i = tf.range(N)
    H_i = tf.range(h+1, h+H+1)
    W_i = tf.range(w+1, w+W+1)
    n, h, w, = tf.meshgrid(N_i, H_i, W_i, indexing='ij')
    n = tf.expand_dims(n, axis=3) # [N, H, W, 1]
    h = tf.expand_dims(h, axis=3) # [N, H, W, 1]
    w = tf.expand_dims(w, axis=3) # [N, H, W, 1]
    n = tf.cast(n, tf.float32) # [N, H, W, 1]
    h = tf.cast(h, tf.float32) # [N, H, W, 1]
    w = tf.cast(w, tf.float32) # [N, H, W, 1]

    return n, h, w
# code to test bilinear_sampler.py

x = tf.random_uniform([1, 5, 5, 3])
v = tf.ones([1, 5, 5, 2]) 
v2 = tf.ones([1, 5, 5, 2]) *2.

print('v')

#y = bilinear_sampler(x, v, resize=True, crop=(0,4,0,4))
z = bilinear_sampler(x, v2)

shape = tf.shape(x)
N = shape[0]
H_ = H = shape[1]
W_ = W = shape[2]
h = w = 0
n, h, w = _get_grid_array(N, H, W, h, w) # [N, H, W, 3]
stacked =  tf.stack([h,w],3)
#stacked =  tf.expand_dims(stacked, axis=0) # [N, H, W, 1]
#stacked = stacked+v

stacked = tf.squeeze(stacked, [4])  # [1, 2, 3, 1]
stacked = stacked+v
z2 = resampler(x,stacked)
z2 = tf.transpose(z2,[0,2,1,3])

with tf.Session() as sess:


  	#add coords
  	#z3 = resampler(x_,)



	x_,z_,n,h,w,stacked,z2 = sess.run([x,z,n,h,w,stacked, z2])

	#h+=2
	#w+=2

	print(n)
	print('............')
	print(h)
	print('............')
	print(w)
	print('............')
	print(z_)
	print('............')
	print(x_)
	print('............')
	print(stacked)
	print(stacked.shape)
	print('............')
	print(z2)
  

  	#print(x_)

  	#print(z_)



