import numpy as np
import tensorflow as tf
from skimage import data, img_as_float
from ssim import *

image = data.camera()
img = img_as_float(image)
rows, cols = img.shape

noise = np.ones_like(img) * 0.2 * (img.max() - img.min())
noise[np.random.random(size=noise.shape) > 0.5] *= -1

img_noise = img + noise

## TF CALC START
BATCH_SIZE = 1
CHANNELS = 1
image1 = tf.placeholder(tf.float32, shape=[rows, cols])
image2 = tf.placeholder(tf.float32, shape=[rows, cols])

def image_to_4d(image):
    image = tf.expand_dims(image, 0)
    image = tf.expand_dims(image, -1)
    return image

image4d_1 = image_to_4d(image1)
image4d_2 = image_to_4d(image2)

ssim_index = tf_ssim(image4d_1, image4d_2)

msssim_index = tf_ms_ssim(image4d_1, image4d_2)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    tf_ssim_none = sess.run(ssim_index,
                            feed_dict={image1: img, image2: img})
    tf_ssim_noise = sess.run(ssim_index,
                             feed_dict={image1: img, image2: img_noise})

    tf_msssim_none = sess.run(msssim_index,
                            feed_dict={image1: img, image2: img})
    tf_msssim_noise = sess.run(msssim_index,
                             feed_dict={image1: img, image2: img_noise})
###TF CALC END

print('tf_ssim_none', tf_ssim_none)
print('tf_ssim_noise', tf_ssim_noise)
print('tf_msssim_none', tf_msssim_none)
print('tf_msssim_noise', tf_msssim_noise)