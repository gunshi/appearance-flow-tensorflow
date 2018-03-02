from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from scipy import misc
from bilinear_sampler import bilinear_sampler
import math
#from tf.contrib.resampler import resampler


class Net_MultiView(object):
    def initalize(self, sess):
        pre_trained_weights = np.load(open(self.weight_path, "rb"), encoding="latin1").item()
        keys = sorted(pre_trained_weights.keys())
        #for k in keys:
        for k in list(filter(lambda x: 'conv' in x,keys)):
            with tf.variable_scope(k, reuse=True):
                temp = tf.get_variable('weights')
                sess.run(temp.assign(pre_trained_weights[k]['weights']))
            with tf.variable_scope(k, reuse=True):
                temp = tf.get_variable('biases')
                sess.run(temp.assign(pre_trained_weights[k]['biases']))

    def conv(self, input_, filter_size, in_channels, out_channels, name, strides, padding, groups, pad_input=1):
        if pad_input==1:
            paddings = tf.constant([ [0, 0], [1, 1,], [1, 1], [0, 0] ])
            input_ = tf.pad(input_, paddings, "CONSTANT")

        with tf.variable_scope(name) as scope:
            filt = tf.get_variable('weights', shape=[filter_size, filter_size, int(in_channels/groups), out_channels], trainable=self.trainable)
            bias = tf.get_variable('biases',  shape=[out_channels], trainable=self.trainable)
        if groups == 1:
            return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(input_, filt, strides=strides, padding=padding), bias))
        else:
            # Split input_ and weights and convolve them separately
            input_groups = tf.split(axis = 3, num_or_size_splits=groups, value=input_)
            filt_groups = tf.split(axis = 3, num_or_size_splits=groups, value=filt)
            output_groups = [ tf.nn.conv2d( i, k, strides = strides, padding = padding) for i,k in zip(input_groups, filt_groups)]

            conv = tf.concat(axis = 3, values = output_groups)
            return tf.nn.relu(tf.nn.bias_add(conv, bias))


    def fc(self, input_, in_channels, out_channels, name, relu):
        input_ = tf.reshape(input_ , [-1, in_channels])
        with tf.variable_scope(name) as scope:
            filt = tf.get_variable('weights', shape=[in_channels , out_channels], trainable=self.trainable)
            bias = tf.get_variable('biases',  shape=[out_channels], trainable=self.trainable)
        if relu:
            return tf.nn.relu(tf.nn.bias_add(tf.matmul(input_, filt), bias))
        else:
            return tf.nn.bias_add(tf.matmul(input_, filt), bias)


    def pool(self, input_, padding, name):
        return tf.nn.max_pool(input_, ksize=[1,3,3,1], strides=[1,2,2,1], padding=padding, name= name)



    def model(self):

        debug=True
        net_layers={}
        #placeholder for a random set of <batch_size> images of fixed size -- 224,224
        self.input_imgs = tf.placeholder(tf.float32, shape = [None, 224, 224, 3], name = "input_imgs")
        self.aux_imgs = tf.placeholder(tf.float32, shape = [None, 224, 224, 3], name = "input_imgs")

        self.input_batch_size = tf.shape(self.input_imgs)[0]  # Returns a scalar `tf.Tensor`
        self.tform = tf.placeholder(tf.float32, shape = [None, 224, 224, 6], name = "tform")
        self.tform_aux = tf.placeholder(tf.float32, shape = [None, 224, 224, 6], name = "tform")

        net_layers['input_stack'] = tf.concat([self.input_imgs, self.tform], 3)
        net_layers['input_stack_aux'] = tf.concat([self.aux_imgs, self.tform_aux], 3)

        #mean is already subtracted in helper.py as part of preprocessing
        # Conv-Layers
        net_layers['Convolution0'] = self.conv(net_layers['input_stack'], 3, 9 , 16, name= 'Convolution0', strides=[1,2,2,1] ,padding='VALID', groups=1,pad_input=1)
        net_layers['Convolution1'] = self.conv(net_layers['Convolution0'], 3, 16 , 32, name= 'Convolution1', strides=[1,2,2,1] ,padding='VALID', groups=1,pad_input=1)
        net_layers['Convolution2'] = self.conv(net_layers['Convolution1'], 3, 32 , 64, name= 'Convolution2', strides=[1,2,2,1] ,padding='VALID', groups=1,pad_input=1)
        net_layers['Convolution3'] = self.conv(net_layers['Convolution2'], 3, 64 , 128, name= 'Convolution3', strides=[1,2,2,1] ,padding='VALID', groups=1,pad_input=1)
        net_layers['Convolution4'] = self.conv(net_layers['Convolution3'], 3, 128 , 256, name= 'Convolution4', strides=[1,2,2,1] ,padding='VALID', groups=1,pad_input=1)
        net_layers['Convolution5'] = self.conv(net_layers['Convolution4'], 3, 256 , 512, name= 'Convolution5', strides=[1,2,2,1] ,padding='VALID', groups=1,pad_input=1)

        #deconv
        net_layers['deconv1'] = self._upscore_layer(net_layers['Convolution5'], shape=None,
                                           num_classes=512,
                                           debug=debug, name='deconv1', ksize=3, stride=2, pad_input=1)

        net_layers['deconv2'] = self._upscore_layer(net_layers['deconv1'], shape=None,
                                           num_classes=256,
                                           debug=debug, name='deconv2', ksize=3, stride=2, pad_input=1)

        net_layers['deconv3'] = self._upscore_layer(net_layers['deconv2'], shape=None,
                                           num_classes=128,
                                           debug=debug, name='deconv3', ksize=3, stride=2, pad_input=1)

        net_layers['deconv4'] = self._upscore_layer(net_layers['deconv3'], shape=None,
                                           num_classes=64,
                                           debug=debug, name='deconv4', ksize=3, stride=2, pad_input=1)
        net_layers['deconv5'] = self._upscore_layer(net_layers['deconv4'], shape=None,
                                           num_classes=32,
                                           debug=debug, name='deconv5', ksize=3, stride=2, pad_input=1)
        net_layers['deconv6'] = self._upscore_layer(net_layers['deconv5'], shape=None,
                                           num_classes=3,
                                           debug=debug, name='deconv6', ksize=3, stride=1, pad_input=1)

        net_layers['rs'] = tf.resize_bilinear(net_layers['deconv6'],(224.224))

        net_layers['flow'] = tf.slice( net_layers['rs'], [0,0,0,0] , [self.batch_size,224,224,2] )
        net_layers['conf'] = tf.slice( net_layers['rs'], [0,0,0,2] , [self.batch_size,224,224,1] )

        net_layers['predImg_single']=bilinear_sampler(self.input_imgs,net_layers['flow'], resize=True)
        
        #alternatively add coords
        #resampler(self.input_imgs,net_layers['flow_aux'],name='resampler')


        net_layers['Convolution0_aux'] = self.conv(net_layers['input_stack_aux'], 3, 9 , 16, name= 'Convolution0_aux', strides=[1,2,2,1] ,padding='VALID', groups=1,pad_input=1)
        net_layers['Convolution1_aux'] = self.conv(net_layers['Convolution0_aux'], 3, 16 , 32, name= 'Convolution1_aux', strides=[1,2,2,1] ,padding='VALID', groups=1,pad_input=1)
        net_layers['Convolution2_aux'] = self.conv(net_layers['Convolution1_aux'], 3, 32 , 64, name= 'Convolution2_aux', strides=[1,2,2,1] ,padding='VALID', groups=1,pad_input=1)
        net_layers['Convolution3_aux'] = self.conv(net_layers['Convolution2_aux'], 3, 64 , 128, name= 'Convolution3_aux', strides=[1,2,2,1] ,padding='VALID', groups=1,pad_input=1)
        net_layers['Convolution4_aux'] = self.conv(net_layers['Convolution3_aux'], 3, 128 , 256, name= 'Convolution4_aux', strides=[1,2,2,1] ,padding='VALID', groups=1,pad_input=1)
        net_layers['Convolution5_aux'] = self.conv(net_layers['Convolution4_aux'], 3, 256 , 512, name= 'Convolution5_aux', strides=[1,2,2,1] ,padding='VALID', groups=1,pad_input=1)

        #deconv
        net_layers['deconv1_aux'] = self._upscore_layer(net_layers['Convolution5_aux'], shape=None,
                                           num_classes=512,
                                           debug=debug, name='deconv1_aux', ksize=3, stride=2, pad_input=1)

        net_layers['deconv2_aux'] = self._upscore_layer(net_layers['deconv1_aux'], shape=None,
                                           num_classes=256,
                                           debug=debug, name='deconv2_aux', ksize=3, stride=2, pad_input=1)

        net_layers['deconv3_aux'] = self._upscore_layer(net_layers['deconv2_aux'], shape=None,
                                           num_classes=128,
                                           debug=debug, name='deconv3_aux', ksize=3, stride=2, pad_input=1)

        net_layers['deconv4_aux'] = self._upscore_layer(net_layers['deconv3_aux'], shape=None,
                                           num_classes=64,
                                           debug=debug, name='deconv4_aux', ksize=3, stride=2, pad_input=1)
        net_layers['deconv5_aux'] = self._upscore_layer(net_layers['deconv4_aux'], shape=None,
                                           num_classes=32,
                                           debug=debug, name='deconv5_aux', ksize=3, stride=2, pad_input=1)
        net_layers['deconv6_aux'] = self._upscore_layer(net_layers['deconv5_aux'], shape=None,
                                           num_classes=3,
                                           debug=debug, name='deconv6_aux', ksize=3, stride=1, pad_input=1)

        net_layers['rs_aux'] = tf.resize_bilinear(net_layers['deconv6_aux'],(224,224))

        net_layers['flow_aux'] = tf.slice( net_layers['rs_aux'], [0,0,0,0] , [self.batch_size,224,224,2] )
        net_layers['conf_aux'] = tf.slice( net_layers['rs_aux'], [0,0,0,2] , [self.batch_size,224,224,1] )
        net_layers['predImg_single_aux'] = bilinear_sampler(self.input_imgs,net_layers['flow_aux'], resize=True)


        net_layers['concat_conf'] = tf.concat([net_layers['conf'], net_layers['conf_aux']], 3)
        net_layers['concat_soft'] = tf.nn.softmax(net_layers['concat_conf'])
        net_layers['srcSelect'] = tf.slice(net_layers['concat_soft'], [0,0,0,0] , [self.batch_size,224,224,1])
        net_layers['auxSelect'] = tf.slice(net_layers['concat_soft'], [0,0,0,1] , [self.batch_size,224,224,1])
        net_layers['concat_pixelwise_src'] = tf.concat([net_layers['srcSelect'], net_layers['srcSelect'], net_layers['srcSelect']], 3)
        net_layers['concat_pixelwise_aux'] = tf.concat([net_layers['auxSelect'], net_layers['auxSelect'], net_layers['auxSelect']], 3)
        net_layers['srcSelectImg'] = tf.multiply(net_layers['predImg_single'],net_layers['concat_pixelwise_src'])
        net_layers['auxSelectImg'] = tf.multiply(net_layers['predImg_single_aux'], net_layers['concat_pixelwise_aux'])
        
        net_layers['predImg'] = tf.add(net_layers['srcSelectImg'], net_layers['auxSelectImg'], name='predImg')


        self.net_layers = net_layers



    def _upscore_layer(self, bottom, shape,num_classes, name, debug, ksize=3, stride=2, pad_input=1, relu=1):

        strides = [1, stride, stride, 1]
        with tf.variable_scope(name):
            in_features = bottom.get_shape()[3].value
            if shape is None:
                # Compute shape out of Bottom
                in_shape = bottom.get_shape()
                h = ((in_shape[1].value - 1) * stride) + 1
                w = ((in_shape[2].value - 1) * stride) + 1
                new_shape = [in_shape[0].value, h, w, num_classes]
            else:
                new_shape = [shape[0], shape[1], shape[2], num_classes]


            deconv_shape = tf.stack([self.batch_size, new_shape[1], new_shape[2], num_classes])


            #logging.debug("Layer: %s, Fan-in: %d" % (name, in_features))
            f_shape = [ksize, ksize, num_classes, in_features]
            # create
            num_input = ksize * ksize * in_features / stride
            stddev = (2 / num_input)**0.5

            ##add padding
            if pad_input==1:
                paddings = tf.constant([ [0, 0], [1, 1,], [1, 1], [0, 0] ])
                #bottom = tf.pad(bottom, paddings, "CONSTANT")
            weights = self.get_deconv_filter(f_shape)
            if relu==1:
                deconv = tf.nn.relu(tf.nn.conv2d_transpose(bottom, weights, deconv_shape,
                                            strides=strides, padding='SAME'))
            else:
                deconv = tf.nn.conv2d_transpose(bottom, weights, deconv_shape,
                                            strides=strides, padding='SAME')

            if debug:
                deconv = tf.Print(deconv, [tf.shape(deconv)],
                                  message='Shape of %s' % name,
                                  summarize=4, first_n=1)


        return deconv

    def get_deconv_filter(self, f_shape):
        width = f_shape[0]
        height = f_shape[1]
        f = math.ceil(width/2.0)
        c = (2 * f - 1 - f % 2) / (2.0 * f)
        bilinear = np.zeros([f_shape[0], f_shape[1]])
        for x in range(width):
            for y in range(height):
                value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
                bilinear[x, y] = value
        weights = np.zeros(f_shape)
        for i in range(f_shape[2]):
            weights[:, :, i, i] = bilinear

        init = tf.constant_initializer(value=weights,
                                       dtype=tf.float32)
        return tf.get_variable(name="up_filter", initializer=init,
                               shape=weights.shape)

    def reconstruction_loss(self,real_images, generated_images):
        """
        The reconstruction loss is defined as the sum of the L1 distances
        between the target images and their generated counterparts
        """
        return tf.reduce_mean(tf.abs(real_images - generated_images))


    def __init__(self, batch_size, trainable):
        self.batch_size = batch_size
        self.trainable = trainable
        self.is_train=tf.placeholder(tf.bool, name="is_train")
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        self.tgt_imgs = tf.placeholder(tf.float32, shape = [None, 224, 224, 3], name = "tgt_imgs")
        mean = [104, 117, 123]
        scale_size = (224,224)
        self.mean = tf.constant([104, 117, 123], dtype=tf.float32)
        self.spec = [mean, scale_size]

        self.model()

        ##assign
        ##assert and cast them to same size!!!!
        self.tgts=self.net_layers['predImg']
        print('.......')
        print(self.tgts.get_shape())
        with tf.name_scope("loss"):
          self.loss = self.reconstruction_loss(self.tgts, self.tgt_imgs)


        tf.summary.scalar('loss', self.loss)
