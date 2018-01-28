from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from scipy import misc
from bilinear_sampler import bilinear_sampler
import math
from tf.contrib.layers import batch_norm


class Net_tvsn(object):
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

    def conv(self, input_, filter_size, in_channels, out_channels, name, strides, padding, groups, pad_input=1, relu=1):
        if pad_input==1:
            paddings = tf.constant([ [0, 0], [1, 1,], [1, 1], [0, 0] ])
            input_ = tf.pad(input_, paddings, "CONSTANT")

        with tf.variable_scope(name) as scope:
            filt = tf.get_variable('weights', shape=[filter_size, filter_size, int(in_channels/groups), out_channels], trainable=self.trainable)
            bias = tf.get_variable('biases',  shape=[out_channels], trainable=self.trainable)
        if groups == 1:
            if relu:
                return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(input_, filt, strides=strides, padding=padding), bias))
            else:
                return tf.nn.bias_add(tf.nn.conv2d(input_, filt, strides=strides, padding=padding), bias)

        else:
            # Split input_ and weights and convolve them separately
            input_groups = tf.split(axis = 3, num_or_size_splits=groups, value=input_)
            filt_groups = tf.split(axis = 3, num_or_size_splits=groups, value=filt)
            output_groups = [ tf.nn.conv2d( i, k, strides = strides, padding = padding) for i,k in zip(input_groups, filt_groups)]

            conv = tf.concat(axis = 3, values = output_groups)
            if relu:
                return tf.nn.relu(tf.nn.bias_add(conv, bias))
            else:
                return tf.nn.bias_add(conv, bias)



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


#add explainability~~
#complete generator//
#add discrim//
#add semantic seg
#add vgg, resnet, alexnet


    def doafn():
        debug = True
        net_layers = {}
        #placeholder for a random set of <batch_size> images of fixed size -- 224,224
        self.input_imgs = tf.placeholder(tf.float32, shape = [None, 256, 256, 3], name = "input_imgs")
        self.input_batch_size = tf.shape(self.input_imgs)[0]  # Returns a scalar `tf.Tensor`
        self.tform = tf.placeholder(tf.float32, shape = [None, 6], name = "tform") #or 12?
        #mean is already subtracted in helper.py as part of preprocessing
        # Conv-Layers
        net_layers['Convolution1'] = self.conv(net_layers['input_imgs'], 5, 3 , 16, name= 'Convolution1', strides=[1,2,2,1] ,padding='VALID', groups=1,pad_input=1)
        net_layers['Convolution2'] = self.conv(net_layers['Convolution1'], 5, 16 , 32, name= 'Convolution2', strides=[1,2,2,1] ,padding='VALID', groups=1,pad_input=1)
        net_layers['Convolution3'] = self.conv(net_layers['Convolution2'], 5, 32 , 64, name= 'Convolution3', strides=[1,2,2,1] ,padding='VALID', groups=1,pad_input=1)
        net_layers['Convolution4'] = self.conv(net_layers['Convolution3'], 3, 64 , 128, name= 'Convolution4', strides=[1,2,2,1] ,padding='VALID', groups=1,pad_input=1)
        net_layers['Convolution5'] = self.conv(net_layers['Convolution4'], 3, 128 , 256, name= 'Convolution5', strides=[1,2,2,1] ,padding='VALID', groups=1,pad_input=1)
        net_layers['Convolution6'] = self.conv(net_layers['Convolution4'], 3, 256 , 512, name= 'Convolution5', strides=[1,2,2,1] ,padding='VALID', groups=1,pad_input=1)


        ##add fcs for bottleneck with transform info
        net_layers['fc6'] = self.fc(net_layers['Convolution6'], 4*4*512 , 2048, name='fc1', relu = 1)


        #if self.is_train:
        #    net_layers['fc6'] = tf.nn.dropout(net_layers['fc6'], self.keep_prob)

        net_layers['view_fc1'] = self.fc(self.tform, 6 , 128, name='view_fc1', relu = 1)
        #if self.is_train:
        #    net_layers['fc2'] = tf.nn.dropout(net_layers['fc2'], self.keep_prob)

        net_layers['view_fc2'] = self.fc(self.tform, 128 , 256, name='view_fc2', relu = 1)

        net_layers['view_concat'] = tf.concat([net_layers['fc6'], net_layers['view_fc2']], 0) ##is this 0 dimension correct?

        net_layers['de_fc1'] = self.fc(net_layers['view_concat'], 2304 , 2048, name='de_fc1', relu = 1)
        
        if self.is_train:
            net_layers['de_fc1'] = tf.nn.dropout(net_layers['de_fc1'], self.keep_prob)
        
        net_layers['de_fc2'] = self.fc(net_layers['view_concat'], 2048 , 2048, name='de_fc2', relu = 1)
        
        if self.is_train:
            net_layers['de_fc2'] = tf.nn.dropout(net_layers['de_fc2'], self.keep_prob)

        net_layers['de_fc3'] = self.fc(net_layers['view_concat'], 2048 , 512*4*4, name='de_fc3', relu = 1)
        
        

        net_layers['de_fc3_rs'] = tf.reshape(net_layers['de_fc3'],shape=[-1, 4, 4, 512], name='de_fc3_rs')




        #check paddings! especially for 5 size kernel case!

        deconv1_x2 = tf.image.resize_bilinear(net_layers['de_fc3_rs'], [8, 8])
        net_layers['deconv1'] = self.conv(deconv1_x2, 3, 512 , 256, name= 'deconv1', strides=[1,1,1,1] ,padding='VALID', groups=1,pad_input=1)


        deconv2_x2 = tf.image.resize_bilinear(net_layers['deconv1'], [16, 16])
        net_layers['deconv2'] = self.conv(deconv2_x2, 3, 256 , 128, name= 'deconv2', strides=[1,1,1,1] ,padding='VALID', groups=1,pad_input=1)

        deconv3_x2 = tf.image.resize_bilinear(net_layers['deconv2'], [32, 32])
        net_layers['deconv3'] = self.conv(deconv3_x2, 3, 128 , 64, name= 'deconv3', strides=[1,1,1,1] ,padding='VALID', groups=1,pad_input=1)

        deconv4_x2 = tf.image.resize_bilinear(net_layers['deconv3'], [64, 64])
        net_layers['deconv4'] = self.conv(deconv4_x2, 5, 64 , 32, name= 'deconv4', strides=[1,1,1,1] ,padding='VALID', groups=1,pad_input=1)

        deconv5_x2 = tf.image.resize_bilinear(net_layers['deconv4'], [128, 128])
        net_layers['deconv5'] = self.conv(deconv5_x2, 5, 32 , 16, name= 'deconv5', strides=[1,1,1,1] ,padding='VALID', groups=1,pad_input=1)
        
        deconv6_x2 = tf.image.resize_bilinear(net_layers['deconv5'], [256, 256])
        net_layers['deconv6'] = tf.nn.tanh(self.conv(deconv6_x2, 5, 16 , 2, name= 'deconv6', strides=[1,1,1,1] ,padding='VALID', groups=1,pad_input=1))


        #remap using bilinear on (flow(deconv6) and input_imgs) to get predImg
        net_layers['predImg']=bilinear_sampler(self.input_imgs,net_layers['deconv6'], resize=True)

        deconv_x2_mask = tf.image.resize_bilinear(net_layers['deconv5'], [256, 256])
        net_layers['deconv_mask'] = self.conv(deconv_x2_mask, 5, 16 , 1, name= 'deconv_mask', strides=[1,1,1,1] ,padding='VALID', groups=1,pad_input=1)






        #deconv5 (16 channels should be), upsample, then conv to 1 channel

        self.net_layers = net_layers


    def afn_old(self):

        debug=True
        net_layers={}
        #placeholder for a random set of <batch_size> images of fixed size -- 224,224
        self.input_imgs = tf.placeholder(tf.float32, shape = [None, 224, 224, 3], name = "input_imgs")
        self.input_batch_size = tf.shape(self.input_imgs)[0]  # Returns a scalar `tf.Tensor`
        self.tform = tf.placeholder(tf.float32, shape = [None, 224, 224, 6], name = "tform")
        net_layers['input_stack'] = tf.concat([self.input_imgs, self.tform], 3)

        #mean is already subtracted in helper.py as part of preprocessing
        # Conv-Layers

        net_layers['Convolution1'] = self.conv(net_layers['input_stack'], 3, 9 , 32, name= 'Convolution1', strides=[1,2,2,1] ,padding='VALID', groups=1,pad_input=1)
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
                                           num_classes=2,
                                           debug=debug, name='deconv6', ksize=3, stride=1, pad_input=1)

       #resize to 224 224 to give flow(deconv6) - not needed-function will handle
       ##add gxy to flow to get coords !! not needed -function will handle
       #remap using bilinear on (flow(deconv6) and input_imgs) to get predImg
        net_layers['predImg']=bilinear_sampler(self.input_imgs,net_layers['deconv6'], resize=True)

        #deconv5 (16 channels should be), upsample, then conv to 1 channel

        self.net_layers = net_layers


    def tvsn_gen():

    #check padding!!!!
    ##check what this reuse means!!!!

        #encoder
        gen_conv1 =  self.conv(self.net_layers['predImg'],4,3,16,name='',strides=[1,2,2,1] ,padding='VALID', groups=1,pad_input=1)
        self.net_layers['gen_conv_bn1']= tf.nn.leaky_relu(batch_norm(gen_conv1,decay=0.9, is_training = phase, updates_collections = None,reuse=reuse,zero_debias_moving_mean=True, scope='gen_conv1'))
        
        gen_conv2 =  self.conv(self.net_layers['gen_conv_bn1'],4,16,32,name='',strides=[1,2,2,1] ,padding='VALID', groups=1,pad_input=1)
        self.net_layers['gen_conv_bn2']= tf.nn.leaky_relu(batch_norm(gen_conv2,decay=0.9, is_training = phase, updates_collections = None,reuse=reuse,zero_debias_moving_mean=True, scope='gen_conv2'))
        
        gen_conv3 =  self.conv(self.net_layers['gen_conv_bn2'],4,32,64,name='',strides=[1,2,2,1] ,padding='VALID', groups=1,pad_input=1)
        self.net_layers['gen_conv_bn3']= tf.nn.leaky_relu(batch_norm(gen_conv3,decay=0.9, is_training = phase, updates_collections = None,reuse=reuse,zero_debias_moving_mean=True, scope='gen_conv3'))
        
        gen_conv4 =  self.conv(self.net_layers['gen_conv_bn3'],4,64,128,name='',strides=[1,2,2,1] ,padding='VALID', groups=1,pad_input=1)
        self.net_layers['gen_conv_bn4']= tf.nn.leaky_relu(batch_norm(gen_conv4,decay=0.9, is_training = phase, updates_collections = None,reuse=reuse,zero_debias_moving_mean=True, scope='gen_conv4'))
        
        gen_conv5 =  self.conv(self.net_layers['gen_conv_bn4'],4,128,256,name='',strides=[1,2,2,1] ,padding='VALID', groups=1,pad_input=1)
        self.net_layers['gen_conv_bn5']= tf.nn.leaky_relu(batch_norm(gen_conv5,decay=0.9, is_training = phase, updates_collections = None,reuse=reuse,zero_debias_moving_mean=True, scope='gen_conv5'))
             
        gen_conv6 =  self.conv(self.net_layers['gen_conv_bn5'],4,256,512,name='',strides=[1,2,2,1] ,padding='VALID', groups=1,pad_input=1)
        self.net_layers['gen_conv_bn6']= tf.nn.leaky_relu(batch_norm(gen_conv6,decay=0.9, is_training = phase, updates_collections = None,reuse=reuse,zero_debias_moving_mean=True, scope='gen_conv6'))
        

        #bottleneck
        self.net_layers['gen_view_fc1'] = self.fc(self.tform, 6 , 128, name='gen_view_fc1', relu = 1)
        #if self.is_train:
        #    net_layers['fc2'] = tf.nn.dropout(net_layers['fc2'], self.keep_prob)
        self.net_layers['gen_view_fc2'] = tf.reshape(self.fc(self.tform, 128 , 128, name='gen_view_fc2', relu = 1), shape=[-1,1,1,128])

        net_layers['gen_view_conv'] = tf.contrib.layers.conv2d_transpose(net_layers['gen_view_fc2'],128,4) 

        net_layers['gen_view_conv'] = tf.nn.relu(batch_norm(net_layers['gen_view_conv'], is_training=phase, updates_collections=None,reuse=reuse,zero_debias_moving_mean=True, scope='gen_view_conv'))


        net_layers['concat1'] = tf.concat([net_layers['gen_conv_bn6'], net_layers['gen_view_conv'], net_layers['']], 0) ##is this 0 dimension correct?
        concat2 =  self.conv(self.net_layers['concat1'], 3, 512+512+128, 512, name='', strides=[1,1,1,1] , padding='VALID', groups=1,pad_input=1, relu=0)
        concat2 = tf.nn.relu(batch_norm(concat2,is_training=phase, updates_collections=None,reuse=reuse,zero_debias_moving_mean=True, scope='concat2' ))


        concat3 =  self.conv(self.net_layers['concat2'], 3, 512, 512, name='', strides=[1,1,1,1] , padding='VALID', groups=1,pad_input=1, relu=0)
        concat3 = tf.nn.relu(batch_norm(concat3,is_training=phase, updates_collections=None,reuse=reuse,zero_debias_moving_mean=True, scope='concat3' ))


##decoder        



#convert the whole thing to self.netlayers 

    def tvsn_discrim():
        feat1 =  self.conv(self.net_layers['predImg_final'],4,3,64,name='',strides=[1,2,2,1] ,padding='VALID', groups=1,pad_input=1)
        self.net_layers['feat1']= tf.nn.leaky_relu(batch_norm(gen_conv1,decay=0.9, is_training = phase, updates_collections = None,reuse=reuse,zero_debias_moving_mean=True, scope='feat1'))
        
        feat2 =  self.conv(self.net_layers['feat1'],4,64,64,name='',strides=[1,2,2,1] ,padding='VALID', groups=1,pad_input=1)
        self.net_layers['feat2']= tf.nn.leaky_relu(batch_norm(gen_conv2,decay=0.9, is_training = phase, updates_collections = None,reuse=reuse,zero_debias_moving_mean=True, scope='feat2'))
        
        feat3 =  self.conv(self.net_layers['feat2'],4,64,64,name='',strides=[1,2,2,1] ,padding='VALID', groups=1,pad_input=1)
        self.net_layers['feat3']= tf.nn.leaky_relu(batch_norm(gen_conv3,decay=0.9, is_training = phase, updates_collections = None,reuse=reuse,zero_debias_moving_mean=True, scope='feat3'))
        
        feat4 =  self.conv(self.net_layers['feat3'],4,64,128,name='',strides=[1,2,2,1] ,padding='VALID', groups=1,pad_input=1)
        self.net_layers['feat4']= tf.nn.leaky_relu(batch_norm(gen_conv4,decay=0.9, is_training = phase, updates_collections = None,reuse=reuse,zero_debias_moving_mean=True, scope='feat4'))
        
        feat5 =  self.conv(self.net_layers['feat4'],4,128,256,name='',strides=[1,2,2,1] ,padding='VALID', groups=1,pad_input=1)
        self.net_layers['feat5']= tf.nn.leaky_relu(batch_norm(gen_conv5,decay=0.9, is_training = phase, updates_collections = None,reuse=reuse,zero_debias_moving_mean=True, scope='feat5'))
             
        feat6 =  self.conv(self.net_layers['feat5'],4,256,512,name='',strides=[1,2,2,1] ,padding='VALID', groups=1,pad_input=1)
        self.net_layers['feat6']= tf.nn.leaky_relu(batch_norm(gen_conv6,decay=0.9, is_training = phase, updates_collections = None,reuse=reuse,zero_debias_moving_mean=True, scope='feat6'))
        out =  self.conv(self.net_layers['feat6'],4,512,1,name='',strides=[1,1,1,1] ,padding='VALID', groups=1,pad_input=0)
        out = tf.nn.sigmoid(out)
        #out -> reshape to flatten out
        ##check if this works identically in terms of output size


    def _upscore_layer(self, bottom, shape,num_classes, name, debug, ksize=3, stride=2, pad_input=1, relu=1, mode='bilinear'):

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

            ##add a condition for bilinear here    
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

    def vgg_loss():

    def tvloss(generated_images):
        return tf.image.total_variation(generated_images) 

    def loss_gen():

    def loss_discrim():

    def loss_doafn():

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

        self.doafn()

        ##assign
        ##assert and cast them to same size!!!!
        self.tgts=self.net_layers['predImg']
        print('.......')
        print(self.tgts.get_shape())
        with tf.name_scope("loss"):
          self.loss = self.reconstruction_loss(self.tgts, self.tgt_imgs)


        tf.summary.scalar('loss', self.loss)

