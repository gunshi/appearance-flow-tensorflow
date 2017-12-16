from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from scipy import misc
from bilinear_sampler import bilinear_sampler

class Net(object): 
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

        #placeholder for a random set of <batch_size> images of fixed size -- 224,224
        self.input_imgs = tf.placeholder(tf.float32, shape = [None, 224, 224, 3], name = "input_imgs")
        self.input_batch_size = tf.shape(self.input_imgs)[0]  # Returns a scalar `tf.Tensor`
        assert(self.input_batch_size == self.batch_size)
        self.tform = tf.placeholder(tf.float32, shape = [None, 20], name = "tform")


        #mean is already subtracted in helper.py as part of preprocessing
        # Conv-Layers
        net_layers={}
        net_layers['Convolution1'] = self.conv(self.input_imgs_m, 3, 3 , 16, name= 'Convolution1', strides=[1,2,2,1] ,padding='VALID', groups=1,pad_input=1)
        net_layers['Convolution2'] = self.conv(net_layers['Convolution1'], 3, 16 , 32, name= 'Convolution2', strides=[1,2,2,1] ,padding='VALID', groups=1,pad_input=1)
        net_layers['Convolution3'] = self.conv(net_layers['Convolution2'], 3, 32 , 64, name= 'Convolution3', strides=[1,2,2,1] ,padding='VALID', groups=1,pad_input=1)
        net_layers['Convolution4'] = self.conv(net_layers['Convolution3'], 3, 64 , 128, name= 'Convolution4', strides=[1,2,2,1] ,padding='VALID', groups=1,pad_input=1)
        net_layers['Convolution5'] = self.conv(net_layers['Convolution4'], 3, 128 , 256, name= 'Convolution5', strides=[1,2,2,1] ,padding='VALID', groups=1,pad_input=1)
        net_layers['Convolution6'] = self.conv(net_layers['Convolution5'], 3, 256 , 512, name= 'Convolution6', strides=[1,2,2,1] ,padding='VALID', groups=1,pad_input=1)


        ##input sizes!!
        net_layers['fc1'] = self.fc(net_layers['Convolution6'], 8*8*512 , 4096, name='fc1', relu = 1)
        if self.is_train:
            net_layers['fc1'] = tf.nn.dropout(net_layers['fc1'], self.keep_prob)
        
        net_layers['fc2'] = self.fc(net_layers['fc1'], 4096 , 4096, name='fc2', relu = 1)
        if self.is_train:
            net_layers['fc2'] = tf.nn.dropout(net_layers['fc2'], self.keep_prob)

        net_layers['fc3'] = self.fc(self.tform, 20 , 128, name='fc3', relu = 1)
        net_layers['fc4'] = self.fc(net_layers['fc3'], 128 , 256, name='fc4', relu = 1)



        net_layers['feat'] = tf.concat([net_layers['drop_out2'], net_layers['fc4']], 0)
        net_layers['fc5'] = self.fc(net_layers['feat'], 4352 , 4096, name='fc5', relu = 1)
        net_layers['fc6'] = self.fc(net_layers['fc5'], 4096 , 4096, name='fc6', relu = 1)
        net_layers['fc6_rs'] = tf.reshape(net_layers['fc6'],shape=[-1, 8, 8, 64], name='fc6_rs')

        #deconv
        net_layers['deconv1'] = upscore = self._upscore_layer(net_layers['fc6_rs'], shape=tf.shape(bgr),
                                           num_classes=256,
                                           debug=debug, name='deconv1', ksize=3, stride=2, pad_input=1)

        net_layers['deconv2'] = upscore = self._upscore_layer(net_layers['deconv1'], shape=tf.shape(bgr),
                                           num_classes=128,
                                           debug=debug, name='deconv1', ksize=3, stride=2, pad_input=1)

        net_layers['deconv3'] = upscore = self._upscore_layer(net_layers['deconv2'], shape=tf.shape(bgr),
                                           num_classes=64,
                                           debug=debug, name='deconv1', ksize=3, stride=2, pad_input=1)

        net_layers['deconv4'] = upscore = self._upscore_layer(net_layers['deconv3'], shape=tf.shape(bgr),
                                           num_classes=32,
                                           debug=debug, name='deconv1', ksize=3, stride=2, pad_input=1)
        net_layers['deconv5'] = upscore = self._upscore_layer(net_layers['deconv4'], shape=tf.shape(bgr),
                                           num_classes=16,
                                           debug=debug, name='deconv1', ksize=3, stride=2, pad_input=1)
        net_layers['deconv6'] = upscore = self._upscore_layer(net_layers['deconv5'], shape=tf.shape(bgr),
                                           num_classes=2,
                                           debug=debug, name='deconv1', ksize=3, stride=2, pad_input=1)      

       #resize to 224 224 to give flow(deconv6) - not needed-function will handle
       ##add gxy to flow to get coords !! not needed -function will handle
       #remap using bilinear on (flow(deconv6) and input_imgs) to get predImg
       net_layers['predImg']=bilinear_sampler(self.input_imgs,net_layers['deconv6'], resize=True)

       net_layers['fc7'] = self.fc(net_layers['feat'], 4352 , 1024, name='fc7', relu = 1)
       net_layers['fc8'] = self.fc(net_layers['fc7'], 1024 , 1024, name='fc8', relu = 1)
       #reshape 8x8x16
       net_layers['fc8_rs'] = tf.reshape(net_layers['fc6'],shape=[-1, 8, 8, 16], name='fc8_rs')

       #deconvs+resize+softmax to get mask output

        net_layers['deconv7'] = upscore = self._upscore_layer(net_layers['fc8_rs'], shape=tf.shape(bgr),
                                           num_classes=256,
                                           debug=debug, name='deconv7', ksize=3, stride=2, pad_input=1)

        net_layers['deconv8'] = upscore = self._upscore_layer(net_layers['deconv7'], shape=tf.shape(bgr),
                                           num_classes=128,
                                           debug=debug, name='deconv8', ksize=3, stride=2, pad_input=1)

        net_layers['deconv9'] = upscore = self._upscore_layer(net_layers['deconv8'], shape=tf.shape(bgr),
                                           num_classes=64,
                                           debug=debug, name='deconv9', ksize=3, stride=2, pad_input=1)

        net_layers['deconv10'] = upscore = self._upscore_layer(net_layers['deconv9'], shape=tf.shape(bgr),
                                           num_classes=32,
                                           debug=debug, name='deconv10', ksize=3, stride=2, pad_input=1)
        net_layers['deconv11'] = upscore = self._upscore_layer(net_layers['deconv10'], shape=tf.shape(bgr),
                                           num_classes=16,
                                           debug=debug, name='deconv11', ksize=3, stride=2, pad_input=1)


        net_layers['deconv12'] = upscore = self._upscore_layer(net_layers['deconv11'], shape=tf.shape(bgr),
                                           num_classes=2,
                                           debug=debug, name='deconv12', ksize=3, stride=1, pad_input=1, relu=0) 

        net_layers['deconv12_rs'] = tf.image.resize_bilinear(net_layers['deconv12'], [224, 224], name='deconv12_rs') ##make 224 as param
        net_layers['predmask_SM'] = tf.nn.softmax(net_layers['deconv12_rs'], name='predmask_SM') 

        self.net_layers = net_layers



   def _upscore_layer(self, bottom, shape,
                       num_classes, name, debug,
                       ksize=3, stride=2, pad_input=1, relu=1):

        if pad_input==1:
            paddings = tf.constant([ [0, 0], [1, 1,], [1, 1], [0, 0] ])
            bottom = tf.pad(bottom, paddings, "CONSTANT")
        strides = [1, stride, stride, 1]
        with tf.variable_scope(name):
            in_features = bottom.get_shape()[3].value

            if shape is None:
                # Compute shape out of Bottom
                in_shape = tf.shape(bottom)

                h = ((in_shape[1] - 1) * stride) + 1
                w = ((in_shape[2] - 1) * stride) + 1
                new_shape = [in_shape[0], h, w, num_classes]
            else:
                new_shape = [shape[0], shape[1], shape[2], num_classes]
            output_shape = tf.stack(new_shape)

            logging.debug("Layer: %s, Fan-in: %d" % (name, in_features))
            f_shape = [ksize, ksize, num_classes, in_features]

            # create
            num_input = ksize * ksize * in_features / stride
            stddev = (2 / num_input)**0.5

            ##add padding

            weights = self.get_deconv_filter(f_shape)
            if relu==1:
                deconv = tf.nn.relu(tf.nn.conv2d_transpose(bottom, weights, output_shape,
                                            strides=strides, padding='VALID'))
            else:
                deconv = tf.nn.conv2d_transpose(bottom, weights, output_shape,
                                            strides=strides, padding='VALID')
                

            if debug:
                deconv = tf.Print(deconv, [tf.shape(deconv)],
                                  message='Shape of %s' % name,
                                  summarize=4, first_n=1)


        return deconv

    def get_deconv_filter(self, f_shape):
        width = f_shape[0]
        height = f_shape[1]
        f = ceil(width/2.0)
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

    def reconstruction_loss(real_images, generated_images):
        """
        The reconstruction loss is defined as the sum of the L1 distances
        between the target images and their generated counterparts
        """
        return tf.reduce_mean(tf.abs(real_images - generated_images))


    def __init__(self, layer, batch_size, trainable):
        self.batch_size = batch_size

        self.layer = layer
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
        with tf.name_scope("loss"):
          self.loss = self.reconstruction_loss(self.tgts, self.tgt_imgs)


      tf.summary.scalar('loss', self.loss)

