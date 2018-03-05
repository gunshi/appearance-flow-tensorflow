from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
from scipy import misc
from bilinear_sampler import bilinear_sampler
import math
from tensorflow.contrib.layers import batch_norm
#from tensorflow.contrib.resampler import resampler
import layers


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

    def conv(self, input_, filter_size, in_channels, out_channels, name, strides, padding, groups, pad_input=1, relu=1, pad_num=1):
        if pad_input==1:
            paddings = tf.constant([ [0, 0], [pad_num, pad_num,], [pad_num, pad_num], [0, 0] ])
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


    def build_resnet_block(inputres, dim, name="resnet", padding="REFLECT"):
        """Build a single block of resnet.

        :param inputres: inputres
        :param dim: dim
        :param name: name
        :param padding: for tensorflow version use REFLECT; for pytorch version use
         CONSTANT
        :return: a single block of resnet.
        """
        with tf.variable_scope(name):
            out_res = tf.pad(inputres, [[0, 0], [1, 1], [
                1, 1], [0, 0]], padding)
            out_res = layers.general_conv2d(
                out_res, dim, 3, 3, 1, 1, 0.02, "VALID", "c1")
            out_res = tf.pad(out_res, [[0, 0], [1, 1], [1, 1], [0, 0]], padding)
            out_res = layers.general_conv2d(
                out_res, dim, 3, 3, 1, 1, 0.02, "VALID", "c2", do_relu=False)

            return tf.nn.relu(out_res + inputres)

    def encdec_resnet_fcn(inputs, network_id, num_separate_layers, num_no_skip_layers):
        """

        The generator consists of three parts: Conv, ResNet blocks and DeConv.
        in the middle blocks.

        Args:
            inputs: a tensor as the input image.
            network_id: an integer as the id of the network (1-4).
            num_separate_layers: an integer as the number of separate layers.
            num_no_skip_layers: a dummy variable which is not used.
        """
        fl_ks = 7  # kernel size of the first and last layer
        ks = 3
        padding = "CONSTANT"

        _num_generator_filters = 16
        reuse = False

        #scope, reuse = get_scope_and_reuse_conv(network_id)
        scope = 'conv_enc_initial'
        with tf.variable_scope(scope):
            if reuse is True:
                tf.get_variable_scope().reuse_variables()
            pad_input = tf.pad(
                inputs, [[0, 0], [ks, ks], [ks, ks], [0, 0]], padding)

            o_c1 = layers.general_conv2d(
                pad_input, _num_generator_filters, fl_ks, fl_ks, 1, 1, 0.02, name="c1")  # noqa
            
            o_c2 = layers.general_conv2d(
                o_c1, _num_generator_filters * 2, ks, ks, 2, 2, 0.02, "SAME", "c2")  # noqa
            
            o_c3 = layers.general_conv2d(
                o_c2, _num_generator_filters * 4, ks, ks, 2, 2, 0.02, "SAME", "c3")  # noqa

        in_t = o_c3
        channel_factor = 4 
        for i in range(3):
            scope = 'conv_enc_res_middle'
            #scope, reuse = get_scope_and_reuse_resnet(
                #network_id, i, 9, num_separate_layers)
            with tf.variable_scope(scope):
                if reuse is True:
                    tf.get_variable_scope().reuse_variables()
                out = build_resnet_block(
                    in_t, _num_generator_filters * channel_factor, 'r{}'.format(i),
                    padding)
                in_t = out
                channel_factor = channel_factor*2
                in_t = layers.general_conv2d(
                    out, _num_generator_filters * channel_factor, ks, ks, 2, 2, 0.02, "SAME", "c"+str(i))  # noqa

        scope = 'conv_enc_after_res'
        with tf.variable_scope(scope):
            if reuse is True:
                tf.get_variable_scope().reuse_variables()
            #pad_input = tf.pad(
                #inputs, [[0, 0], [ks, ks], [ks, ks], [0, 0]], padding)

            o_c7 = layers.general_conv2d(
                in_t, _num_generator_filters*channel_factor, ks, ks, 2, 2, 0.02, "SAME", "c7")  # noqa
            
            o_c8 = layers.general_conv2d(
                o_c7, _num_generator_filters *channel_factor, ks, ks, 2, 2, 0.02, "SAME", "c8")  # noqa
            
        #check channels

        # add bottleneck processing + transfrmation fc

        # 4 more deconv or resize + conv1x1 + 2 residual
        #viewconcat
        #net_layerify

        #scope, reuse = get_scope_and_reuse_deconv(network_id)
        scope = 'resnet_deconv'
        with tf.variable_scope(scope):
            if reuse is True:
                tf.get_variable_scope().reuse_variables()

            o_dc1 = layers.general_deconv2d(
                viewconcat, [BATCH_SIZE, 128, 128, _num_generator_filters *
                      2], _num_generator_filters * 2, ks, ks, 2, 2, 0.02,
                "SAME", "c4")

            o_dc2 = layers.general_deconv2d(
                o_dc1, [BATCH_SIZE, 128, 128, _num_generator_filters *
                      2], _num_generator_filters * 2, ks, ks, 2, 2, 0.02,
                "SAME", "c4")

            o_dc3 = layers.general_deconv2d(
                o_dc2, [BATCH_SIZE, 128, 128, _num_generator_filters *
                      2], _num_generator_filters * 2, ks, ks, 2, 2, 0.02,
                "SAME", "c4")




            
            o_dc4 = layers.general_deconv2d(
                o_dc3, [BATCH_SIZE, 128, 128, _num_generator_filters *
                      2], _num_generator_filters * 2, ks, ks, 2, 2, 0.02,
                "SAME", "c4")
            
            o_dc5 = layers.general_deconv2d(
                o_dc4, [BATCH_SIZE, 256, 256, _num_generator_filters],
                _num_generator_filters, ks, ks, 2, 2, 0.02,
                "SAME", "c5")
            

            o_dc6 = layers.general_conv2d(o_dc5, IMG_CHANNELS, fl_ks, fl_ks,
                                         1, 1, 0.02, "SAME", "c6",
                                         do_norm=False, do_relu=False)

            out_gen = tf.nn.tanh(o_c6, "t1")

        return out_gen


    def generator_resnet_9blocks(inputs, network_id, num_separate_layers, num_no_skip_layers):
        """Build 9 blocks of ResNet as generator.

        The generator consists of three parts: Conv, ResNet blocks and DeConv.
        Conv and DeConv are not shared. The ResNet blocks are partially shared
        in the middle blocks.

        Args:
            inputs: a tensor as the input image.
            network_id: an integer as the id of the network (1-4).
            num_separate_layers: an integer as the number of separate layers.
            num_no_skip_layers: a dummy variable which is not used.
        """
        fl_ks = 7  # kernel size of the first and last layer
        ks = 3
        padding = "CONSTANT"

        _num_generator_filters = 32
        reuse = False

        #scope, reuse = get_scope_and_reuse_conv(network_id)
        scope = 'resnet_conv'
        with tf.variable_scope(scope):
            if reuse is True:
                tf.get_variable_scope().reuse_variables()
            pad_input = tf.pad(
                inputs, [[0, 0], [ks, ks], [ks, ks], [0, 0]], padding)

            o_c1 = layers.general_conv2d(
                pad_input, _num_generator_filters, fl_ks, fl_ks, 1, 1, 0.02, name="c1")  # noqa
            
            o_c2 = layers.general_conv2d(
                o_c1, _num_generator_filters * 2, ks, ks, 2, 2, 0.02, "SAME", "c2")  # noqa
            
            o_c3 = layers.general_conv2d(
                o_c2, _num_generator_filters * 4, ks, ks, 2, 2, 0.02, "SAME", "c3")  # noqa

        in_t = o_c3
        for i in range(9):
            scope = 'resnet_middle'
            #scope, reuse = get_scope_and_reuse_resnet(
                #network_id, i, 9, num_separate_layers)
            with tf.variable_scope(scope):
                if reuse is True:
                    tf.get_variable_scope().reuse_variables()
                out = build_resnet_block(
                    in_t, _num_generator_filters * 4, 'r{}'.format(i),
                    padding)
                in_t = out

        #scope, reuse = get_scope_and_reuse_deconv(network_id)
        scope = 'resnet_deconv'
        with tf.variable_scope(scope):
            if reuse is True:
                tf.get_variable_scope().reuse_variables()
            
            o_c4 = layers.general_deconv2d(
                out, [BATCH_SIZE, 128, 128, _num_generator_filters *
                      2], _num_generator_filters * 2, ks, ks, 2, 2, 0.02,
                "SAME", "c4")
            
            o_c5 = layers.general_deconv2d(
                o_c4, [BATCH_SIZE, 256, 256, _num_generator_filters],
                _num_generator_filters, ks, ks, 2, 2, 0.02,
                "SAME", "c5")
            
            o_c6 = layers.general_conv2d(o_c5, IMG_CHANNELS, fl_ks, fl_ks,
                                         1, 1, 0.02, "SAME", "c6",
                                         do_norm=False, do_relu=False)

            out_gen = tf.nn.tanh(o_c6, "t1")

        return out_gen


    def get_encoder_layer_specs():
        """Return number of output channels of each encoder layer."""
        return [
            # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
            32 * 2,
            # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
            32 * 4,
            # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
            32 * 8,
            # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
            32 * 8,
            # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
            32 * 16,
            # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
            32 * 16,
        ]


    def get_decoder_layer_specs():
        """Get number of output channels and dropout ratio in decoder."""
        return [
            # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 *
            # 2]
            (32 * 16, 0.5),
            # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8
            # * 2]
            (32 * 8, 0.0),
            # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 *
            # 2]
            (32 * 8, 0.0),
            # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 *
            # 2]
            (32 * 4, 0.0),
            # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 *
            # 2]
            (32 * 2, 0.0),
            # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf
            # * 2]
            (32, 0.0),
        ]

    def encoder_decoder():
        """The autoencoder network architecture from pix2pix.

        Args:
            inputs: a tensor as the input to the encoder decoder.
            network_id: an integer as the index of the network.
                If network_id == 1, it is the encoder_deoder of the first domain.
                if network_id == 2, it is the encoder_deoder of the second domain.
                if network_id == 3, it is the encoder_deoder that reuses network 2.
                if network_id == 4, it is the encoder_deoder that reuses network 1.
            num_separate_layers: an integer as the number of separate layers.
            num_no_skip_layers: an integer as the number of layers without skip
                connection.

        Return:
            A tensor as the output of the encoder_decoder.
        """
        all_layers = []
        reuse=False
        #scope, reuse = get_scope_and_reuse_encoder(
            #network_id, 0, num_separate_layers)
        scope='encoder'
        with tf.variable_scope(scope):
            if reuse is True:
                tf.get_variable_scope().reuse_variables()
            output = layers.conv(inputs, 32, stride=2)
            all_layers.append(output)

        layer_specs = get_encoder_layer_specs()

        total_num_layers = len(layer_specs) + 1
        for i, out_channels in enumerate(layer_specs):
            #scope, reuse = get_scope_and_reuse_encoder(
                #network_id, (i + 1), num_separate_layers)
            with tf.variable_scope(scope):
                if reuse is True:
                    tf.get_variable_scope().reuse_variables()
                rectified = layers.p2p_lrelu(all_layers[-1], 0.2) ##################################swap order with batch norm later
                convolved = layers.conv(rectified, out_channels, stride=2)
                output = layers.batchnorm(convolved)
                print(i)
                print(output.shape)
                all_layers.append(output)


        """

        ##add bottleneck
        net_layers['fc_conv6'] = self.fc(net_layers['Convolution6'], 4*4*512 , 2048, name='fc_conv6', relu = 1)
        net_layers['view_fc1'] = self.fc(self.tform, 6 , 128, name='view_fc1', relu = 1)
        net_layers['view_fc2'] = self.fc(view_fc1, 128 , 256, name='view_fc2', relu = 1)
        net_layers['view_concat'] = tf.concat([net_layers['fc_conv6'], net_layers['view_fc2']], 0) ##is this 0 dimension correct?

        net_layers['de_fc1'] = self.fc(net_layers['view_concat'], 2304 , 2048, name='de_fc1', relu = 1)
        
        if self.is_train:
            net_layers['de_fc1'] = tf.nn.dropout(net_layers['de_fc1'], self.keep_prob)
        
        net_layers['de_fc2'] = self.fc(net_layers['view_concat'], 2048 , 2048, name='de_fc2', relu = 1)
        
        if self.is_train:
            net_layers['de_fc2'] = tf.nn.dropout(net_layers['de_fc2'], self.keep_prob)


        """ 
        print('now decoding')
        # decoder part
        layer_specs = get_decoder_layer_specs()
        for i, (out_channels, dropout) in enumerate(layer_specs):
            current_layer = total_num_layers - i - 1
            #scope, reuse = get_scope_and_reuse_decoder(
                #network_id, current_layer, num_separate_layers)
            scope=''
            with tf.variable_scope(scope):
                if reuse is True:
                    tf.get_variable_scope().reuse_variables()

                
                input = all_layers[-1]
               
                rectified = tf.nn.relu(input)
                output = layers.deconv(rectified, out_channels)
                output = layers.batchnorm(output)
                print(output.shape)
                if dropout > 0.0:
                    output = tf.nn.dropout(output, keep_prob=1 - dropout)
                all_layers.append(output)

        #scope, reuse = get_scope_and_reuse_decoder(
            #network_id, 0, num_separate_layers)
        scope=''
        with tf.variable_scope(scope):
            if reuse is True:
                tf.get_variable_scope().reuse_variables()
            input = all_layers[-1]
            rectified = tf.nn.relu(input)
            output = layers.deconv(rectified, 3)
            output = tf.tanh(output)
            all_layers.append(output)

        return all_layers[-1]


    def doafn():

        #input is mean subtracted, normalised to -1 to 1
        debug = True
        net_layers = {}
        self.input_imgs = tf.placeholder(tf.float32, shape = [None, 256, 256, 3], name = "input_imgs")
        self.input_batch_size = tf.shape(self.input_imgs)[0]  # Returns a scalar `tf.Tensor`
        self.tform = tf.placeholder(tf.float32, shape = [None, 6], name = "tform") #or 12?

        # Conv-Layers
        net_layers['Convolution1'] = self.conv(net_layers['input_imgs'], 5, 3 , 16, name= 'Convolution1', strides=[1,2,2,1] ,padding='VALID', groups=1,pad_input=1, pad_num=2)

        net_layers['Convolution2'] = self.conv(net_layers['Convolution1'], 5, 16 , 32, name= 'Convolution2', strides=[1,2,2,1] ,padding='VALID', groups=1,pad_input=1, pad_num=2)

        net_layers['Convolution3'] = self.conv(net_layers['Convolution2'], 5, 32 , 64, name= 'Convolution3', strides=[1,2,2,1] ,padding='VALID', groups=1,pad_input=1, pad_num=2)

        net_layers['Convolution4'] = self.conv(net_layers['Convolution3'], 3, 64 , 128, name= 'Convolution4', strides=[1,2,2,1] ,padding='VALID', groups=1,pad_input=1)

        net_layers['Convolution5'] = self.conv(net_layers['Convolution4'], 3, 128 , 256, name= 'Convolution5', strides=[1,2,2,1] ,padding='VALID', groups=1,pad_input=1)

        net_layers['Convolution6'] = self.conv(net_layers['Convolution4'], 3, 256 , 512, name= 'Convolution5', strides=[1,2,2,1] ,padding='VALID', groups=1,pad_input=1)


        ##add fcs for bottleneck with transform info
        net_layers['fc_conv6'] = self.fc(net_layers['Convolution6'], 4*4*512 , 2048, name='fc_conv6', relu = 1)
        net_layers['view_fc1'] = self.fc(self.tform, 6 , 128, name='view_fc1', relu = 1)
        net_layers['view_fc2'] = self.fc(view_fc1, 128 , 256, name='view_fc2', relu = 1)
        net_layers['view_concat'] = tf.concat([net_layers['fc_conv6'], net_layers['view_fc2']], 0) ##is this 0 dimension correct?

        net_layers['de_fc1'] = self.fc(net_layers['view_concat'], 2304 , 2048, name='de_fc1', relu = 1)
        
        if self.is_train:
            net_layers['de_fc1'] = tf.nn.dropout(net_layers['de_fc1'], self.keep_prob)
        
        net_layers['de_fc2'] = self.fc(net_layers['de_fc1'], 2048 , 2048, name='de_fc2', relu = 1)
        
        if self.is_train:
            net_layers['de_fc2'] = tf.nn.dropout(net_layers['de_fc2'], self.keep_prob)

        net_layers['de_fc3'] = self.fc(net_layers['de_fc2'], 2048 , 512*4*4, name='de_fc3', relu = 1)
        net_layers['de_fc3_rs'] = tf.reshape(net_layers['de_fc3'],shape=[-1, 4, 4, 512], name='de_fc3_rs')
       

        deconv1_x2 = tf.image.resize_bilinear(net_layers['de_fc3_rs'], [8, 8])
        net_layers['deconv1'] = self.conv(deconv1_x2, 3, 512 , 256, name= 'deconv1', strides=[1,1,1,1] ,padding='VALID', groups=1,pad_input=1)

        deconv2_x2 = tf.image.resize_bilinear(net_layers['deconv1'], [16, 16])
        net_layers['deconv2'] = self.conv(deconv2_x2, 3, 256 , 128, name= 'deconv2', strides=[1,1,1,1] ,padding='VALID', groups=1,pad_input=1)

        deconv3_x2 = tf.image.resize_bilinear(net_layers['deconv2'], [32, 32])
        net_layers['deconv3'] = self.conv(deconv3_x2, 3, 128 , 64, name= 'deconv3', strides=[1,1,1,1] ,padding='VALID', groups=1,pad_input=1)

        deconv4_x2 = tf.image.resize_bilinear(net_layers['deconv3'], [64, 64])
        net_layers['deconv4'] = self.conv(deconv4_x2, 5, 64 , 32, name= 'deconv4', strides=[1,1,1,1] ,padding='VALID', groups=1,pad_input=1, pad_num=2)

        deconv5_x2 = tf.image.resize_bilinear(net_layers['deconv4'], [128, 128])
        net_layers['deconv5'] = self.conv(deconv5_x2, 5, 32 , 16, name= 'deconv5', strides=[1,1,1,1] ,padding='VALID', groups=1,pad_input=1, pad_num=2)
        
        deconv6_x2 = tf.image.resize_bilinear(net_layers['deconv5'], [256, 256])
        net_layers['deconv6'] = tf.nn.tanh(self.conv(deconv6_x2, 5, 16 , 3, name= 'deconv6', strides=[1,1,1,1] ,padding='VALID', groups=1,pad_input=1, pad_num=2))

        #do something additonal to the image here?
        #batch norm?

        deconv_x2_mask = tf.image.resize_bilinear(net_layers['deconv5'], [256, 256])
        net_layers['deconv_mask'] = self.conv(deconv_x2_mask, 5, 16 , 2, name= 'deconv_mask', strides=[1,1,1,1] ,padding='VALID', groups=1,pad_input=1, pad_num=2)

        self.net_layers = net_layers


    def doafn_aspect_wide(self):

        #input is mean subtracted, normalised to -1 to 1
        debug = True
        net_layers = {}
        self.input_imgs = tf.placeholder(tf.float32, shape = [None, 224, 448, 3], name = "input_imgs")
        self.input_batch_size = tf.shape(self.input_imgs)[0]  # Returns a scalar `tf.Tensor`
        self.tform = tf.placeholder(tf.float32, shape = [None, 6], name = "tform") #or 12?

        # Conv-Layers
        net_layers['Convolution1'] = self.conv(self.input_imgs, 5, 3 , 16, name= 'Convolution1', strides=[1,2,2,1] ,padding='VALID', groups=1,pad_input=1, pad_num=2)
        print(net_layers['Convolution1'].shape)


        net_layers['Convolution2'] = self.conv(net_layers['Convolution1'], 5, 16 , 32, name= 'Convolution2', strides=[1,2,2,1] ,padding='VALID', groups=1,pad_input=1, pad_num=2)
        print(net_layers['Convolution2'].shape)

        net_layers['Convolution3'] = self.conv(net_layers['Convolution2'], 5, 32 , 64, name= 'Convolution3', strides=[1,2,2,1] ,padding='VALID', groups=1,pad_input=1, pad_num=2)
        print(net_layers['Convolution3'].shape)


        net_layers['Convolution4'] = self.conv(net_layers['Convolution3'], 3, 64 , 128, name= 'Convolution4', strides=[1,2,2,1] ,padding='VALID', groups=1,pad_input=1)
        print(net_layers['Convolution4'].shape)

        net_layers['Convolution5'] = self.conv(net_layers['Convolution4'], 3, 128 , 256, name= 'Convolution5', strides=[1,2,2,1] ,padding='VALID', groups=1,pad_input=1)
        print(net_layers['Convolution5'].shape)

        net_layers['Convolution6'] = self.conv(net_layers['Convolution5'], 3, 256 , 512, name= 'Convolution6', strides=[1,2,2,1] ,padding='VALID', groups=1,pad_input=1)

        print(net_layers['Convolution6'].shape)
        print(tf.shape(net_layers['Convolution6']))
        ##add fcs for bottleneck with transform info
        net_layers['fc_conv6'] = self.fc(net_layers['Convolution6'], 4*7*512 , 2048, name='fc_conv6', relu = 1)
        net_layers['view_fc1'] = self.fc(self.tform, 6 , 128, name='view_fc1', relu = 1)
        net_layers['view_fc2'] = self.fc(net_layers['view_fc1'], 128 , 256, name='view_fc2', relu = 1)
        print(net_layers['fc_conv6'].shape)
        print(net_layers['view_fc2'].shape)

        net_layers['view_concat'] = tf.concat([net_layers['fc_conv6'], net_layers['view_fc2']], 1) ##is this 0 dimension correct?

        net_layers['de_fc1'] = self.fc(net_layers['view_concat'], 2304 , 2048, name='de_fc1', relu = 1)
        
        net_layers['de_fc1'] = tf.cond(self.is_train, lambda:tf.nn.dropout(net_layers['de_fc1'], self.keep_prob) , lambda: net_layers['de_fc1'])
        
        net_layers['de_fc2'] = self.fc(net_layers['de_fc1'], 2048 , 2048, name='de_fc2', relu = 1)
        
        net_layers['de_fc2'] = tf.cond(self.is_train, lambda:tf.nn.dropout(net_layers['de_fc2'], self.keep_prob) , lambda: net_layers['de_fc2'])

        net_layers['de_fc3'] = self.fc(net_layers['de_fc2'], 2048 , 512*4*7, name='de_fc3', relu = 1)
        net_layers['de_fc3_rs'] = tf.reshape(net_layers['de_fc3'],shape=[-1, 4, 7, 512], name='de_fc3_rs')
       




        #check paddings! especially for 5 size kernel case!
        #THEY HAVE DONE NEAREST NEIGHBOUR RESAMPLING NOT BILINEAR
        deconv1_x2 = tf.image.resize_bilinear(net_layers['de_fc3_rs'], [7, 14])
        net_layers['deconv1'] = self.conv(deconv1_x2, 3, 512 , 256, name= 'deconv1', strides=[1,1,1,1] ,padding='VALID', groups=1,pad_input=1)


        deconv2_x2 = tf.image.resize_bilinear(net_layers['deconv1'], [14, 28])
        net_layers['deconv2'] = self.conv(deconv2_x2, 3, 256 , 128, name= 'deconv2', strides=[1,1,1,1] ,padding='VALID', groups=1,pad_input=1)

        deconv3_x2 = tf.image.resize_bilinear(net_layers['deconv2'], [28, 56])
        net_layers['deconv3'] = self.conv(deconv3_x2, 3, 128 , 64, name= 'deconv3', strides=[1,1,1,1] ,padding='VALID', groups=1,pad_input=1)

        deconv4_x2 = tf.image.resize_bilinear(net_layers['deconv3'], [56, 112])
        net_layers['deconv4'] = self.conv(deconv4_x2, 5, 64 , 32, name= 'deconv4', strides=[1,1,1,1] ,padding='VALID', groups=1,pad_input=1, pad_num=2)

        deconv5_x2 = tf.image.resize_bilinear(net_layers['deconv4'], [112, 224])
        net_layers['deconv5'] = self.conv(deconv5_x2, 5, 32 , 16, name= 'deconv5', strides=[1,1,1,1] ,padding='VALID', groups=1,pad_input=1, pad_num=2)
        
        deconv6_x2 = tf.image.resize_bilinear(net_layers['deconv5'], [224, 448])
        net_layers['deconv6'] = tf.nn.tanh(self.conv(deconv6_x2, 5, 16 , 3, name= 'deconv6', strides=[1,1,1,1] ,padding='VALID', groups=1,pad_input=1, pad_num=2))

        net_layers['predImg'] = net_layers['deconv6']

        deconv_x2_mask = tf.image.resize_bilinear(net_layers['deconv5'], [224, 448])

        net_layers['deconv_mask'] = self.conv(deconv_x2_mask, 5, 16 , 2, name= 'deconv_mask', strides=[1,1,1,1] ,padding='VALID', groups=1,pad_input=1, pad_num=2)

        self.net_layers = net_layers


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

    def reconstruction_loss_exp(self, real_images, generated_images, mask):
        """
        The reconstruction loss is defined as the sum of the L1 distances
        between the target images and their generated counterparts
        """
        ref_exp_mask = self.get_reference_explain_mask(self.batch_size, self.spec[1][0], self.spec[1][1])
        exp_loss = self.explain_reg_weight * self.compute_exp_reg_loss(mask, ref_exp_mask)
        curr_exp = tf.nn.softmax(mask)
        curr_proj_error = tf.abs(real_images - generated_images)
        pixel_loss = tf.reduce_mean(curr_proj_error * tf.expand_dims(curr_exp[:,:,:,1], -1))
        self.masks = curr_exp[:,:,:,1]
	print('masks')
	print((self.masks).shape)
        return pixel_loss + exp_loss

    def get_reference_explain_mask(self, batch_size,height, width):
        tmp = np.array([0,1])
        ref_exp_mask = np.tile(tmp, 
                               (batch_size, 
                                height, 
                                width, 
                                1))
        ref_exp_mask = tf.constant(ref_exp_mask, dtype=tf.float32)
        return ref_exp_mask

    def reconstruction_loss(self,real_images,generated_images):
        return tf.reduce_mean(tf.abs(real_images - generated_images))
    
    def compute_exp_reg_loss(self, pred, ref):
        l = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.reshape(ref, [-1, 2]),
            logits=tf.reshape(pred, [-1, 2]))
        return tf.reduce_mean(l)

    def tvloss(generated_images):
        return tf.image.total_variation(generated_images) 

    def loss_ae(self):
	return self.reconstruction_loss(self.tgts, self.tgt_imgs)

    def loss_doafn(self):
        return self.reconstruction_loss_exp( self.tgts, self.tgt_imgs, self.net_layers['deconv_mask'])
        #explainability weighted loss with input img

    def __init__(self, batch_size, trainable, exp_weight):
        self.batch_size = batch_size
        self.trainable = trainable
        self.is_train=tf.placeholder(tf.bool, name="is_train")
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        self.tgt_imgs = tf.placeholder(tf.float32, shape = [None, 224, 448, 3], name = "tgt_imgs")
        mean = [104, 117, 123]
        scale_size = (224,448)
        self.mean = tf.constant([104, 117, 123], dtype=tf.float32)
        self.spec = [mean, scale_size]
        self.explain_reg_weight = exp_weight

        self.doafn_aspect_wide()

        self.tgts=self.net_layers['predImg']
	#self.masks = self.net_layers['deconv_mask']
        print('.......')
        print(self.tgts.get_shape())
        with tf.name_scope("loss"):
          self.loss = self.loss_doafn()


        tf.summary.scalar('loss', self.loss)

