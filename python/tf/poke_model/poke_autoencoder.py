# -*- coding: utf-8 -*-
"""
Create a customized auto-encoder with action unit at the bottomneck for learning distangled state representations.
"""
import numpy as np
import tensorflow as tf

class ConvAE(object):
    def conv_layer(self, inputs, field_size, channels_size, stride,
                   initializer_type, name, act_func=tf.nn.relu):
        inputs_shape = inputs.get_shape().as_list()
        assert inputs_shape[-1] == channels_size[0], (
            'Number of input channels does not match filter inputs channels.'
        )
        with tf.variable_scope(name):
            filter_size = field_size + channels_size
            bias_size = [channels_size[-1]]

            if initializer_type:
                initializer = tf.contrib.layers.xavier_initializer()
            else:
                initializer = tf.truncated_normal_initializer(stddev=.1)

            weights = tf.get_variable('W', filter_size, initializer=initializer)
            biases = tf.get_variable(
                'b', bias_size, initializer=tf.constant_initializer(.1))

            conv = tf.nn.conv2d(
                inputs, weights, strides=[1,stride,stride,1], padding='SAME')
            conv_bias = tf.nn.bias_add(conv, biases)
            if act_func == None:
                output = conv_bias
            else:
                output = act_func(conv_bias)

        return output

    def conv_bn_layer(self, inputs, field_size, channels_size, stride, is_training,
                      initializer_type, name, act_func=tf.nn.relu):
        inputs_shape = inputs.get_shape().as_list()
        assert inputs_shape[-1] == channels_size[0], (
            'Number of input channels does not match filter inputs channels.'
        )
        with tf.variable_scope(name):
            filter_size = field_size + channels_size
            bias_size = [channels_size[-1]]

            if initializer_type:
                initializer = tf.contrib.layers.xavier_initializer()
            else:
                initializer = tf.truncated_normal_initializer(stddev=.1)

            weights = tf.get_variable('W', filter_size, initializer=initializer)
            biases = tf.get_variable(
                'b', bias_size, initializer=tf.constant_initializer(.1))

            conv = tf.nn.conv2d(
                inputs, weights, strides=[1,stride,stride,1], padding='SAME')
            conv_bias = tf.nn.bias_add(conv, biases)
            conv_normed = tf.contrib.layers.batch_norm(conv_bias, is_training=is_training)
            output = act_func(conv_normed)

        return output

    def deconv_layer(self, inputs, field_size, channels_size,
                     initializer_type, name, act_func=tf.nn.relu):
        """
        In tf.nn.conv2d_transpose, filter has [height, width, out_c, in_c]
        in_c must match that of the inputs. In the decoder, out_c < in_c.
        This means shapes from encoder can be directly used in transposed conv.
        But in channels_size, we keep it as [in_c, out_c].
        A default stride of 1 and same padding are applied. So output feature
        map has the same height and width as the input feature map.
        Only number of channels are changed.
        For a placeholder with none in the first size, conv2d_transpose won't work.
        One can either infer it from the input if the input is not none.
        Or one can pass in a placeholder for the output shape.
        """
        batch, height, width, in_channels = inputs.get_shape().as_list()
        #shape = tf.shape(inputs)
        assert in_channels == channels_size[0], (
            'Number of input channels doe not match filter inputs channels.'
        )
        with tf.variable_scope(name):
            channels_size.reverse() # now [out_c, in_c]
            filter_size = field_size + channels_size
            bias_size = [channels_size[0]]

            if initializer_type:
                initializer = tf.contrib.layers.xavier_initializer()
            else:
                initializer = tf.truncated_normal_initializer(stddev=.1)

            weights = tf.get_variable('W', filter_size, initializer=initializer)
            biases = tf.get_variable(
                'b', bias_size, initializer=tf.constant_initializer(.1))

            #target_shape_tensor = tf.stack([shape[0], height, width, channels_size[0]])
            conv = tf.nn.conv2d_transpose(
                inputs,
                weights,
                #target_shape_tensor,
                [batch, height, width, channels_size[0]],
                [1, 1, 1, 1],
                padding='SAME')
            conv_bias = tf.nn.bias_add(conv, biases)
            output = act_func(conv_bias)
            #set_shape does not accept tensor
            #output.set_shape([batch, height, width, channels_size[0]])
            #this sets first size to none. why? Not used.
            #output = tf.reshape(output, target_shape_tensor)

        return output

    def deconv_bn_layer(self, inputs, field_size, channels_size, is_training,
                        initializer_type, name, act_func=tf.nn.relu):
        batch, height, width, in_channels = inputs.get_shape().as_list()
        assert in_channels == channels_size[0], (
            'Number of input channels doe not match filter inputs channels.'
        )
        with tf.variable_scope(name):
            channels_size.reverse() # now [out_c, in_c]
            filter_size = field_size + channels_size
            bias_size = [channels_size[0]]

            if initializer_type:
                initializer = tf.contrib.layers.xavier_initializer()
            else:
                initializer = tf.truncated_normal_initializer(stddev=.1)

            weights = tf.get_variable('W', filter_size, initializer=initializer)
            biases = tf.get_variable(
                'b', bias_size, initializer=tf.constant_initializer(.1))

            conv = tf.nn.conv2d_transpose(
                inputs,
                weights,
                [batch, height, width, channels_size[0]],
                [1, 1, 1, 1],
                padding='SAME')
            conv_bias = tf.nn.bias_add(conv, biases)
            conv_normed = tf.contrib.layers.batch_norm(conv_bias, is_training=is_training)
            output = act_func(conv_normed)

        return output

    def deconv_layer_with_stride(self, inputs, field_size, channels_size, stride,
                                 initializer_type, name, act_func=tf.nn.relu):
        """
        A stride of 2 double the size of input. (Original conv with stride 2
        downsize the image by two ignoring padding)
        """
        batch, height, width, in_channels = inputs.get_shape().as_list()
        #shape0 = tf.shape(inputs)[0]
        assert in_channels == channels_size[0], (
            'Number of input channels doe not match filter inputs channels.'
        )
        with tf.variable_scope(name):
            channels_size.reverse() # now [out_c, in_c]
            filter_size = field_size + channels_size
            bias_size = [channels_size[0]]

            if initializer_type:
                initializer = tf.contrib.layers.xavier_initializer()
            else:
                initializer = tf.truncated_normal_initializer(stddev=.1)

            weights = tf.get_variable('W', filter_size, initializer=initializer)
            biases = tf.get_variable(
                'b', bias_size, initializer=tf.constant_initializer(.1))

            #target_shape_tensor = tf.stack(
            #    [shape0, stride*height, stride*width, channels_size[0]])

            conv = tf.nn.conv2d_transpose(
                inputs,
                weights,
                #target_shape_tensor,
                [batch, stride*height, stride*width, channels_size[0]],
                [1, stride, stride, 1],
                padding='SAME')
            conv_bias = tf.nn.bias_add(conv, biases)
            output = act_func(conv_bias)
            #output.set_shape([batch, stride*height, stride*width, channels_size[0]])

        return output

    def deconv_bn_layer_with_stride(self, inputs, field_size, channels_size, stride,
                                    is_training, initializer_type, name, act_func=tf.nn.relu):
        batch, height, width, in_channels = inputs.get_shape().as_list()
        assert in_channels == channels_size[0], (
            'Number of input channels doe not match filter inputs channels.'
        )
        with tf.variable_scope(name):
            channels_size.reverse() # now [out_c, in_c]
            filter_size = field_size + channels_size
            bias_size = [channels_size[0]]

            if initializer_type:
                initializer = tf.contrib.layers.xavier_initializer()
            else:
                initializer = tf.truncated_normal_initializer(stddev=.1)

            weights = tf.get_variable('W', filter_size, initializer=initializer)
            biases = tf.get_variable(
                'b', bias_size, initializer=tf.constant_initializer(.1))

            conv = tf.nn.conv2d_transpose(
                inputs,
                weights,
                [batch, stride*height, stride*width, channels_size[0]],
                [1, stride, stride, 1],
                padding='SAME')
            conv_bias = tf.nn.bias_add(conv, biases)
            conv_normed = tf.contrib.layers.batch_norm(conv_bias, is_training=is_training)
            output = act_func(conv_normed)

        return output

    def upsample(self, inputs, stride, name, mode='ZEROS'):
        """
        Imitate reverse operation of Max-Pooling by either placing origina
        max values into a fixed postion of upsampled cell:
        [0.9] =>[[.9, 0],   (stride=2)
                [ 0, 0]]
        or copying the value into each cell:
        [0.9] =>[[.9, .9],  (stride=2)
                [ .9, .9]]
        :param inputs: 4D input tensor with [batch_size, width, heights, channels]
        :param stride:
        :param mode:
        string 'ZEROS' or 'COPY' indicating which value to use for undefined cells
        :return:  4D tensor of size
        [batch_size, width*stride, heights*stride, channels]
        If batch_size is none, reshape in _upsample_along_axis cannot work properly as well as
        conv2d_transpose. A workaround is to specify the batch size at the class init phase.
        """
        assert mode in ['COPY', 'ZEROS']
        with tf.name_scope(name):
            outputs1 = self._upsample_along_axis(inputs, 2, stride, mode=mode)
            outputs2 = self._upsample_along_axis(outputs1, 1, stride, mode=mode)
        return outputs2

    def _upsample_along_axis(self, volume, axis, stride, mode='ZEROS'):
        shape = volume.get_shape().as_list()
        #shape0 = tf.shape(volume)[0]

        assert mode in ['ZEROS', 'COPY']
        assert 0 <= axis < len(shape)
        target_shape = shape[:]
        target_shape[axis] *= stride

        #target_shape_tensor = tf.stack(
        #    [shape0, target_shape[1], target_shape[2], target_shape[3]])

        if mode == 'ZEROS':
            #padding = tf.zeros(shape, dtype=volume.dtype)
            padding = tf.zeros_like(volume, dtype=volume.dtype)
        else:
            padding = volume
        parts = [volume] + [padding for _ in range(stride-1)]
        outputs = tf.concat(parts, min(axis+1, len(shape)-1))
        outputs = tf.reshape(outputs, target_shape) #target_shape_tensor
        return outputs

    def upsize(self, inputs, stride, name):
        _, height, width, _ = inputs.get_shape().as_list()
        new_height = height*stride
        new_width = width*stride
        with tf.name_scope(name):
            volume = tf.image.resize_images(inputs, [new_height, new_width])
        return volume

    def hidden_code_sample(self, z_mean, z_log_sigma_sq, name):
        shape0 = z_mean.get_shape().as_list()
        shape1 = z_log_sigma_sq.get_shape().as_list()
        assert shape0 == shape1, (
            'mean and variance must have equal dimensions!'
        )
        with tf.variable_scope(name):
            epsilon = tf.random_normal(shape0, 0, 1, dtype=tf.float32)
            # element-wise operation
            z = tf.add(z_mean, tf.multiply(tf.sqrt(tf.exp(z_log_sigma_sq)), epsilon))
        return z

    def random_sample(self, batch_size, dim):
        epsilon = tf.random_normal([batch_size, dim])
        return epsilon

    def corrupt(self, inputs, prob):
        # 50 percents value set to zero
        with tf.name_scope('corruption'):
            corrupted = tf.multiply(inputs, tf.cast(tf.random_uniform(shape=tf.shape(inputs),
                                                                      minval=0,
                                                                      maxval=2,
                                                                      dtype=tf.int32), tf.float32))
            corrupted_input = corrupted*prob + inputs*(1-prob)
        return corrupted_input

    def fc_no_activation(self, inputs, n_output, initializer_type, name):
        with tf.variable_scope(name):
            shape = inputs.get_shape().as_list()
            assert len(shape) == 2, (
                'Input tensor has more than two dimensions'
            )
            n_input = shape[1]
            if initializer_type:
                initializer = tf.contrib.layers.xavier_initializer()
            else:
                initializer = tf.truncated_normal_initializer(stddev=.1)
            weights = tf.get_variable(
                'W', [n_input, n_output], initializer=initializer)
            biases = tf.get_variable('b', [n_output], initializer=initializer)
            weights_biases = tf.nn.bias_add(tf.matmul(inputs, weights), biases)
        return weights_biases

    def fc_layer(self, inputs, n_output, initializer_type, name, act_func=tf.nn.relu):
        with tf.variable_scope(name):
            shape = inputs.get_shape().as_list()
            assert len(shape) == 2, (
                'Input tensor has more than two dimensions.'
            )
            n_input = shape[1]
            if initializer_type:
                initializer = tf.contrib.layers.xavier_initializer()
            else:
                initializer = tf.truncated_normal_initializer(stddev=.1)
            weights = tf.get_variable(
                'W', [n_input, n_output], initializer=initializer)
            biases = tf.get_variable('b', [n_output], initializer=initializer)
            weights_biases = tf.nn.bias_add(tf.matmul(inputs, weights), biases)
            fc = act_func(weights_biases)
        return fc

    def fc_bn_layer(self, inputs, n_output, is_training,
                    initializer_type, name, act_func=tf.nn.relu):
        with tf.variable_scope(name):
            shape = inputs.get_shape().as_list()
            assert len(shape) == 2, (
                'Input tensor has more than two dimensions.'
            )
            n_input = shape[1]
            if initializer_type:
                initializer = tf.contrib.layers.xavier_initializer()
            else:
                initializer = tf.truncated_normal_initializer(stddev=.1)
            weights = tf.get_variable(
                'W', [n_input, n_output], initializer=initializer)
            biases = tf.get_variable('b', [n_output], initializer=initializer)
            weights_biases = tf.nn.bias_add(tf.matmul(inputs, weights), biases)
            normed = tf.contrib.layers.batch_norm(weights_biases, is_training=is_training)
            fc = act_func(normed)
        return fc

    def lrelu(self, inputs, leak=0.2, name='lrelu'):
        return tf.maximum(inputs, leak*inputs)

    def patch_average_error(self, image_1, image_2, height, width, center_x, center_y):
        """
        center coordinates of crop window are normalized. Height and width are int type.
        """
        size = tf.constant([height, width], dtype=tf.int32)
        offset = tf.constant([[center_x, center_y]], dtype=tf.float32)
        image_1 = tf.constant(image_1, dtype=tf.float32)
        image_2 = tf.constant(image_2, dtype=tf.float32)
        #print(image_1.get_shape().as_list(), image_2.get_shape().as_list())
        patch_1 = tf.image.extract_glimpse(image_1, size, offset, centered=False, normalized=True)
        patch_2 = tf.image.extract_glimpse(image_2, size, offset, centered=False, normalized=True)

        shape_1 = patch_1.get_shape().as_list()
        shape_2 = patch_2.get_shape().as_list()
        assert shape_1 == shape_2, (
            'Patch to compare must have the same shape'
        )
        patch_1 = tf.squeeze(patch_1)
        patch_2 = tf.squeeze(patch_2)
        mean_pixel_error = tf.reduce_mean(tf.sqrt(tf.square(patch_1-patch_2)))

        return mean_pixel_error, patch_1, patch_2

class PokeAE(ConvAE):
    """
    Auto-encoder model for distangling state representations.
    For 240X240 images, it will be troublesome to upsample.
    240 -> 120 -> 60 -> 30 -> 15 -> 8 -> 16 -> 32 ...
    From 15X15, convolution with stride 2 and 5X5 filter will lead to:
    6X6 (valid padding ceil((15-5+1)/2)=6)
    8X8 (same padding ceil(15/2)=8, SAME padding 2 to the left and right)
    """
    def __init__(self, learning_rate=1e-4, episilon=1e-7, batch_size=16,
                 split_size=128, in_channels=3, corrupted=0, is_training=True):
        self.in_channels = in_channels
        self.is_training = is_training
        self.i1 = tf.placeholder(tf.float32, [batch_size, 64, 64, in_channels], 'i1')
        self.i2 = tf.placeholder(tf.float32, [batch_size, 64, 64, in_channels], 'i2')
        self.u_labels = tf.placeholder(tf.float32, [batch_size, 4], 'u_d')
        tf.summary.image('image1', self.i1)
        tf.summary.image('image2', self.i2)
        tf.summary.histogram('u', self.u_labels)

        if corrupted:
            self.i1_corrupted = self.corrupt(self.i1, 0.05)
            tf.summary.image('image1_corrupted', self.i1_corrupted)
            self.encoding, self.feature_map_shape = self.encode(self.i1_corrupted)
        else:
            self.encoding, self.feature_map_shape = self.encode(self.i1)

        self.encoding_out = self.transit(self.encoding, self.u_labels, split_size)
        self.decoding = self.decode(self.encoding_out, self.feature_map_shape)
        tf.summary.image('image2_rec', self.decoding)

        self.loss = self.loss_function(self.i2, self.decoding)
        tf.summary.scalar('loss_rec', self.loss)

        self.train_op = self.train(self.loss, learning_rate, episilon)

        self.merged = tf.summary.merge_all()

    def encode(self, img):
        """
        224 -> 112 -> 56 -> 28 -> 14 -> 7
        conv5 is (batchX) 7X7X256
        fc layer leads to 7X7X256X1000 = 12544000 parameters.
        """
        with tf.variable_scope('encoder'):
            #conv1 = self.conv_layer(
            #    img, [5, 5], [3, 32], stride=2, initializer_type=1, name='conv1')
            #conv2 = self.conv_layer(
            #    conv1, [5, 5], [32, 32], stride=2, initializer_type=1, name='conv2')
            conv3 = self.conv_layer(
                img, [5, 5], [self.in_channels, 64], stride=2, initializer_type=1, name='conv3')
            #conv4 = self.conv_bn_layer(
            conv4 = self.conv_layer(
                conv3, [5, 5], [64, 128], stride=2, #is_training=self.is_training,
                initializer_type=1, name='conv4')
            #conv5 = self.conv_bn_layer(
            conv5 = self.conv_layer(
                conv4, [5, 5], [128, 256], stride=2, #is_training=self.is_training,
                initializer_type=1, name='conv5')
            shape = conv5.get_shape().as_list()
            feature_map_size = shape[1]*shape[2]*shape[3]
            conv5_flat = tf.reshape(
                conv5, [-1, feature_map_size], 'conv5_flat')
            #fc6 = self.fc_bn_layer(conv5_flat, 1024, is_training=self.is_training,
            fc6 = self.fc_layer(conv5_flat, 1024,
                                initializer_type=1, name='fc6')
            #fc7 = self.fc_layer(fc6, 1024, initializer_type=1, name='fc7')
        return fc6, shape

    def transit(self, code, u, split_size):
        with tf.variable_scope('representation'):
            identity, pose = tf.split(code, num_or_size_splits=[1024-split_size, split_size], axis=1)
            pose_u = tf.concat([pose, u], 1)
            #pose_transit_1 = self.fc_bn_layer(
            pose_transit_1 = self.fc_layer(
                pose_u, split_size, #is_training=self.is_training,
                initializer_type=1, name='transit_1')
            #pose_transit_2 = self.fc_bn_layer(
            pose_transit_2 = self.fc_layer(
                pose_transit_1, split_size, #is_training=self.is_training,
                initializer_type=1, name='transit_2')
            #pose_transit_3 = self.fc_layer(
            #    pose_transit_2, split_size, initializer_type=1, name='transit_3')
            code_out = tf.concat([identity, pose_transit_2], 1)
            """
            pose_transit = self.fc_layer(
                pose_u, 512, initializer_type=1, name='transit')
            code_out = tf.concat([identity, pose_transit], 1)
            """
            tf.summary.histogram('identity', identity)
            tf.summary.histogram('pose', pose)
            tf.summary.histogram('pose_transit', pose_transit_2)
        return code_out

    def decode(self, code, shape):
        with tf.variable_scope('decoder'):
            #fc8 = self.fc_layer(code, 1024, initializer_type=1, name='fc8')
            feature_map_size = shape[1]*shape[2]*shape[3]
            #fc9 = self.fc_bn_layer(code, feature_map_size, is_training=self.is_training,
            fc9 = self.fc_layer(code, feature_map_size,
                                initializer_type=1, name='fc9')
            deconv5 = tf.reshape(fc9, [shape[0], shape[1], shape[2], shape[3]])
            upsampling4 = self.upsample(
                deconv5, stride=2, name='upsampling4', mode='ZEROS')
            #deconv4 = self.deconv_bn_layer(
            deconv4 = self.deconv_layer(
                upsampling4, [5, 5], [256, 128], #is_training=self.is_training,
                initializer_type=1, name='deconv4')
            upsampling3 = self.upsample(
                deconv4, stride=2, name='upsampling3', mode='ZEROS')
            #deconv3 = self.deconv_bn_layer(
            deconv3 = self.deconv_layer(
                upsampling3, [5, 5], [128, 64], #is_training=self.is_training,
                initializer_type=1, name='deconv3')
            upsampling2 = self.upsample(
                deconv3, stride=2, name='upsampling2', mode='ZEROS')
            #deconv2 = self.deconv_bn_layer(
            deconv2 = self.deconv_layer(
                upsampling2, [5, 5], [64, self.in_channels], #is_training=self.is_training,
                initializer_type=1, name='deconv2')
            deconv1 = self.deconv_layer(deconv2, [5, 5], [self.in_channels, self.in_channels],
                                        initializer_type=1, name='deconv1')
            #upsampling1 = self.upsample(
            #    deconv2, stride=2, name='upsampling1', mode='COPY')#112x112x32
            #deconv1 = self.deconv_layer(
            #    upsampling1, [5, 5], [32, 32], initializer_type=1, name='deconv1')#112x112x32
            #upsampling0 = self.upsample(
            #    deconv1, stride=2, name='upsampling0', mode='COPY')#224x224x32
            #deconv0 = self.deconv_layer(
            #    upsampling0, [5, 5], [32, 3], initializer_type=1, name='deconv0')#224x224x3
            """
            feature_map_size = shape[1]*shape[2]*shape[3]
            fc9 = self.fc_layer(code, feature_map_size, initializer_type=1, name='fc9')
            deconv5 = tf.reshape(fc9, [shape[0], shape[1], shape[2], shape[3]])
            print deconv5.get_shape().as_list()
            deconv4 = self.deconv_layer_with_stride(deconv5, [5, 5], [256, 128], 2,
                                                    initializer_type=1, name='deconv4')
            print deconv4.get_shape().as_list()
            deconv3 = self.deconv_layer_with_stride(deconv4, [5, 5], [128, 64], 2,
                                                    initializer_type=1, name='deconv3')
            print deconv3.get_shape().as_list()
            deconv2 = self.deconv_layer_with_stride(deconv3, [5, 5], [64, 3], 2,
                                                    initializer_type=1, name='deconv2')
            print deconv2.get_shape().as_list()
            """
        return deconv1

    def loss_function(self, img, img_reconstruction):
        with tf.variable_scope('loss'):
            loss1 = tf.reduce_mean(tf.square(img-img_reconstruction))
            #loss = tf.reduce_mean(tf.reduce_sum(tf.square(img-img_reconstruction),
            #                                    [1, 2, 3]))
            #loss2 = 0.1*tf.reduce_mean(tf.square(img_before-img_reconstruction))
        return loss1#, loss2

    def train(self, loss, learning_rate, epsilon):
        with tf.variable_scope('train'):
            optimizer = tf.train.AdamOptimizer(
                learning_rate=learning_rate, epsilon=epsilon)
            train_op = optimizer.minimize(loss)
        return train_op
