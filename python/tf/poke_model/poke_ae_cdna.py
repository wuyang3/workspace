# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from poke_autoencoder import ConvAE
from batch_operation import read_data_list_ae_rnn, batch_images_actions_ae_rnn

DNA_KERN_SIZE = 5
RELU_SHIFT = 1e-12

class PokeAECDNA_t(ConvAE):
    """
    Total state size without spliting. Transform on the same scale of images.
    """
    def __init__(self, learning_rate=1e-4, epsilon=1e-7, batch_size=16,
                 split_size=128, in_channels=3, is_training=True, num_masks=4, use_bg=1):
        self.in_channels = in_channels
        self.is_training = is_training
        if use_bg:
            num_actual = num_masks + 1
        else:
            num_actual = num_masks
        self.i1 = tf.placeholder(tf.float32, [batch_size, 64, 64, in_channels], 'i1')
        self.i2 = tf.placeholder(tf.float32, [batch_size, 64, 64, in_channels], 'i2')
        self.i3 = tf.placeholder(tf.float32, [batch_size, 64, 64, in_channels], 'i3')
        self.u2 = tf.placeholder(tf.float32, [batch_size, 4], 'u2')
        self.u3 = tf.placeholder(tf.float32, [batch_size, 4], 'u3')
        self.u1 = tf.zeros([batch_size, 4], tf.float32)
        tf.summary.image('image1', self.i1)
        tf.summary.image('image2', self.i2)
        tf.summary.image('image3', self.i3)
        tf.summary.histogram('u2', self.u2)
        tf.summary.histogram('u3', self.u3)


        self.encoded, self.feature_map_shape = self.encode(self.i1)

        self.transits, self.cdna_kerns = self.Reccurent_state_transit(
            [self.u1, self.u2, self.u3], self.encoded, num_masks, kern_func=self.kernal_prepare)

        self.masks_1 = self.decode_mask(
            self.transits[0], self.feature_map_shape, num_actual, False)
        self.masks_2 = self.decode_mask(
            self.transits[1], self.feature_map_shape, num_actual, True)
        self.masks_3 = self.decode_mask(
            self.transits[2], self.feature_map_shape, num_actual, True)

        self.rec_1 = self.cdna_transform(self.i1, self.cdna_kerns[0], self.masks_1, use_bg)
        self.rec_2 = self.cdna_transform(self.i1, self.cdna_kerns[1], self.masks_2, use_bg)
        self.rec_3 = self.cdna_transform(self.i1, self.cdna_kerns[2], self.masks_3, use_bg)

        tf.summary.image('image_rec1', self.rec_1)
        tf.summary.image('image_rec2', self.rec_2)
        tf.summary.image('image_rec3', self.rec_3)

        self.loss = self.loss_function(self.i1, self.i2, self.i3,
                                       self.rec_1, self.rec_2, self.rec_3)
        tf.summary.scalar('loss_rec', self.loss)

        self.train_op = self.train(self.loss, learning_rate, epsilon)

        self.merged = tf.summary.merge_all()

    def encode(self, img):
        with tf.variable_scope('encoder'):
            conv2 = self.conv_layer(
                img, [5, 5], [self.in_channels, 32], stride=1,
                initializer_type=1, name='conv2')
            conv3 = self.conv_layer(
                conv2, [5, 5], [32, 64], stride=2, #is_training=self.is_training,
                initializer_type=1, name='conv3')
            conv4 = self.conv_layer(
                conv3, [5, 5], [64, 128], stride=2, #is_training=self.is_training,
                initializer_type=1, name='conv4')
            conv5 = self.conv_layer(
                conv4, [5, 5], [128, 256], stride=2, #is_training=self.is_training,
                initializer_type=1, name='conv5')
            shape = conv5.get_shape().as_list()
            feature_map_size = shape[1]*shape[2]*shape[3]
            conv5_flat = tf.reshape(
                conv5, [-1, feature_map_size], 'conv5_flat')
            fc6 = self.fc_layer(conv5_flat, 1024, #is_training=self.is_training,
                                initializer_type=1, name='fc6')
        return fc6, shape

    def kernal_prepare(self, state, num_masks, reuse):
        batch_size, _ = state.get_shape().as_list()
        with tf.variable_scope('kern_prep', reuse=reuse):
            cdna = self.fc_layer(state,
                                 DNA_KERN_SIZE*DNA_KERN_SIZE*num_masks,
                                 initializer_type=1,
                                 name='kern_fc1')
            cdna = self.fc_no_activation(cdna,
                                         DNA_KERN_SIZE*DNA_KERN_SIZE*num_masks,
                                         initializer_type=1,
                                         name='kern_fc2')
            cdna_kern = tf.reshape(cdna,
                                   [batch_size, DNA_KERN_SIZE, DNA_KERN_SIZE, 1, num_masks])
            cdna_kern = tf.nn.relu(cdna_kern - RELU_SHIFT) + RELU_SHIFT
            norm_factor = tf.reduce_sum(cdna_kern, [1,2,3], keep_dims=True)
            cdna_kern /= norm_factor

            cdna_kern = tf.transpose(cdna_kern, [1, 2, 0, 4, 3])
            cdna_kern = tf.reshape(cdna_kern,
                                   [DNA_KERN_SIZE, DNA_KERN_SIZE, batch_size, num_masks])
        return cdna_kern

    def Reccurent_state_transit(self, input_series, hidden_states,
                                num_masks, kern_func):
        batch_size, split_size = hidden_states.get_shape().as_list()
        transit_series = []

        with tf.variable_scope('LSTM'):
            cell_states = tf.zeros_like(hidden_states)
            init_layer1_state = tf.contrib.rnn.LSTMStateTuple(cell_states, hidden_states)
            init_layer2_state = tf.contrib.rnn.LSTMStateTuple(cell_states, cell_states)
            init_tupe_states = tuple([init_layer1_state, init_layer2_state])

            cells = [tf.contrib.rnn.LSTMCell(split_size, state_is_tuple=True),
                     tf.contrib.rnn.LSTMCell(split_size, state_is_tuple=True)]
            cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
            state_series, current_states = tf.contrib.rnn.static_rnn(
                cell, input_series, init_tupe_states)

        cdna_kerns = []
        for i, state in enumerate(state_series):
            if i == 0:
                reuse = False
            else:
                reuse = True
            cdna_kern = kern_func(state, num_masks, reuse)
            cdna_kerns.append(cdna_kern)

        return state_series, cdna_kerns

    def decode_mask(self, code, shape, num_masks, reuse):
        with tf.variable_scope('decoder', reuse=reuse):
            feature_map_size = shape[1]*shape[2]*shape[3]
            fc9 = self.fc_layer(code, feature_map_size, #is_training=self.is_training,
                                initializer_type=1, name='fc9')
            deconv5 = tf.reshape(fc9, [shape[0], shape[1], shape[2], shape[3]])
            upsampling4 = self.upsample(
                deconv5, stride=2, name='upsampling4', mode='COPY')
            deconv4 = self.deconv_layer(
                upsampling4, [5, 5], [256, 128], #is_training=self.is_training,
                initializer_type=1, name='deconv4')
            upsampling3 = self.upsample(
                deconv4, stride=2, name='upsampling3', mode='COPY')
            deconv3 = self.deconv_layer(
                upsampling3, [5, 5], [128, 64], #is_training=self.is_training,
                initializer_type=1, name='deconv3')
            upsampling2 = self.upsample(
                deconv3, stride=2, name='upsampling2', mode='COPY')
            deconv2 = self.deconv_layer(
                upsampling2, [5, 5], [64, 32], #is_training=self.is_training,
                initializer_type=1, name='deconv2')
            deconv1 = self.deconv_layer(deconv2, [5, 5], [32, num_masks],
                                        initializer_type=1, name='deconv1', act_func=None)
            batch_size, height, width, _ = deconv1.get_shape().as_list()
            masks = tf.reshape(
                tf.nn.softmax(tf.reshape(deconv1, [-1, num_masks])),
                [batch_size, height, width, num_masks]
            )
            mask_list = tf.split(masks, axis=3, num_or_size_splits=num_masks)

        return mask_list

    def cdna_transform(self, img, cdna_kern, masks_list, use_bg=1):
        with tf.name_scope('kern_mask_comp'):
            batch_size, height, width, channels = img.get_shape().as_list()
            _, _, _, num_masks = cdna_kern.get_shape().as_list()
            assert use_bg + num_masks == len(masks_list)
            if use_bg:
                output = masks_list[0]*img
                masks_list = masks_list[1:]
            else:
                output = 0

            img = tf.transpose(img, [3, 1, 2, 0])
            transformed = tf.nn.depthwise_conv2d(img, cdna_kern, [1, 1, 1, 1], 'SAME')
            transformed = tf.reshape(transformed, [channels, height, width, batch_size, num_masks])
            transformed = tf.transpose(transformed, [3, 1, 2, 0, 4])
            transformed_list = tf.unstack(transformed, axis=-1)

            for trans, mask in zip(transformed_list, masks_list):
                output+=trans*mask

        return output

    def loss_function(self, i1, i2, i3, i_rec1, i_rec2, i_rec3):
        with tf.variable_scope('loss'):
            loss = tf.reduce_mean(
                tf.square(i1 - i_rec1)+tf.square(i2 - i_rec2)+tf.square(i3 - i_rec3))
        return loss

    def train(self, loss, learning_rate, epsilon, adam=1):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.variable_scope('train'):
            if adam:
                optimizer = tf.train.AdamOptimizer(
                    learning_rate=learning_rate, epsilon=epsilon)
            else:
                optimizer = tf.train.RMSPropOptimizer(
                    learning_rate=learning_rate)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(loss)
        return train_op

def batch_ori_images_actions_rnn(input_queue, batch_size, num_threads, min_after_dequeue,
                                 type_img=1, normalized=0):
    """
    Not resizing images. Let the network resizing images itself.
    """
    image_1_file = tf.read_file(input_queue[0])
    image_2_file = tf.read_file(input_queue[1])
    image_3_file = tf.read_file(input_queue[2])
    u1 = input_queue[3]
    u2 = input_queue[4]

    if type_img:
        image_1 = tf.image.decode_jpeg(image_1_file)
        image_1_float = tf.cast(image_1, tf.float32)
        if normalized:
            image_1_float=image_1_float/255.0
        image_1_float.set_shape((240, 240, 3))

        image_2 = tf.image.decode_jpeg(image_2_file)
        image_2_float = tf.cast(image_2, tf.float32)
        if normalized:
            image_2_float=image_2_float/255.0
        image_2_resized = tf.image.resize_images(image_2_float, [64, 64])
        image_2_resized.set_shape((64, 64, 3))

        image_3 = tf.image.decode_jpeg(image_3_file)
        image_3_float = tf.cast(image_3, tf.float32)
        if normalized:
            image_3_float=image_3_float/255.0
        image_3_resized = tf.image.resize_images(image_3_float, [64, 64])
        image_3_resized.set_shape((64, 64, 3))

    else:
        image_1 = tf.image.decode_png(image_1_file, dtype=tf.uint8)
        image_1_float = tf.cast(image_1, tf.float32)
        if normalized:
            image_1_float=image_1_float/255.0
        image_1_float.set_shape((240, 240, 3))

        image_2 = tf.image.decode_png(image_2_file, dtype=tf.uint8)
        image_2_float = tf.cast(image_2, tf.float32)
        if normalized:
            image_2_float=image_2_float/255.0
        image_2_resized = tf.image.resize_images(image_2_float, [64, 64])
        image_2_resized.set_shape((64, 64, 3))

        image_3 = tf.image.decode_png(image_3_file, dtype=tf.uint8)
        image_3_float = tf.cast(image_3, tf.float32)
        if normalized:
            image_3_float=image_3_float/255.0
        image_3_resized = tf.image.resize_images(image_3_float, [64, 64])
        image_3_resized.set_shape((64, 64, 3))

    images_1, images_2, images_3, u1s, u2s = tf.train.batch(
        [image_1_float, image_2_resized, image_3_resized, u1, u2],
        batch_size=batch_size,
        num_threads=num_threads,
        capacity=min_after_dequeue+3*batch_size)
    output = (images_1, images_2, images_3, u1s, u2s)

    return output

class PokeCDNA_py(PokeAECDNA_t):
    """
    Transform on image pyramid. Use py_multi to use more than one kernal on each scale.
    """
    def __init__(self, learning_rate=1e-4, epsilon=1e-7, batch_size=16,
                 split_size=128, in_channels=3, is_training=True, num_masks=4, use_bg=1, adam=1):
        self.in_channels = in_channels
        self.is_training = is_training
        if use_bg:
            num_actual = num_masks+1
        else:
            num_actual = num_masks

        if num_masks==4:
            cdna_transform = self.cdna_transform_py
        elif num_masks==8:
            cdna_transform = self.cdna_py_multi
        elif num_masks==9:
            cdna_transform = self.cdna_py_triple
        else:
            raise ValueError('wrong number of masks')

        self.i1 = tf.placeholder(tf.float32, [batch_size, 240, 240, in_channels], 'i1')
        self.i2 = tf.placeholder(tf.float32, [batch_size, 64, 64, in_channels], 'i2')
        self.i3 = tf.placeholder(tf.float32, [batch_size, 64, 64, in_channels], 'i3')
        self.u2 = tf.placeholder(tf.float32, [batch_size, 4], 'u2')
        self.u3 = tf.placeholder(tf.float32, [batch_size, 4], 'u3')
        #self.u1 = tf.zeros([batch_size, 4], tf.float32)

        self.i1r = tf.image.resize_images(self.i1, [64, 64])
        tf.summary.image('image1', self.i1r)
        tf.summary.image('image2', self.i2)
        tf.summary.image('image3', self.i3)


        self.encoded, self.feature_map_shape = self.encode(self.i1r)

        self.transits, self.cdna_kerns = self.Reccurent_state_transit(
            [self.u2, self.u3], self.encoded, num_masks, kern_func=self.kernal_prepare)#self.u1, 

        self.masks_2 = self.decode_mask(
            self.transits[0], self.feature_map_shape, num_actual, False)
        self.masks_3 = self.decode_mask(
            self.transits[1], self.feature_map_shape, num_actual, True)
        # self.masks_3 = self.decode_mask(
        #     self.transits[2], self.feature_map_shape, num_actual, True)

        #self.decoding_1 = cdna_transform(self.i1, self.cdna_kerns[0], self.masks_1, use_bg)
        self.decoding_1 = self.i1r
        self.decoding_2 = cdna_transform(self.i1, self.cdna_kerns[0], self.masks_2, use_bg)
        self.decoding_3 = cdna_transform(self.i1, self.cdna_kerns[1], self.masks_3, use_bg)

        #tf.summary.image('image_rec1', self.decoding_1)
        tf.summary.image('image_rec2', self.decoding_2)
        tf.summary.image('image_rec3', self.decoding_3)

        # self.loss = self.loss_function(self.i1r, self.i2, self.i3,
        #                                self.decoding_1, self.decoding_2, self.decoding_3)
        self.loss = self.loss_pred(self.i2, self.i3, self.decoding_2, self.decoding_3)
        tf.summary.scalar('loss_rec', self.loss)

        self.train_op = self.train(self.loss, learning_rate, epsilon, adam=adam)

        self.merged = tf.summary.merge_all()

    def cdna_transform_py(self, img, cdna_kern, mask_list, use_bg=1):
        """
        images size 240x240. Kernel depth of one for each scale.
        """
        with tf.name_scope('kern_mask_comp'):
            batch_size, height, width, channels = img.get_shape().as_list()
            _, _, _, num_masks = cdna_kern.get_shape().as_list()
            assert num_masks == 4
            assert num_masks + use_bg == len(mask_list)

            kern_list = tf.split(cdna_kern, num_or_size_splits=4, axis=3)
            img_list = [tf.image.resize_images(img, [8, 8]),
                        tf.image.resize_images(img, [16, 16]),
                        tf.image.resize_images(img, [32, 32]),
                        tf.image.resize_images(img, [64, 64])]
            if use_bg:
                output = mask_list[0]*img_list[-1]
                mask_list = mask_list[1:]
            else:
                output = 0

            transformed_list = []
            stride_list = [8, 4, 2]
            for i, (kern, im) in enumerate(zip(kern_list, img_list)):
                transformed = tf.transpose(im, [3, 1, 2, 0])
                transformed = tf.nn.depthwise_conv2d(transformed, kern, [1,1,1,1], 'SAME')
                transformed = tf.transpose(transformed, [3, 1, 2, 0])
                print 'transformed shape', transformed.get_shape().as_list()
                if i != 3:
                    transformed = self.upsample(transformed, stride_list[i],
                                                'upsample%d'%i, mode='COPY')
                    print 'upsampled shape', transformed.get_shape().as_list()
                transformed_list.append(transformed)

            for trans, mask in zip(transformed_list, mask_list):
                output += trans*mask

        return output

    def cdna_py_multi(self, img, cdna_kern, mask_list, use_bg=1):
        """
        images size 240x240. cdna_kern has a depth dimension larger than 1 which depthwise
        convolve with each scale of the image.
        """
        with tf.name_scope('kern_mask_comp'):
            _, _, _, num_masks = cdna_kern.get_shape().as_list()
            assert num_masks == 8
            assert num_masks + use_bg == len(mask_list)

            kern_list = tf.split(cdna_kern, num_or_size_splits=[2, 2, 2, 2], axis=3)
            img_list = [tf.image.resize_images(img, [8, 8]),
                        tf.image.resize_images(img, [16, 16]),
                        tf.image.resize_images(img, [32, 32]),
                        tf.image.resize_images(img, [64, 64])]
            if use_bg:
                output = mask_list[0]*img_list[-1]
                mask_list = mask_list[1:]
            else:
                output = 0

            transformed_list = []
            stride_list = [8, 4, 2]
            for i, (kern, im) in enumerate(zip(kern_list, img_list)):
                batch_size, height, width, channels = im.get_shape().as_list()
                transformed = tf.transpose(im, [3, 1, 2, 0])
                transformed = tf.nn.depthwise_conv2d(transformed, kern, [1,1,1,1], 'SAME')
                transformed = tf.reshape(transformed, [channels, height, width, batch_size, 2])
                transformed = tf.transpose(transformed, [3, 1, 2, 0, 4])
                trans_unstacked = tf.unstack(transformed, axis=-1)
                if i != 3:
                    for item in trans_unstacked:
                        trans_unstacked_item = self.upsample(item, stride_list[i],
                                                             'upsample%d'%i, mode='COPY')
                        print 'before shape', item.get_shape().as_list()
                        print 'upsampled shape', trans_unstacked_item.get_shape().as_list()
                        transformed_list.append(trans_unstacked_item)
                else:
                    transformed_list.extend(trans_unstacked)
                    print 'convolved shape', trans_unstacked[0].get_shape().as_list()
                    print 'convolved shape', trans_unstacked[1].get_shape().as_list()
                print '\n'
            for trans, mask in zip(transformed_list, mask_list):
                output += trans*mask

        return output

    def cdna_py_triple(self, img, cdna_kern, mask_list, use_bg=1):
        """
        3 kernels and masks for each scale.
        """
        with tf.name_scope('kern_mask_comp'):
            _, _, _, num_masks = cdna_kern.get_shape().as_list()
            assert num_masks == 9
            assert num_masks + use_bg == len(mask_list)

            kern_list = tf.split(cdna_kern, num_or_size_splits=[3, 3, 3], axis=3)
            img_list = [tf.image.resize_images(img, [16, 16]),
                        tf.image.resize_images(img, [32, 32]),
                        tf.image.resize_images(img, [64, 64])]
            if use_bg:
                output = mask_list[0]*img_list[-1]
                mask_list = mask_list[1:]
            else:
                output = 0

            transformed_list = []
            stride_list = [4, 2]
            for i, (kern, im) in enumerate(zip(kern_list, img_list)):
                batch_size, height, width, channels = im.get_shape().as_list()
                transformed = tf.transpose(im, [3, 1, 2, 0])
                transformed = tf.nn.depthwise_conv2d(transformed, kern, [1,1,1,1], 'SAME')
                transformed = tf.reshape(transformed, [channels, height, width, batch_size, 3])
                transformed = tf.transpose(transformed, [3, 1, 2, 0, 4])
                trans_unstacked = tf.unstack(transformed, axis=-1)
                if i != 2:
                    for item in trans_unstacked:
                        trans_unstacked_item = self.upsample(item, stride_list[i],
                                                             'upsample%d'%i, mode='COPY')
                        print 'before shape', item.get_shape().as_list()
                        print 'upsampled shape', trans_unstacked_item.get_shape().as_list()
                        transformed_list.append(trans_unstacked_item)
                else:
                    transformed_list.extend(trans_unstacked)
                    print 'convolved shape', trans_unstacked[0].get_shape().as_list()
                    print 'convolved shape', trans_unstacked[1].get_shape().as_list()
                print '\n'
            for trans, mask in zip(transformed_list, mask_list):
                output += trans*mask

        return output

    def loss_pred(self, i2, i3, i_rec2, i_rec3):
        with tf.variable_scope('loss'):
            loss = tf.reduce_mean(
                tf.square(i2 - i_rec2)+tf.square(i3 - i_rec3))
        return loss

class PokeCDNA_py_diff(PokeAECDNA_t):
    """
    Transform on image pyramid. However, kernal size for each scale is different.
    This is the only difference from PokeCDNA_py. Use multi_diff for multi channel different
    size kernal.
    """
    def __init__(self, learning_rate=1e-4, epsilon=1e-7, batch_size=16,
                 split_size=128, in_channels=3, is_training=True, num_masks=8, use_bg=1, adam=1):
        self.in_channels = in_channels
        self.is_training = is_training
        if use_bg:
            num_actual = num_masks+1
        else:
            num_actual = num_masks
        self.i1 = tf.placeholder(tf.float32, [batch_size, 240, 240, in_channels], 'i1')
        self.i2 = tf.placeholder(tf.float32, [batch_size, 64, 64, in_channels], 'i2')
        self.i3 = tf.placeholder(tf.float32, [batch_size, 64, 64, in_channels], 'i3')
        self.u2 = tf.placeholder(tf.float32, [batch_size, 4], 'u2')
        self.u3 = tf.placeholder(tf.float32, [batch_size, 4], 'u3')
        self.u1 = tf.zeros([batch_size, 4], tf.float32)

        self.i1r = tf.image.resize_images(self.i1, [64, 64])
        tf.summary.image('image1', self.i1r)
        tf.summary.image('image2', self.i2)
        tf.summary.image('image3', self.i3)


        self.encoded, self.feature_map_shape = self.encode(self.i1r)

        self.transits, self.cdna_kerns = self.Reccurent_state_transit(
            [self.u1, self.u2, self.u3], self.encoded, num_masks,
            kern_func=self.kernal_prepare_diff)

        self.masks_1 = self.decode_mask(
            self.transits[0], self.feature_map_shape, num_actual, False)
        self.masks_2 = self.decode_mask(
            self.transits[1], self.feature_map_shape, num_actual, True)
        self.masks_3 = self.decode_mask(
            self.transits[2], self.feature_map_shape, num_actual, True)


        self.rec_1 = self.cdna_multi_diff(self.i1, self.cdna_kerns[0], self.masks_1, use_bg)
        self.rec_2 = self.cdna_multi_diff(self.i1, self.cdna_kerns[1], self.masks_2, use_bg)
        self.rec_3 = self.cdna_multi_diff(self.i1, self.cdna_kerns[2], self.masks_3, use_bg)

        tf.summary.image('image_rec1', self.rec_1)
        tf.summary.image('image_rec2', self.rec_2)
        tf.summary.image('image_rec3', self.rec_3)

        self.loss = self.loss_function(self.i1r, self.i2, self.i3,
                                       self.rec_1, self.rec_2, self.rec_3)
        tf.summary.scalar('loss_rec', self.loss)

        self.train_op = self.train(self.loss, learning_rate, epsilon, adam=adam)

        self.merged = tf.summary.merge_all()

    def kernal_prepare_diff(self, state, num_masks, reuse):
        assert num_masks == 8 or num_masks == 4
        batch_size, _ = state.get_shape().as_list()
        cdna_kern_list = []
        with tf.variable_scope('kern_prep', reuse=reuse):
            total_size = (4*4 + 4*4 + 8*8 + 8*8)
            cdna = self.fc_layer(state,
                                 total_size,
                                 initializer_type=1,
                                 name='kern_fc1')
            cdna = self.fc_no_activation(cdna,
                                         total_size,
                                         initializer_type=1,
                                         name='kern_fc2')
            cdna = tf.nn.relu(cdna - RELU_SHIFT) + RELU_SHIFT
            cdna_list = tf.split(cdna, num_or_size_splits=[4*4, 4*4, 8*8, 8*8], axis=1)
            cdna_list_new = [tf.reshape(cdna_list[0], [batch_size, 4, 4, 1]),
                             tf.reshape(cdna_list[1], [batch_size, 4, 4, 1]),
                             tf.reshape(cdna_list[2], [batch_size, 8, 8, 1]),
                             tf.reshape(cdna_list[3], [batch_size, 8, 8, 1])]
            for item in cdna_list_new:
                b, h, w, nm = item.get_shape().as_list()
                norm_factor = tf.reduce_sum(item, [1,2,3], keep_dims=True)
                item /= norm_factor
                item = tf.transpose(item, [1, 2, 0, 3])
                item.set_shape((h, w, b, nm))
                cdna_kern_list.append(item)

        return cdna_kern_list

    def cdna_multi_diff(self, img, cdna_kern_list, mask_list, use_bg=1):
        """
        Two kernals for each scale. Each scale has a different kernal size.
        """
        with tf.name_scope('kern_mask_comp'):
            assert len(cdna_kern_list)==4
            assert use_bg + len(cdna_kern_list) == len(mask_list)
            img_list = [tf.image.resize_images(img, [8, 8]),
                        tf.image.resize_images(img, [16, 16]),
                        tf.image.resize_images(img, [32, 32]),
                        tf.image.resize_images(img, [64, 64])]
            if use_bg:
                output = mask_list[0]*img_list[-1]
                mask_list = mask_list[1:]
            else:
                output = 0

            transformed_list = []
            stride_list = [8, 4, 2]
            for i, (kern, im) in enumerate(zip(cdna_kern_list, img_list)):
                batch_size, height, width, channels = im.get_shape().as_list()
                transformed = tf.transpose(im, [3, 1, 2, 0])
                transformed = tf.nn.depthwise_conv2d(transformed, kern, [1,1,1,1], 'SAME')
                transformed = tf.reshape(transformed, [channels, height, width, batch_size, 1])
                transformed = tf.transpose(transformed, [3, 1, 2, 0, 4])
                trans_unstacked = tf.unstack(transformed, axis=-1)
                assert len(trans_unstacked) == 1

                if i != 3:
                    # for item in trans_unstacked:
                    #     trans_unstacked_item = self.upsample(item, stride_list[i],
                    #                                          'upsample%d'%i, mode='COPY')
                    #     print 'before shape', item.get_shape().as_list()
                    #     print 'upsampled shape', trans_unstacked_item.get_shape().as_list()
                    #     transformed_list.append(trans_unstacked_item)
                    transformed_up = self.upsample(trans_unstacked[0], stride_list[i],
                                                   'upsample%d'%i, 'COPY')
                    print 'before shape', trans_unstacked[0].get_shape().as_list()
                    print 'upsampled shape', transformed_up.get_shape().as_list()
                    transformed_list.append(transformed_up)
                else:
                    transformed_list.extend(trans_unstacked)
                    print 'convolved shape', trans_unstacked[0].get_shape().as_list()
                    print '\n'
            for trans, mask in zip(transformed_list, mask_list):
                output += trans*mask

        return output

class PokeCDNA_diff_stack(PokeAECDNA_t):
    """
    Transform on image pyramid. However, kernal size for each scale is different.
    Moreover, kernals are convolved on one image sequentially.
    """
    def __init__(self, learning_rate=1e-4, epsilon=1e-7, batch_size=16,
                 split_size=128, in_channels=3, is_training=True, num_masks=3, use_bg=1, adam=1):
        self.in_channels = in_channels
        self.is_training = is_training
        if use_bg:
            num_actual = num_masks + 1
        else:
            num_actual = num_masks
        self.i1 = tf.placeholder(tf.float32, [batch_size, 240, 240, in_channels], 'i1')
        self.i2 = tf.placeholder(tf.float32, [batch_size, 64, 64, in_channels], 'i2')
        self.i3 = tf.placeholder(tf.float32, [batch_size, 64, 64, in_channels], 'i3')
        self.u2 = tf.placeholder(tf.float32, [batch_size, 4], 'u2')
        self.u3 = tf.placeholder(tf.float32, [batch_size, 4], 'u3')
        self.u1 = tf.zeros([batch_size, 4], tf.float32)

        self.i1r = tf.image.resize_images(self.i1, [64, 64])
        tf.summary.image('image1', self.i1r)
        tf.summary.image('image2', self.i2)
        tf.summary.image('image3', self.i3)


        self.encoded, self.feature_map_shape = self.encode(self.i1r)

        self.transits, self.cdna_kerns = self.Reccurent_state_transit(
            [self.u1, self.u2, self.u3], self.encoded, num_masks, kern_func=self.kernal_diff_stack)

        self.masks_1 = self.decode_mask(
            self.transits[0], self.feature_map_shape, num_actual, False)
        self.masks_2 = self.decode_mask(
            self.transits[1], self.feature_map_shape, num_actual, True)
        self.masks_3 = self.decode_mask(
            self.transits[2], self.feature_map_shape, num_actual, True)


        self.rec_1 = self.cdna_diff_stack(self.i1, self.cdna_kerns[0], self.masks_1, use_bg)
        self.rec_2 = self.cdna_diff_stack(self.i1, self.cdna_kerns[1], self.masks_2, use_bg)
        self.rec_3 = self.cdna_diff_stack(self.i1, self.cdna_kerns[2], self.masks_3, use_bg)

        tf.summary.image('image_rec1', self.rec_1)
        tf.summary.image('image_rec2', self.rec_2)
        tf.summary.image('image_rec3', self.rec_3)

        self.loss = self.loss_function(self.i1r, self.i2, self.i3,
                                       self.rec_1, self.rec_2, self.rec_3)
        tf.summary.scalar('loss_rec', self.loss)

        self.train_op = self.train(self.loss, learning_rate, epsilon, adam=adam)

        self.merged = tf.summary.merge_all()

    def kernal_diff_stack(self, state, num_masks, reuse):
        assert num_masks == 4
        batch_size, _ = state.get_shape().as_list()
        cdna_kern_list = []
        with tf.variable_scope('kern_prep', reuse=reuse):
            total_size = (4*4 + 8*8 + 8*8)*2
            cdna = self.fc_layer(state,
                                 total_size,
                                 initializer_type=1,
                                 name='kern_fc1')
            cdna = self.fc_no_activation(cdna,
                                         total_size,
                                         initializer_type=1,
                                         name='kern_fc2')
            cdna = tf.nn.relu(cdna - RELU_SHIFT) + RELU_SHIFT
            cdna_list = tf.split(cdna, num_or_size_splits=[4*4*2, 8*8*2, 8*8*2], axis=1)
            cdna_list_new = [tf.reshape(cdna_list[0], [batch_size, 4, 4, 1, 2]),
                             tf.reshape(cdna_list[1], [batch_size, 8, 8, 1, 2]),
                             tf.reshape(cdna_list[2], [batch_size, 8, 8, 1, 2])]
            for item in cdna_list_new:
                b, h, w, _, nm = item.get_shape().as_list()
                norm_factor = tf.reduce_sum(item, [1,2,3], keep_dims=True)
                item /= norm_factor
                item = tf.transpose(item, [1, 2, 0, 4, 3])
                item = tf.reshape(item, [h, w, b, nm])
                cdna_kern_list.append(item)

        return cdna_kern_list

    def cdna_diff_stack(self, img, cdna_kern_list, mask_list, use_bg=1):
        """
        cdna_kern_list: [[4,4,b,2],[4,4,b,2],[8,8,b,2],[8,8,b,2]]
        Two kernals for each scale. Two kernals are applied sequentially.
        Each scale has a different kernal size.
        """
        with tf.name_scope('kern_mask_comp'):
            assert len(cdna_kern_list)==4
            assert len(cdna_kern_list) + use_bg == len(mask_list)
            img_list = [tf.image.resize_images(img, [16, 16]),
                        tf.image.resize_images(img, [32, 32]),
                        tf.image.resize_images(img, [64, 64])]
            if use_bg:
                output = mask_list[0]*img_list[-1]
                mask_list = mask_list[1:]
            else:
                output = 0

            transformed_list = []
            stride_list = [4, 2]
            for i, (kern, im) in enumerate(zip(cdna_kern_list, img_list)):
                batch_size, height, width, channels = im.get_shape().as_list()
                transformed = tf.transpose(im, [3, 1, 2, 0])

                kern1, kern2 = tf.split(kern, num_or_size_splits=2, axis=3)

                transformed = tf.nn.depthwise_conv2d(transformed, kern1, [1,1,1,1], 'SAME')
                transformed = tf.nn.depthwise_conv2d(transformed, kern2, [1,1,1,1], 'SAME')

                transformed = tf.transpose(transformed, [3, 1, 2, 0])

                if i != 2:
                    trans_upsampled = self.upsample(transformed, stride_list[i],
                                                    'unsample%d'%i, mode='COPY')
                    print 'before shape', transformed.get_shape().as_list()
                    print 'upsampled shape', trans_upsampled.get_shape().as_list()
                    transformed_list.append(trans_upsampled)
                else:
                    transformed_list.append(transformed)
                    print 'convolved shape', transformed.get_shape().as_list()
                    print '\n'
            for trans, mask in zip(transformed_list, mask_list):
                output += trans*mask

        return output

def ae_cdna_train():
    tf.reset_default_graph()
    shuffle = True
    num_epochs = 12
    batch_size = 16
    num_threads = 4
    min_after_dequeue = 128

    type_img = 1 #1\0
    split_size = 512
    in_channels = 3 #3\1

    input_queue = read_data_list_ae_rnn(
        '../../poke/train_cube_table_rnn.txt', num_epochs, shuffle)

    images_1, images_2, images_3, u1s, u2s = batch_images_actions_ae_rnn(
       input_queue, batch_size, num_threads, min_after_dequeue, type_img=type_img,
       normalized=1)
    # images_1, images_2, images_3, u1s, u2s = batch_ori_images_actions_rnn(
    #     input_queue, batch_size, num_threads, min_after_dequeue, type_img=type_img,
    #     normalized=1)

    # Initial training parameters.
    learning_rate = 0.0001
    epsilon = 1e-08
    step = 0

    with tf.Session() as sess:
        poke_ae = PokeAECDNA_t(learning_rate, epsilon, batch_size, split_size,
                               in_channels=in_channels, is_training=True,
                               num_masks=2, use_bg=1)
        # poke_ae = PokeCDNA_py(learning_rate, epsilon, batch_size, split_size,
        #                       in_channels=in_channels, is_training=True,
        #                       num_masks=9, use_bg=0, adam=1) # or 8, 4
        # poke_ae = PokeCDNA_py_diff(learning_rate, epsilon, batch_size, split_size,
        #                           in_channels=in_channels, is_training=True,
        #                            num_masks=4, use_bg=1, adam=1)

        saver = tf.train.Saver(max_to_keep=2)
        train_writer = tf.summary.FileWriter('../logs/pokeAECDNA/', sess.graph)

        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            while not coord.should_stop():
                i1, i2, i3, u1, u2 = sess.run([images_1, images_2, images_3, u1s, u2s])

                feed_dict = {poke_ae.i1: i1, poke_ae.i2: i2, poke_ae.i3: i3,
                             poke_ae.u2: u1, poke_ae.u3: u2}

                _, loss, summary = sess.run(
                    [poke_ae.train_op, poke_ae.loss, poke_ae.merged],
                    feed_dict=feed_dict)

                if step%10 == 0:
                    train_writer.add_summary(summary, step)
                    print('step %d: loss->%.4f' %(step, loss))

                if step%1000 == 0:
                    saver.save(sess, '../logs/pokeAECDNA/', global_step=step)
                step+=1

            train_writer.close()
        except tf.errors.OutOfRangeError:
            print('Done queuing: epoch limit reached.')

        finally:
            coord.request_stop()
            coord.join(threads)
            print('steps of batch enqueuing is %d'%step)

if __name__ == '__main__':
    ae_cdna_train()
