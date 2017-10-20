# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from poke_autoencoder import ConvAE
from poke_ae_cdna import batch_ori_images_actions_rnn
from batch_operation import read_data_list_ae_rnn, batch_images_actions_ae_rnn
from spatial_transformer import transformer

class PokeAESTP(ConvAE):
    def __init__(self, learning_rate=1e-4, epsilon=1e-7, batch_size=16,
                 split_size=128, in_channels=3, is_training=True, num_masks=4, use_bg=0):
        self.in_channels = in_channels
        self.is_training = is_training
        if use_bg:
            num_actual = num_masks+1
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


        self.encoded, self.feature_map_shape = self.encode(self.i1, split_size)

        self.transits, self.stp_params = self.Reccurent_state_transit(
            [self.u1, self.u2, self.u3], self.encoded, num_masks)

        self.masks_1 = self.decode_mask(
            self.transits[0], self.feature_map_shape, num_actual, False)
        self.masks_2 = self.decode_mask(
            self.transits[1], self.feature_map_shape, num_actual, True)
        self.masks_3 = self.decode_mask(
            self.transits[2], self.feature_map_shape, num_actual, True)

        self.rec_1 = self.stp_transform(self.i1, self.stp_params[0], self.masks_1, use_bg=use_bg)
        self.rec_2 = self.stp_transform(self.i1, self.stp_params[1], self.masks_2, use_bg=use_bg)
        self.rec_3 = self.stp_transform(self.i1, self.stp_params[2], self.masks_3, use_bg=use_bg)

        tf.summary.image('image_rec1', self.rec_1)
        tf.summary.image('image_rec2', self.rec_2)
        tf.summary.image('image_rec3', self.rec_3)

        self.loss = self.loss_function(self.i1, self.i2, self.i3,
                                       self.rec_1, self.rec_2, self.rec_3)
        tf.summary.scalar('loss_rec', self.loss)

        self.train_op = self.train(self.loss, learning_rate, epsilon)

        self.merged = tf.summary.merge_all()

    def encode(self, img, split_size):
        # originally: 3->16->32->64->128->256.
        with tf.variable_scope('encoder'):
            conv2 = self.conv_layer(
                img, [5, 5], [self.in_channels, 16], stride=1,
                initializer_type=1, name='conv2')
            conv3 = self.conv_bn_layer(
                conv2, [5, 5], [16, 32], stride=2, is_training=self.is_training,
                initializer_type=1, name='conv3')
            conv4 = self.conv_bn_layer(
                conv3, [5, 5], [32, 64], stride=2, is_training=self.is_training,
                initializer_type=1, name='conv4')
            conv5 = self.conv_bn_layer(
                conv4, [5, 5], [64, 64], stride=2, is_training=self.is_training,
                initializer_type=1, name='conv5')
            shape = conv5.get_shape().as_list()
            feature_map_size = shape[1]*shape[2]*shape[3]
            conv5_flat = tf.reshape(
                conv5, [-1, feature_map_size], 'conv5_flat')
            fc6 = self.fc_bn_layer(conv5_flat, 512, is_training=self.is_training,
                                   initializer_type=1, name='fc6')
        return fc6, shape

    def stp_prepare(self, state, num_masks, u, reuse):
        stp_param_list = []
        with tf.variable_scope('stp_prep', reuse=reuse):
            state_concat = tf.concat([state, u], axis=1)
            state_fc = self.fc_layer(state_concat, 100, initializer_type=1, name='stp_fc')
            identity_params = tf.convert_to_tensor(
                np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], np.float32))
            for i in range(num_masks):
                params = self.fc_no_activation(
                    state_fc, 6, initializer_type=1, name='stp_params%d'%i) + identity_params
                stp_param_list.append(params)

        return stp_param_list

    def Reccurent_state_transit(self, input_series, hidden_states, num_masks):
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

        stp_params = []
        for i, state in enumerate(state_series):
            if i == 0:
                reuse = False
            else:
                reuse = True
            stp_param_list = self.stp_prepare(state, num_masks, input_series[i], reuse)
            stp_params.append(stp_param_list)

        return state_series, stp_params

    def stp_transform(self, img, stp_param_list, masks_list, use_bg=0):
        print 'len stp param %d, len mask list %d'%(len(stp_param_list), len(masks_list))
        assert use_bg + len(stp_param_list) == len(masks_list)
        if use_bg:
            output = img*masks_list[0]
            masks_list = masks_list[1:]
            print 'mask background'
        else:
            output = 0
            print 'not mask background'

        with tf.name_scope('stp_mask_comp'):
            transformed_list = []
            for stp_param in stp_param_list:
                transformed = transformer(img, stp_param, (64, 64))
                transformed_list.append(transformed)
            for trans, mask in zip(transformed_list, masks_list):
                output += trans*mask

        return output

    def decode_mask(self, code, shape, num_masks, reuse):
        # originally, 256->128->64->32->3.
        print 'decoding %d masks'%num_masks
        with tf.variable_scope('decoder', reuse=reuse):
            feature_map_size = shape[1]*shape[2]*shape[3]
            fc9 = self.fc_bn_layer(code, feature_map_size, is_training=self.is_training,
                                initializer_type=1, name='fc9')
            deconv5 = tf.reshape(fc9, [shape[0], shape[1], shape[2], shape[3]])
            upsampling4 = self.upsample(
                deconv5, stride=2, name='upsampling4', mode='ZEROS')
            deconv4 = self.deconv_bn_layer(
                upsampling4, [5, 5], [64, 64], is_training=self.is_training,
                initializer_type=1, name='deconv4')
            upsampling3 = self.upsample(
                deconv4, stride=2, name='upsampling3', mode='ZEROS')
            deconv3 = self.deconv_bn_layer(
                upsampling3, [5, 5], [64, 32], is_training=self.is_training,
                initializer_type=1, name='deconv3')
            upsampling2 = self.upsample(
                deconv3, stride=2, name='upsampling2', mode='ZEROS')
            deconv2 = self.deconv_bn_layer(
                upsampling2, [5, 5], [32, 16], is_training=self.is_training,
                initializer_type=1, name='deconv2')
            deconv1 = self.deconv_layer(deconv2, [5, 5], [16, num_masks],
                                        initializer_type=1, name='deconv1', act_func=None)
            batch_size, height, width, _ = deconv1.get_shape().as_list()
            masks = tf.reshape(
                tf.nn.softmax(tf.reshape(deconv1, [-1, num_masks])),
                [batch_size, height, width, num_masks]
            )
            mask_list = tf.split(masks, axis=3, num_or_size_splits=num_masks)

        return mask_list

    def loss_function(self, i1, i2, i3, i_rec1, i_rec2, i_rec3):
        with tf.variable_scope('loss'):
            loss = tf.reduce_mean(
                tf.square(i1 - i_rec1)+tf.square(i2 - i_rec2)+tf.square(i3 - i_rec3))
        return loss

    def train(self, loss, learning_rate, epsilon):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.variable_scope('train'):
            optimizer = tf.train.AdamOptimizer(
                learning_rate=learning_rate, epsilon=epsilon)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(loss)
        return train_op

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

class PokeAESTP_py(PokeAESTP):
    def __init__(self, learning_rate=1e-4, epsilon=1e-7, batch_size=16,
                 split_size=128, in_channels=3, is_training=True,
                 num_masks=6, upsample=1, use_bg=0):
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
        tf.summary.image('image1', self.i1)
        tf.summary.image('image2', self.i2)
        tf.summary.image('image3', self.i3)
        tf.summary.histogram('u2', self.u2)
        tf.summary.histogram('u3', self.u3)


        self.i1r = tf.image.resize_images(self.i1, [64, 64])
        self.encoded, self.feature_map_shape = self.encode(self.i1r, split_size)

        self.transits, self.stp_params = self.Reccurent_state_transit(
            [self.u1, self.u2, self.u3], self.encoded, num_masks)

        self.masks_1 = self.decode_mask(
            self.transits[0], self.feature_map_shape, num_actual, False)
        self.masks_2 = self.decode_mask(
            self.transits[1], self.feature_map_shape, num_actual, True)
        self.masks_3 = self.decode_mask(
            self.transits[2], self.feature_map_shape, num_actual, True)

        self.decoding_1 = self.stp_scale_transform(
            self.i1, self.stp_params[0], self.masks_1, upsample, use_bg)
        self.decoding_2 = self.stp_scale_transform(
            self.i1, self.stp_params[1], self.masks_2, upsample, use_bg)
        self.decoding_3 = self.stp_scale_transform(
            self.i1, self.stp_params[2], self.masks_3, upsample, use_bg)

        tf.summary.image('image_rec1', self.decoding_1)
        tf.summary.image('image_rec2', self.decoding_2)
        tf.summary.image('image_rec3', self.decoding_3)

        self.loss = self.loss_function(self.i1r, self.i2, self.i3,
                                       self.decoding_1, self.decoding_2, self.decoding_3)
        tf.summary.scalar('loss_rec', self.loss)

        self.train_op = self.train(self.loss, learning_rate, epsilon)

        self.merged = tf.summary.merge_all()

    def stp_scale_transform(self, img, stp_param_list, masks_list, upsample=1, use_bg=0):
        print 'len stp param %d, len mask list %d'%(len(stp_param_list), len(masks_list))
        assert use_bg + len(stp_param_list) == len(masks_list)

        img_l = [tf.image.resize_images(img, [16, 16]),
                 tf.image.resize_images(img, [32, 32]),
                 tf.image.resize_images(img, [64, 64])]

        if use_bg:
            output = img_l[-1]*masks_list[0]
            masks_list = masks_list[1:]
            print 'mask background'
        else:
            output = 0
            print 'not mask background'

        img_list = [item for item in img_l for _ in range(2)]

        stride_list = [4, 4, 2, 2]
        transformed_list = []
        with tf.name_scope('stp_mask_comp'):
            for i, (stp_param, img) in enumerate(zip(stp_param_list, img_list)):
                b, h, w, c = img.get_shape().as_list()

                if upsample:
                    transformed = transformer(img, stp_param, (h, w))
                    transformed.set_shape((b, h, w, c))
                    if i < 4:
                        transformed = self.upsample(
                            transformed, stride_list[i], 'upsample%d'%i, mode='COPY')
                        print 'upsampled %d'%(i+1)
                else:
                    transformed = transformer(img, stp_param, (64, 64))
                    transformed.set_shape((b, 64, 64, c))

                transformed_list.append(transformed)

            for trans, mask in zip(transformed_list, masks_list):
                output += trans*mask

        return output

def ae_stp_train():
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

    # images_1, images_2, images_3, u1s, u2s = batch_images_actions_ae_rnn(
    #     input_queue, batch_size, num_threads, min_after_dequeue, type_img=type_img,
    #     normalized=1)
    images_1, images_2, images_3, u1s, u2s = batch_ori_images_actions_rnn(
       input_queue, batch_size, num_threads, min_after_dequeue, type_img=type_img,
       normalized=1)

    # Initial training parameters.
    learning_rate = 0.0001
    epsilon = 1e-05
    step = 0

    with tf.Session() as sess:
        # poke_ae = PokeAESTP(learning_rate, epsilon, batch_size, split_size,
        #                     in_channels=in_channels, is_training=True, num_masks=8, use_bg=1)
        poke_ae = PokeAESTP_py(learning_rate, epsilon, batch_size, split_size,
                               in_channels=in_channels, is_training=True,
                               num_masks=6, upsample=1, use_bg=1)

        saver = tf.train.Saver(max_to_keep=2)
        train_writer = tf.summary.FileWriter('../logs/pokeAESTP/', sess.graph)

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
                    saver.save(sess, '../logs/pokeAESTP/', global_step=step)
                step+=1

            train_writer.close()
        except tf.errors.OutOfRangeError:
            print('Done queuing: epoch limit reached.')

        finally:
            coord.request_stop()
            coord.join(threads)
            print('steps of batch enqueuing is %d'%step)

if __name__ == '__main__':
    ae_stp_train()
