# -*- coding: utf-8 -*-
"""
Create a Siamese CNN with subsequent fc layers. From the Siamese network,
an inverse and a forward model is built. All inputs, states, predictions and
loss are initialized as the model's attributes.

@author: wuyang
"""
import tensorflow as tf
import numpy as np

class Poke(object):
    """
    Model of Siamese network for predicting the poke action.
    """
    def __init__(self, train_layers=['inverse','forward','siamese'],
                 learning_rate=0.001, epsilon=1e-08, lamb=0.1, include_type = 0):
        self.include_type = include_type
        self.i1 = tf.placeholder(tf.float32, [None, 227, 227, 3], 'i1')
        self.i2 = tf.placeholder(tf.float32, [None, 227, 227, 3], 'i2')
        self.u = tf.placeholder(tf.float32, [None, 4], 'u_c')
        self.p_labels = tf.placeholder(tf.int32, [None], 'p_d') # discrete
        self.theta_labels = tf.placeholder(tf.int32, [None], 'theta_d')
        self.l_labels = tf.placeholder(tf.int32, [None], 'l_d')

        #tf.summary.image('image1', self.i1)
        #tf.summary.image('image2', self.i2)
        tf.summary.histogram('u', self.u)
        tf.summary.histogram('p', self.p_labels)
        tf.summary.histogram('theta', self.theta_labels)
        tf.summary.histogram('l', self.l_labels)

        if self.include_type:
            self.type_labels = tf.placeholder(tf.int32, [None], 'type_d')
            self.labels = (self.p_labels, self.theta_labels, self.l_labels, self.type_labels)
            tf.summary.histogram('type', self.type_labels)
        else:
            self.labels = (self.p_labels, self.theta_labels, self.l_labels)

        with tf.variable_scope("siamese"):
            x1_intmd = self.siamese(self.i1)
            #self.x1 = self.fc_layer(x1_intmd, 200, 'x')
            self.x1 = tf.nn.relu(self.fc_layer(x1_intmd, 200, 'x'))
            tf.summary.histogram('states', self.x1)

        with tf.variable_scope("siamese", reuse=True):
            x2_intmd = self.siamese(self.i2)
            #self.x2 = self.fc_layer(x2_intmd, 200, 'x')
            self.x2 = tf.nn.relu(self.fc_layer(x2_intmd, 200, 'x'))
            tf.summary.histogram('states', self.x2)

        self.raw_predictions = self.inverse(self.x1, self.x2)
        self.classes = self.prediction(self.raw_predictions)

        self.x_pred = self.forward(self.x1, self.u)

        self.loss_i = self.loss_inverse(self.labels, self.raw_predictions)
        self.loss_f = self.loss_forward(self.x2, self.x_pred)
        self.loss = self.loss_joint(self.loss_i, self.loss_f, lamb)

        self.train_op = self.train(train_layers, learning_rate, epsilon, self.loss)

        self.merged = tf.summary.merge_all()

    def siamese(self,i):
        # Alexnet weights preparartion
        variable_data = np.load(
            "/home/wuyang/workspace/python/tf/alex/bvlc_alexnet.npy",
            encoding='bytes').item()
        conv1_preW = tf.constant(variable_data["conv1"][0])
        conv1_preb = tf.constant(variable_data["conv1"][1])
        conv2_preW = tf.constant(variable_data["conv2"][0])
        conv2_preb = tf.constant(variable_data["conv2"][1])
        conv3_preW = tf.constant(variable_data["conv3"][0])
        conv3_preb = tf.constant(variable_data["conv3"][1])
        conv4_preW = tf.constant(variable_data["conv4"][0])
        conv4_preb = tf.constant(variable_data["conv4"][1])
        conv5_preW = tf.constant(variable_data["conv5"][0])
        conv5_preb = tf.constant(variable_data["conv5"][1])

        # Convolution 1
        # Keep in mind that you need to use tf.get_variable. you can use
        # tf.constant on the numpy initial value and then pass the tensor as
        # initializer. Use tf.variable_scope since tf.get_variable method will
        # ignore tf.name_scope
        with tf.variable_scope('conv1') as scope:
            kernel = tf.get_variable('weights', initializer=conv1_preW)
            biases = tf.get_variable('biases', initializer=conv1_preb)
            conv = tf.nn.conv2d(i, kernel, [1, 4, 4, 1], padding="SAME")
            conv_bias = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(conv_bias)
        # Local response normalization
        lrn1 = tf.nn.local_response_normalization(
            conv1, depth_radius=2, alpha=2e-5, beta=0.75, bias=1.0)
        # Maxpool 1
        pool1 = tf.nn.max_pool(lrn1,
                               ksize=[1, 3, 3, 1],
                               strides=[1, 2, 2, 1],
                               padding='VALID',
                               name='pool1')

        # Convolution 2
        with tf.variable_scope('conv2') as scope:
            kernel = tf.get_variable('weights', initializer=conv2_preW)
            biases = tf.get_variable('biases', initializer=conv2_preb)

            input_a, input_b = tf.split(value=pool1,
                                        num_or_size_splits=2,
                                        axis=3)
            kernel_a, kernel_b = tf.split(value=kernel,
                                          num_or_size_splits=2,
                                          axis=3)
            with tf.variable_scope('A'):
                conv_a = tf.nn.conv2d(
                    input_a, kernel_a, [1, 1, 1, 1], padding="SAME")
            with tf.variable_scope('B'):
                conv_b = tf.nn.conv2d(
                    input_b, kernel_b, [1, 1, 1, 1], padding="SAME")

            conv = tf.concat([conv_a, conv_b],3)
            conv_bias = tf.nn.bias_add(conv, biases)
            conv2 = tf.nn.relu(conv_bias)
        # Local response normalization 2
        lrn2 = tf.nn.local_response_normalization(
            conv2, depth_radius=2, alpha=2e-05, beta=0.75, bias=1.0)
        # Maxpool 2
        pool2 = tf.nn.max_pool(lrn2,
                               ksize=[1, 3, 3, 1],
                               strides=[1, 2, 2, 1],
                               padding='VALID',
                               name='pool2')

        with tf.variable_scope('conv3') as scope:
            kernel = tf.get_variable('weights', initializer=conv3_preW)
            biases = tf.get_variable('biases', initializer=conv3_preb)
            conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
            conv_bias = tf.nn.bias_add(conv, biases)
            conv3 = tf.nn.relu(conv_bias)

        with tf.variable_scope('conv4') as scope:
            kernel = tf.get_variable('weights', initializer=conv4_preW)
            biases = tf.get_variable('biases', initializer=conv4_preb)

            input_a, input_b = tf.split(value=conv3,
                                        num_or_size_splits=2,
                                        axis=3)
            kernel_a, kernel_b = tf.split(value=kernel,
                                          num_or_size_splits=2,
                                          axis=3)
            with tf.variable_scope('A'):
                conv_a = tf.nn.conv2d(
                    input_a, kernel_a, [1, 1, 1, 1], padding="SAME")
            with tf.variable_scope('B'):
                conv_b = tf.nn.conv2d(
                    input_b, kernel_b, [1, 1, 1, 1], padding="SAME")
            conv = tf.concat([conv_a, conv_b],3)
            conv_bias = tf.nn.bias_add(conv, biases)
            conv4 = tf.nn.relu(conv_bias)

        with tf.variable_scope('conv5') as scope:
            kernel = tf.get_variable('weights', initializer=conv5_preW)
            biases = tf.get_variable('biases', initializer=conv5_preb)

            input_a, input_b = tf.split(value = conv4,
                                        num_or_size_splits=2,
                                        axis=3)
            kernel_a, kernel_b = tf.split(value = kernel,
                                          num_or_size_splits=2,
                                          axis=3)
            with tf.variable_scope('A'):
                conv_a = tf.nn.conv2d(
                    input_a, kernel_a, [1, 1, 1, 1], padding="SAME")
            with tf.variable_scope('B'):
                conv_b = tf.nn.conv2d(
                    input_b, kernel_b, [1, 1, 1, 1], padding="SAME")
            conv = tf.concat([conv_a, conv_b],3)
            conv_bias = tf.nn.bias_add(conv, biases)
            conv5 = tf.nn.relu(conv_bias)
        # Maxpool 3 (at layer 5)
        pool5 = tf.nn.max_pool(conv5,
                               ksize=[1, 3, 3, 1],
                               strides=[1, 2, 2, 1],
                               padding='VALID',
                               name='pool5')
        size = pool5.get_shape()[1].value * pool5.get_shape()[2].value * \
               pool5.get_shape()[3].value
        activations = tf.reshape(pool5, [-1, size], name='x_flattened')
        return activations

    # For modeling poke distribution, poke location, and angle and length are
    # discretized. u_pred is continuous, u are discretized into 20*20, 36 and 11
    # classes as targets.
    def inverse(self, x1, x2):
        with tf.variable_scope('inverse'):
            x = tf.concat([x1, x2], 1, name='x_concat')
            x_hidden = tf.nn.relu(self.fc_layer(x, 200, 'x_hidden')) # Crucial

            with tf.variable_scope('l'):
                l_hidden = self.fc_layer(x_hidden,  200, 'l_hidden')
                l_relu = tf.nn.relu(l_hidden, 'l_ReLU')
                #l_relu_2 = tf.nn.relu(self.fc_layer(l_relu, 200, 'l_hidden_2'))
                l_pred = self.fc_layer(l_relu, 10, 'l_pred') # 11 or 10?
                tf.summary.histogram('l_prediction', l_pred)

            xl = tf.concat([x_hidden, l_pred], 1)
            with tf.variable_scope('theta'):
                theta_hidden = self.fc_layer(xl, 200, 'theta_hidden')
                theta_relu = tf.nn.relu(theta_hidden, 'theta_ReLU')
                #theta_relu_2 = tf.nn.relu(self.fc_layer(theta_relu, 200, 'theta_hidden_2'))
                theta_pred = self.fc_layer(theta_relu, 36, 'theta_pred')
                tf.summary.histogram('theta_prediction', theta_pred)

            xt = tf.concat([x_hidden, theta_pred], 1)
            with tf.variable_scope('p'):
                p_hidden = self.fc_layer(xt, 400, 'p_hidden')
                p_relu = tf.nn.relu(p_hidden, 'p_ReLU')
                #p_relu_2 = tf.nn.relu(self.fc_layer(p_relu, 400, 'p_hidden_2'))
                p_pred = self.fc_layer(p_relu, 400, 'p_pred')
                tf.summary.histogram('p_prediction', p_pred)

            # check better configurations.
            if self.include_type:
                with tf.variable_scope('type'):
                    type_hidden = self.fc_layer(x_hidden, 100, 'type_hidden')
                    type_relu = tf.nn.relu(type_hidden, 'type_ReLU')
                    type_pred = self.fc_layer(type_relu, 3, 'type_pred')
                    tf.summary.histogram('type_prediction', type_pred)
                outputs = (p_pred, theta_pred, l_pred, type_pred)

            else:
                outputs = (p_pred, theta_pred, l_pred)
        return outputs

    # In forward model, real vectors of actions are used without discretization.
    # u here is the real valued input action.
    def forward(self, x1, u):
        with tf.variable_scope('forward'):
            xu = tf.concat([x1, u], 1)
            x_hidden_1 = self.fc_layer(xu, 200, 'x_hidden_1')
            x_relu_1 = tf.nn.relu(x_hidden_1, 'x_ReLU_1')
            x_hidden_2 = self.fc_layer(x_relu_1, 200, 'x_hidden_2')
            #x_relu_2 = tf.nn.relu(x_hidden_2, 'x_ReLU_2')
            #x_pred = tf.nn.relu(self.fc_layer(x_relu_2, 200, 'x_pred'))
            x_pred = tf.nn.relu(x_hidden_2, 'x_pred')
            tf.summary.histogram('x_prediction', x_pred)
        return x_pred

    def fc_layer(self, input_tensor, n_weight, name):
        with tf.variable_scope(name):
            assert len(input_tensor.get_shape()) == 2, (
                'Input tensor has more than two dimensions.'
            )
            n_prev_weight = input_tensor.get_shape()[1].value
            #initializer = tf.truncated_normal_initializer(stddev=0.01)
            initializer = tf.contrib.layers.xavier_initializer()
            w = tf.get_variable('Weigths',
                                shape=[n_prev_weight, n_weight],
                                dtype=tf.float32,
                                initializer=initializer)
            b = tf.get_variable('biases',
                                dtype=tf.float32,
                                initializer=tf.constant(
                                    0.1, shape=[n_weight], dtype=tf.float32))
            fc = tf.nn.bias_add(tf.matmul(input_tensor, w), b) # name='fc'
        return fc

    def prediction(self, inputs):
        p_classes = tf.argmax(inputs[0], axis=1)
        theta_classes = tf.argmax(inputs[1], axis=1)
        l_classes = tf.argmax(inputs[2], axis=1)
        tf.summary.histogram('p_classes', p_classes)
        tf.summary.histogram('theta_classes', theta_classes)
        tf.summary.histogram('l_classes', l_classes)
        outputs = (p_classes, theta_classes, l_classes)
        if self.include_type:
            type_classes = tf.argmax(inputs[3], axis=1)
            tf.summary.histogram('type_classes', type_classes)
            outputs = outputs + type_classes

        return outputs

    def loss_inverse(self, labels, predictions):
        #loss_inverse_p = tf.losses.sparse_softmax_cross_entropy(
        #    labels=p_labels, logits=p)
        #loss_inverse_theta = tf.losses.sparse_softmax_cross_entropy(
        #    labels=theta_labels, logits=theta)
        #loss_inverse_l = tf.losses.sparse_softmax_cross_entropy(
        #    labels=l_labels, logits=l)
        with tf.variable_scope('loss_inverse'):
            loss_inverse_p = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=predictions[0], labels=labels[0]))
            loss_inverse_theta = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=predictions[1], labels=labels[1]))
            loss_inverse_l = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=predictions[2], labels=labels[2]))
            if (self.include_type):
                loss_inverse_type = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=predictions[3], labels=labels[3]))
                loss_inverse = tf.add(tf.add(tf.add(loss_inverse_p, loss_inverse_theta),
                                             loss_inverse_l),
                                      loss_inverse_type)
            else:
                loss_inverse = tf.add(tf.add(loss_inverse_p, loss_inverse_theta),
                                      loss_inverse_l)
            tf.summary.scalar('loss_inverse', loss_inverse)
        return loss_inverse

    def loss_forward(self, x2, x_pred):
        with tf.variable_scope('loss_forward'):
            l1_norm = tf.reduce_sum(tf.abs(tf.subtract(x2, x_pred)), axis=1)
            loss_forward = tf.reduce_mean(l1_norm)
            #loss_forward = tf.reduce_sum(tf.abs(tf.subtract(x2, x_pred)))
            tf.summary.scalar('loss_forward', loss_forward)
        return loss_forward

    def loss_joint(self, loss_inverse, loss_forward, lamb):
        loss_joint = tf.add(
            loss_inverse,
            tf.multiply(tf.constant(lamb, tf.float32),loss_forward),
            'loss_joint')
        tf.summary.scalar('loss', loss_joint)
        return loss_joint

    def train(self, train_layers, learning_rate, epsilon, loss):
        # Variables within siamese, inverse and forward scopes are trainable.
        #var_list = [v for v in tf.trainable_variables()
        #            if v.name.split('/')[0] in train_layers
        #            or v.name.split('/')[1] in train_layers]
        with tf.variable_scope('train'):
            #gradients = tf.gradients(loss, var_list)
            #gradients = list(zip(gradients, var_list))

            # Create an optimizer and apply gradients on specified variables.
            optimizer = tf.train.AdamOptimizer(
                learning_rate=learning_rate, epsilon=epsilon)
            #train_op = optimizer.apply_gradients(grads_and_vars=gradients)
            train_op = optimizer.minimize(loss)
        return train_op
