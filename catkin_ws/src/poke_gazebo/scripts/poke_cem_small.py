#!/usr/bin/env python
"""
Cross entropy method for planning on LSTM.
"""
import math
import sys
import tensorflow as tf
import numpy as np

import rospy
from poke_random_new import PokeRandomNew
from poke_test import PokeNewExtend

sys.path.insert(0, '/home/wuyang/workspace/python/tf/poke_model/')
from poke_ae_rnn import PokeVAERNN
from batch_operation import image_feed_ae, image_feed
from poke import Poke
from poke_depth import PokeDepth, PokeTotal


class CEM_sim(object):
    def __init__(self, tN=3, n_iters=5, m_samples=40, topK=10, split_size=128):
        self.tN = tN
        self.n_iters = n_iters
        self.m_samples = m_samples
        self.topK = topK
        self.split_size = split_size

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.ae = PokeVAERNN(
                batch_size=1, split_size=split_size, in_channels=3, corrupted=0,
                is_training=False, lstm=1)
        self.sess = tf.InteractiveSession(graph=self.graph)
        restore_path = '../../../../python/tf/logs/pokeAERNN_new/lstm_vae_8_little/'
        saver = tf.train.Saver()
        saver.restore(self.sess, tf.train.latest_checkpoint(restore_path))

        self.graph2 = tf.Graph()
        with self.graph2.as_default():
            self.poke_tf = Poke(include_type=0)
        self.sess2 = tf.InteractiveSession(graph=self.graph2)
        saver2 = tf.train.Saver()
        restore_path = '/home/wuyang/workspace/python/tf/logs/poke_inv/rgb_1/'
        saver2.restore(self.sess2, tf.train.latest_checkpoint(restore_path))

    def get_state(self, path):
        """
        img: path to image which will have shape of [1, h, w, c].
        Can be retrieved by calling image_feed(path)
        """
        imgt = image_feed_ae(path, type_img=1, normalized=0)
        img = imgt.eval()
        # option 1: zero actions. Old trained model.
        u0 = np.zeros((1,4), dtype=np.float32)
        x0 = self.sess.run(self.ae.transit_series[0],
                           feed_dict={self.ae.i1: img, self.ae.u: (u0, u0, u0)})

        print 'init x shape: ', x0.shape
        return np.squeeze(x0)

    def get_rollout(self, pose, U):
        """
        is non zero action at U[0] ok? Treat m_samples as batch_size of the network.
        pose: initial encoding of pose of shape [num_state,]
        U: an array of shape [tN, num_action]
        """
        #print 'pose shape ', pose.shape, 'U shape ', U.shape
        tN, num_action = U.shape
        assert tN == self.tN
        num_state = pose.shape[0]

        U_feed = U.reshape((1, tN, num_action)).astype(np.float32)
        pose_feed = pose.reshape((1, num_state)).astype(np.float32) # (1, num_state)

        # transits = self.sess.run(self.ae.transit_temp_l,
        #                          feed_dict={self.ae.u: U_feed, self.ae.pose: pose_feed})
        U_feed = np.concatenate([np.zeros((1, 1, num_action), dtype=np.float32), U_feed], axis=1)
        U_feed = np.split(U_feed, indices_or_sections=tN+1, axis=1)
        U_feed_list = [np.squeeze(item, axis=1) for item in U_feed]
        transits = self.sess.run(self.ae.transit_series,
                                 feed_dict={self.ae.u: U_feed_list, self.ae.pose: pose_feed})

        # tN of [m_samples, num_state]
        #print 'length ', len(transits), 'state shape ', transits[0].shape

        trans = np.concatenate(transits, axis=0)
        #print 'return transit shape', trans.shape

        return trans

    def get_cost(self, trans):
        """
        trans: transit states of shape [tN, num_state]
        Only calculate cost for one action sequence.
        """
        tN = trans.shape[0]
        #assert tN == self.tN
        num_state = trans.shape[1]
        assert num_state == self.x_target.shape[0]

        #print 'is target available? ', type(self.x_target)

        errors = self.x_target[:num_state/2] - trans[:, :num_state/2]

        cost = np.sum(errors**2)

        return cost

    def cem(self, pose, mean, cov):
        """
        pose: initial encoding of the pose of shape [num_state,]
        mean: mean vector of size [tN, 4]
        cov: covariance matrix of shape [tN, 4, 4]
        """
        #print 'mean shape', mean.shape, 'cov shape', cov.shape
        assert mean.shape[0] == cov.shape[0] == self.tN
        num_action = mean.shape[1]

        U0 = np.zeros((self.m_samples, self.tN, num_action))
        for ii in range(self.n_iters):
            for kk in range(self.tN):
                U0[:, kk, :] = np.random.multivariate_normal(mean[kk], cov[kk], (self.m_samples))

            costs = np.zeros(self.m_samples)
            for jj in range(self.m_samples):
                trans = self.get_rollout(pose, U0[jj])
                costs[jj] = self.get_cost(trans)
            idx = costs.argsort()[:self.topK]

            UtopK = U0[idx] # [topK, self.tN, num_action]
            for kk in range(self.tN):
                mean[kk] = np.mean(UtopK[:, kk, :], axis=0)
                cov[kk] = np.cov(UtopK[:, kk, ::], rowvar=0)

            print 'iteration %d'%ii
            print 'top action', UtopK[0], '\n'

        return mean, cov, UtopK[0]

    def cem_traj(self, pose, mean, cov):
        """
        pose: initial encoding of the pose of shape [num_state,]
        mean: mean vector of size [2, tN*num_action/2]
        cov: covariance matrix of shape [2, tN*num_action/2, tN*num_action/2]
        """
        #print 'mean shape', mean.shape, 'cov shape', cov.shape
        num_action = mean.shape[0]*mean.shape[1]/self.tN

        U0 = np.zeros((self.m_samples, self.tN, num_action))
        for ii in range(self.n_iters):
            # sample [m_samples, tN*num_action/2]
            xy = np.random.multivariate_normal(mean[0], cov[0], (self.m_samples)) 
            xy = xy.reshape((self.m_samples, self.tN, num_action/2))
            thetal = np.random.multivariate_normal(mean[1], cov[1], (self.m_samples))
            thetal = thetal.reshape((self.m_samples, self.tN, num_action/2))
            #print 'xy shape', xy.shape, 'thetal shape', thetal.shape
            U0 = np.concatenate((xy, thetal), axis=2)

            costs = np.zeros(self.m_samples)
            for jj in range(self.m_samples):
                trans = self.get_rollout(pose, U0[jj])
                costs[jj] = self.get_cost(trans)
            idx = costs.argsort()[:self.topK]

            UtopK = U0[idx] # [topK, self.tN, num_action]
            Uxy = UtopK[:, :, :num_action/2]
            Uthetal = UtopK[:, :, num_action/2:]

            Uxy = Uxy.reshape((self.topK, self.tN*num_action/2))
            Uthetal = Uthetal.reshape((self.topK, self.tN*num_action/2))

            mean[0] = np.mean(Uxy, axis=0)
            cov[0] = np.cov(Uxy, rowvar=0)
            mean[1] = np.mean(Uthetal, axis=0)
            cov[1] = np.cov(Uthetal, rowvar=0)

            #print 'iteration %d'%ii
            #print 'top action', UtopK[0], '\n'

        return mean, cov, UtopK[0]

    def cem_control(self, mean, cov, init_path, target_path=None):

        img0t = image_feed_ae(init_path, type_img=1, normalized=0)
        img0 = img0t.eval()
        u0 = np.zeros((1,4), dtype=np.float32)
        x0, pose = self.sess.run([self.ae.transit_series[0], self.ae.pose],
                                 feed_dict={self.ae.i1: img0, self.ae.u: (u0, u0, u0)})

        self.x_init = np.squeeze(x0)
        self.init_pose = np.squeeze(pose)
        if target_path is not None:
            self.x_target = self.get_state(target_path)

        # init_mean = np.tile(np.array([0.4, 0.4, 0.6, 0.4]), (self.tN, 1))
        # init_cov_list = [0.03*np.eye(4) for _ in range(self.tN)]
        # init_cov = np.stack(init_cov_list, axis=0)
        # #mean, cov, Utop = self.cem(self.init_pose, init_mean, init_cov)

        mean_new, cov_new, Utop = self.cem_traj(self.init_pose, mean, cov)

        #print 'last top action', Utop
        return mean_new, cov_new, Utop

if __name__ == '__main__':
    rospy.init_node('poke_test')

    path = '/home/wuyang/workspace/python/poke/test_poke/'
    init_pose = np.array([[0.105, 0.255, 0, 0, 0.14943813, 0.98877108]])

    poke_ros = PokeNewExtend(0.38)
    rospy.sleep(2.0)
    poke_ros.delete_gazebo_models(0)
    name = poke_ros.load_gazebo_models()
    poke_ros.move_to_start()
    rospy.sleep(2.0)

    cem = CEM_sim(tN=2, n_iters=5, m_samples=100, topK=20, split_size=128)
    cem.x_target = cem.get_state(path+'img%04d.jpg'%100)
    target_img = image_feed(path+'img%04d.jpg'%100, target_shape=227, type_img=1, normalized=0)
    target_i = target_img.eval()

    mean = np.stack(
        [0.5*np.ones(cem.tN*2), 0.6*np.ones(cem.tN*2)], axis=0)
    cov = np.stack(
        [0.005*np.eye(cem.tN*2), 0.001*np.eye(cem.tN*2)], axis=0)

    i = 0
    k = 0
    while not rospy.is_shutdown():
        if i > 15:
            _, _, bp, bo = poke_ros.poke_generation(name)
            pose = np.array([[bp.x, bp.y, bo.x, bo.y, bo.z, bo.w]])
            error = pose - init_pose
            with open(path+'pose_inv_forw.dat', 'ab') as f_handle:
                np.savetxt(f_handle, error)

            poke_ros.delete_gazebo_models(0)
            name = poke_ros.load_gazebo_models()
            rospy.sleep(1.0)
            cem.x_target = cem.get_state(path+'img%04d.jpg'%100)
            target_img = image_feed(path+'img%04d.jpg'%100, target_shape=227, type_img=1)
            target_i = target_img.eval()
            i = 0
            k +=1
            if k > 20:
                break

        current_img = image_feed(path+'img%04d.jpg'%i, target_shape=227, type_img=1, normalized=0)
        feed_dict = {cem.poke_tf.i1: current_img.eval(), cem.poke_tf.i2: target_i}

        p_class, theta_class, l_class = cem.sess2.run(
            [cem.poke_tf.classes[0],
             cem.poke_tf.classes[1],
             cem.poke_tf.classes[2]],
            feed_dict=feed_dict)

        p_x = p_class[0]//20*12+6
        p_y = p_class[0]%20*12+6
        theta = theta_class[0]*np.pi/18 + np.pi/36
        l = l_class[0]*0.004 + 0.002 + 0.04
        print 'inverse action: %.2f, %.2f, %.2f, %.2f'%(p_x, p_y, theta, l)

        mean[0] = np.array((p_x/240.0, p_y/240.0, p_x/240.0, p_y/240.0))
        mean[1] = np.array((theta/(2*np.pi), (l-0.04)/(0.04), theta/(np.pi*2), (l-0.04)/(0.04)))
        #print 'mean updated', mean

        rospy.sleep(0.5)
        _, _, Utop = cem.cem_control(
            mean, cov, path+'img%04d.jpg'%i)

        Utop0 = Utop[0]
        p_x = Utop0[0]*240
        p_y = Utop0[1]*240
        theta = Utop0[2]*np.pi*2
        l = Utop0[0]*0.04 + 0.04

        print 'cem action: %.2f, %.2f, %.2f, %.2f'%(p_x, p_y, theta, l)

        position_world = poke_ros.model_world_projection(p_x, p_y)
        poke_ros.poke(position_world, {'theta': theta, 'l': l})
        rospy.sleep(1.0)
        poke_ros.save_image(path, i+1)
        i+=1
