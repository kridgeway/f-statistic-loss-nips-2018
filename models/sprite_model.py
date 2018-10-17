import numpy as np
import tensorflow as tf
import os
import numpy as np
import embedding as emb
import sys
from sklearn import cross_validation

class SpriteModel(emb.EmbeddingModel):
    def get_output_size(self):
        return [None, 60, 60, 3]
    def load_dataset(self,params):
        ds = np.load(params.dataset)
        X = ds['imagedata']
        ids = ds['identities']
        unique_ids = np.random.permutation(np.unique(ids))
        #n_train_val = int(0.66 * len(unique_ids))
        #n_train_val = 300
        n_train_val = 572
        n_train_ids = 500
        n_test = 100
        train_val_ids = unique_ids[:n_train_val]
        test_ids = unique_ids[n_train_val:n_train_val+n_test]

        # Style factors
        poses=ds['poses']
        # Poses are grouped by action, 4 poses per action
        actions=poses / 4
        new_poses = poses % 4
        # The last pose is actually a frontal pose, but the other poses are
        # missing from the action.
        new_poses[poses == np.max(poses)] = 2
        self.style_factors = np.array( [actions, new_poses] ).T
        print 'style factors shape', self.style_factors.shape

        train_ids = train_val_ids[:n_train_ids]
        val_ids = train_val_ids[n_train_ids:]
        print len(train_ids),'train ids', len(val_ids), 'val ids', len(test_ids), 'test ids'
        train_idx = np.in1d(ids,train_ids)
        val_idx = np.in1d(ids,val_ids)
        test_idx = np.in1d(ids,test_ids)
        self.factors = np.array(ds['labels'].reshape( (-1, 7) ),dtype=np.int32)

        # The last factor, weapon type, is actually determined by action,
        # so just eliminate it from factors and relabel identities
        #print 'before', np.unique(ids).shape, 'unique identities'
        #self.factors = self.factors[:,:6]
        #ids = emb.transform_multi_factor_to_single(self.factors)
        #print 'after', np.unique(ids).shape, 'unique identities'

        if params.multi_factor:
            return X, self.factors, train_idx, val_idx, test_idx
        else:
            return X, ids, train_idx, val_idx, test_idx
    def recognition_net(self):
        bn = self.batchnorm
        n_l1 = 64
        n_l2 = 32
        n_l3 = 2048
        l1 = tf.nn.relu( bn( tf.nn.conv2d(self.x_rescaled, \
                                          self.wrecog([5,5,3,n_l1],'ew1'),
                                          strides=[1,2,2,1],
                                          padding='VALID') +
                            self.brecog([n_l1],'eb1')
                        ))
        self.l1_shape = l1.get_shape().as_list()[1:]
        l2 = tf.nn.relu( bn( tf.nn.conv2d(l1, \
                                          self.wrecog([5,5,n_l1,n_l2],'ew2'),
                                          strides=[1,2,2,1],
                                          padding='VALID') +
                            self.brecog([n_l2],'eb2')
                        ))
        print l2.get_shape()
        self.l2_shape = l2_shape = l2.get_shape().as_list()[1:]
        self.l2_total_output_size = n_l2 = np.prod(l2_shape)
        l2_reshaped = tf.reshape(l2, [-1,n_l2])
        l3 = tf.nn.relu( tf.matmul(l2_reshaped, self.wrecog([n_l2, n_l3],'ew3')) \
                        + self.brecog([n_l3],'eb3') )
        if self.params.beta_vae > 0.:
            nz = self.n_z*2
        else:
            nz = self.n_z
        l4 = tf.matmul(l3, self.wrecog([n_l3, nz], 'ew4') ) +\
            self.brecog([nz],'eb4')
        return l4
    def generator_net(self,z):
        bn = self.batchnorm
        n_l1 = 64
        n_l2 = 32
        n_l3 = 2048
        l4 = tf.nn.relu( bn( tf.matmul( z, self.wrecog([self.n_z, n_l3],'gw4') ) +
                            self.brecog([n_l3],'gb3') ) )
        l3 = tf.nn.relu( bn( tf.matmul( l4, self.wrecog([n_l3, self.l2_total_output_size],'gw3') ) +
                            self.brecog([self.l2_total_output_size],'gb2') ) )
        l3_reshaped = tf.reshape(l3, (-1, self.l2_shape[0], self.l2_shape[1], self.l2_shape[2]))
        l3_upsampled = tf.image.resize_nearest_neighbor(l3_reshaped, (self.l1_shape[0], self.l1_shape[1]))
        l2 = tf.nn.relu( bn( tf.nn.conv2d(l3_upsampled,
                                           self.wrecog([5,5,n_l2,n_l1],'gw2'),
                                           strides=[1,1,1,1],
                                           padding='SAME') +
                            self.brecog([n_l1], 'gb2')) )
        l2_upsampled = tf.image.resize_nearest_neighbor(l2, (60,60) )
        l1 = tf.nn.sigmoid( tf.nn.conv2d(l2_upsampled,
                                          self.wrecog([5,5,n_l1,3], 'gw1'),
                                          strides=[1,1,1,1],
                                          padding='SAME') +
                            self.brecog([3], 'gb1'))
        return l1
