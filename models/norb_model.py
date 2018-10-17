import numpy as np
import tensorflow as tf
import os
import numpy as np
import embedding as emb
class NorbModel(emb.EmbeddingModel):
    def get_output_size(self):
        return [None, 96, 96, 1]
    def load_dataset(self,params):
        ds = np.load(params.dataset)
        self.params = params
        imagedata_cam1 = ds['imagedata'][:,0,:,:]
        imagedata_cam2 = ds['imagedata'][:,1,:,:]
        imagedata = np.concatenate( (imagedata_cam1, imagedata_cam2) )
        imagedata = imagedata.reshape( (-1, 96, 96, 1) )

        if params.norb_use_10_levels_for_pose:
            print '10 levels for pose'
            def makebins(vals,nbins):
                counts,bins = np.histogram(vals,bins=nbins)
                bin_idx=0
                result = -np.ones(len(vals), dtype=np.int32)
                for l,r in zip( bins[:-1], bins[1:]):
                    if bin_idx < (nbins-1): result[(vals >= l)&(vals < r)] = bin_idx
                    else: result[(vals >= l)&(vals <= r)] = bin_idx
                    bin_idx+=1
                assert (result == -1).sum() == 0
                return result
            az_bin = makebins(ds['azimuth'], 10)
            el_bin = makebins(ds['elevation'], 10)
        else:
            el_bin = ds['elevation'] < 5
            az_bin = ds['azimuth'] < 18

        factors = np.zeros( (len(ds['category']), 3), dtype=np.int32 )
        if params.norb_use_category_as_identity:
            identities = ds['category']
        else:
            identities_str = map(lambda x: ','.join(map(str,x)), zip(ds['category'],ds['instance']) )
            unique_identities, identities, counts = np.unique(identities_str, return_inverse=True, return_counts=True)
        factors[:,0] = identities
        factors[:,1] = el_bin
        factors[:,2] = az_bin
        factors = np.concatenate( (factors, factors) )
        self.factors = factors
        if params.multi_factor:
            return imagedata, factors, ds['traintest']==0, ds['traintest']==1, ds['traintest']==1
        else:
            identities_str = map(lambda x: ','.join(map(str,x)), zip(identities, el_bin, az_bin) )
            unique_identities, identities,counts = np.unique(identities_str, return_inverse=True,return_counts=True)
            print len(unique_identities), 'unique ids'
            print 'counts min', np.min(counts), 'max', np.max(counts),'mean', np.mean(counts)
            identities = np.concatenate( (identities, identities) )
            return imagedata, identities, np.ones(len(identities)), np.array([]), np.array([])
    def recognition_net(self):
        bn = self.batchnorm
        self.l1_w = self.wrecog([7,7,1,48], 'l1_w')
        self.l1_b = self.brecog([48], 'l1_b')
        self.l2_w = self.wrecog([3,3,48,64], 'l2_w')
        self.l2_b = self.brecog([64], 'l2_b')
        self.l3_w = self.wrecog([3,3,64,72], 'l3_w')
        self.l3_b = self.brecog([72], 'l3_b')
        l1 = tf.nn.relu( bn( tf.nn.conv2d(self.x_rescaled, self.l1_w, strides=[1,2,2,1], padding='VALID' ) + self.l1_b) )
        l2 = tf.nn.relu( bn( tf.nn.conv2d(l1, self.l2_w, strides=[1,2,2,1], padding='VALID' ) + self.l2_b ) )
        l3 = tf.nn.relu( bn( tf.nn.conv2d(l2, self.l3_w, strides=[1,2,2,1], padding='VALID' ) + self.l3_b ) )
        print 'l3', l3
        self.l1_shape = l1.get_shape().as_list()[1:]
        self.l2_shape = l2.get_shape().as_list()[1:]
        self.l3_shape = l3.get_shape().as_list()[1:]
        print 'input', self.x_rescaled.get_shape().as_list()[1:]
        print 'enc l1', self.l1_shape
        print 'enc l2', self.l2_shape
        print 'enc l3', self.l3_shape
        before_embedding_size = int(np.prod(self.l3_shape))
        l3_reshaped = tf.reshape(l3, [-1, before_embedding_size] )
        print l3_reshaped.shape
        self.l3_reshaped = l3_reshaped
        if self.params.beta_vae > 0.:
            nz = self.n_z*2
        else:
            nz = self.n_z
        self.emb_w = self.wrecog([before_embedding_size, nz], 'emb_w')
        self.emb_b = self.wrecog([nz], 'emb_b')
        z = tf.matmul(l3_reshaped, self.emb_w) + self.emb_b
        return z
    def generator_net(self,z):
        bn = self.batchnorm
        n_l1 = int(self.l1_b.shape[0])
        n_l2 = int(self.l2_b.shape[0])
        n_l3 = int(self.l3_b.shape[0])
        l3_total = int(np.prod(self.l3_shape))
        resize = tf.image.resize_nearest_neighbor
        l3 = tf.nn.relu( bn( tf.matmul( z, self.wrecog([self.n_z, l3_total],'gw4') ) +
                            self.brecog([l3_total],'gb3') ) )
        l2_input = tf.reshape(l3, [-1]+self.l3_shape)
        print 'l2 input', l2_input
        l2_input = resize(l2_input, (self.l2_shape[0], self.l2_shape[1]))
        print 'l2 input', l2_input, 'n_l3', n_l3
        l2 = tf.nn.relu( bn( tf.nn.conv2d(l2_input,
                                           self.wrecog([3,3,n_l3,n_l2],'gw2'),
                                           strides=[1,1,1,1],
                                           padding='SAME') +
                            self.brecog([n_l2], 'gb2')) )
        print 'l2 output', l2
        l2_upsampled = resize(l2, (self.l1_shape[0], self.l1_shape[1]))
        print 'l1 input', l2_upsampled
        l1 = tf.nn.relu( tf.nn.conv2d(l2_upsampled,
                                          self.wrecog([3,3,n_l2,n_l1], 'gw1'),
                                          strides=[1,1,1,1],
                                          padding='SAME') +
                        self.brecog([n_l1], 'gb1'))

        l1_upsampled = resize(l1, self.get_output_size()[:2])
        print 'l1_upsampled', l1_upsampled
        output = tf.nn.sigmoid( tf.nn.conv2d(l1_upsampled,
                                              self.wrecog([7,7,n_l1,1], 'gw0'),
                                              strides=[1,1,1,1],
                                              padding='SAME')+
                                self.brecog([1], 'gb0'))
        print output
        raise ''
        return l1
