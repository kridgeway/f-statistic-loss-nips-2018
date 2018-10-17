import numpy as np
import tensorflow as tf
import embedding as emb

class CUB200Model(emb.EmbeddingModel):
    def get_output_size(self):
        return [None, 2048]
    def load_dataset(self,params):
        ds = np.load(params.dataset)
        X = ds['features']
        ids = ds['identities']
        print 'foobar', X.shape, ids.shape
        return X, ids, [], [], []
    def recognition_net(self):
        #from https://stackoverflow.com/questions/34484148/feeding-image-data-in-tensorflow-for-transfer-learning
        n_h = 1024
        self.fc_w_1 = self.wrecog([2048, n_h], 'ew1')
        self.fc_w_2 = self.wrecog([n_h, self.n_z_id], 'w2')
        l1 = tf.nn.relu( tf.matmul(self.x, self.fc_w_1 ) + \
                        self.brecog([n_h], 'eb1') )
        z = tf.matmul(l1, self.fc_w_2 ) + \
            self.brecog([self.n_z_id], 'b2')
        return z
    def create_loss_optimizer(self, loss_methods):
        fc_w_1 = self.fc_w_1
        fc_w_2 = self.fc_w_2
        class WrappedLossMethods(object):
            def create_loss(self, z_id, is_id):
                beta = 0.
                loss = loss_methods.create_loss(z_id,is_id)
                #loss = loss + beta * tf.nn.l2_loss(fc_w_1) + \
                #    beta * tf.nn.l2_loss(fc_w_2)
                #loss = loss + beta * tf.nn.l2_loss(fc_w_2)
                return loss
        wrapped_loss_methods = WrappedLossMethods()
        super(CUB200Model, self).create_loss_optimizer(wrapped_loss_methods)
