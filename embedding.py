import sys
import os
slash = '/' if os.path.dirname(__file__) != '' else ''
sys.path.append( os.path.dirname(__file__) + slash +  "loss/")
sys.path.append( os.path.dirname(__file__) +  slash + "models/")
sys.path.append( os.path.dirname(__file__) +  slash + "eval/")
sys.path.append( os.path.dirname(__file__) +  slash + "batch_gen/")
sys.path.append( os.path.dirname(__file__) +  slash + "visualization/")
import numpy as np
import tensorflow as tf
import numpy as np
import math
import re

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

import argparse
import traceback
import progressbar
import recall_at_k

from sklearn.cross_validation import *
import pandas as pd
import time

def get_fans(shape, dim_ordering='th'):
    if len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    elif len(shape) == 4 or len(shape) == 5:
        # assuming convolution kernels (2D or 3D).
        # TH kernel shape: (depth, input_depth, ...)
        # TF kernel shape: (..., input_depth, depth)
        if dim_ordering == 'th':
            receptive_field_size = np.prod(shape[2:])
            fan_in = shape[1] * receptive_field_size
            fan_out = shape[0] * receptive_field_size
        elif dim_ordering == 'tf':
            receptive_field_size = np.prod(shape[:2])
            fan_in = shape[-2] * receptive_field_size
            fan_out = shape[-1] * receptive_field_size
        else:
            raise Exception('Invalid dim_ordering: ' + dim_ordering)
    else:
        # no specific assumptions
        fan_in = np.sqrt(np.prod(shape))
        fan_out = np.sqrt(np.prod(shape))
    return fan_in, fan_out

def xavier_init(fan_in, fan_out, shape, constant=1):
    """ Xavier initialization of network weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant*np.sqrt(6.0/(fan_in + fan_out))
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform(shape, minval=low, maxval=high, dtype=tf.float32)

def wvar(shape, name=None):
    fan_in, fan_out = get_fans(shape,'tf')
    initial = xavier_init(fan_in, fan_out, shape)
    return tf.Variable(initial, name=name)

def bvar(shape, name=None):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial, name=name)

class EmbeddingModel(object):
    def __init__(self,n_id,params,lr):
        self.n_z_id = n_id
        self.n_z_nid = params.ae_n_nid
        self.n_z = n_id + params.ae_n_nid
        self.lr = lr
        self.params=params
        self.network_weights = self.initialize_weights()

        self.x = tf.placeholder(tf.float32,self.get_output_size())
        self.x_rescaled = self.x / 255.
        self.z = self.recognition_net()
        self.z_id = self.z[:,:n_id]
        if params.beta_vae > 0.:
            self.mu = self.z[:,:self.n_z]
            self.lsq = self.z[:,self.n_z:]
            self.sigma = tf.sqrt(tf.exp(self.lsq))
            self.z_samp = self.mu + self.sigma * \
                tf.random_normal(tf.shape(self.mu), 0, 1, dtype=tf.float32)
        self.unnormalized_z = self.z
        self.unnormalized_z_id = self.z_id
        if params.l2_normalize == 1:
            if params.autoencoder:
                self.z_id = tf.nn.l2_normalize(self.z_id, dim=1)
            else:
                self.z = tf.nn.l2_normalize(self.z, dim=1)
        else:
            print "No L2 Length Normalization"
        if self.params.autoencoder and self.params.autoencoder_split_z:
            self.z_nid = self.z[:,n_id:]
        if params.autoencoder:
            if params.beta_vae > 0.:
                self.x_hat = self.generator_net(self.z_samp)
            else:
                self.x_hat = self.generator_net(self.z)
            self.x_hat_rescaled = self.x_hat * 255.

    def initialize(self):
        self.create_loss_optimizer(self.params.loss_methods)
        init = tf.global_variables_initializer()
        #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
        #self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        self.sess = tf.Session(config=config)
        if self.params.advanced_summary:
            grads = tf.gradients(self.disentangling_loss, tf.trainable_variables())
            grads = list(zip(grads, tf.trainable_variables()))
            for grad, var in grads:
                tf.summary.scalar(var.name + '/nans', tf.reduce_sum(tf.cast(tf.is_nan(grad), tf.int32)))
                #tf.summary.histogram(var.name + '/gradient', grad)
                tf.summary.scalar(var.name + '/gradient_max', tf.reduce_max(tf.abs(grad)))
                tf.summary.histogram(var.name + '/weights', var)
        self.summary_var = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(self.params.model_prefix + '/train', self.sess.graph)
        self.sess.run(init)
    def initialize_weights(self):
        return {'weights_recog' : {}, 'biases_recog': {}, 'weights_gen':{}, 'biases_gen':{} }
    def wrecog(self,shape,name):
        #print 'wrecog',name,shape
        self.network_weights['weights_recog'][name] = result = wvar(shape, name)
        return result
    def brecog(self,shape,name):
        #print 'brecog',name,shape
        self.network_weights['biases_recog'][name] = result = bvar(shape, name)
        return result

    def get_output_size(self):
        raise 'TODO'
    def recognition_net(self):
        raise 'TODO'
    def generator_net(self, z):
        raise 'TODO'

    def batchnorm(self, input):
        #if self.params.batchnorm:
        return tf.nn.batch_normalization(input,mean=0.,variance=1.0,offset=None,scale=None,variance_epsilon=1e-8)
        #else:
        #    return input

    def create_loss_optimizer(self, loss_methods):
        if self.params.autoencoder:
            self.disentangling_loss = loss_methods.create_loss(self, self.z_id)
            if self.params.beta_vae > 0.:
                mu = self.mu
                lsq = self.lsq
                mu_dev = mu**2.
                lsq_dev = tf.exp(lsq) - lsq
                self.kl_divergence_batch = 0.5 * tf.reduce_sum(mu_dev + lsq_dev, 1)
                self.kl_divergence = tf.reduce_mean(self.kl_divergence_batch)
                def bernoulli_log_likelihood(a_in,b_in):
                    #tanh > sigmoid.
                    eps = 1e-7
                    a = tf.clip_by_value( (a_in+1)/2., clip_value_min=eps, clip_value_max=1.-eps)
                    b = tf.clip_by_value( (b_in+1)/2., clip_value_min=eps, clip_value_max=1.-eps)
                    result = tf.reduce_sum(a*tf.log(b) + (1.-a)*tf.log(1.-b), axis=(1,2,3))
                    return result
                self.recon_likelihood_batch = bernoulli_log_likelihood(self.x_rescaled, self.x_hat)
                self.recon_likelihood = tf.reduce_mean(self.recon_likelihood_batch)
                self.reconstruction_loss = -self.recon_likelihood
                self.elbo = self.recon_likelihood - self.params.beta_vae * self.kl_divergence
                self.cost = -self.elbo
            else:
                self.reconstruction_loss = tf.reduce_sum( tf.reduce_mean( (self.x_rescaled - self.x_hat)**2., axis=[1,2,3] ) )
                self.cost =  self.disentangling_loss * self.params.content_loss_weight + \
                    self.params.recon_loss_weight * self.reconstruction_loss
        else:
            self.disentangling_loss = loss_methods.create_loss(self, self.z)
            self.cost = self.disentangling_loss
        #if not self.params.l2_normalize:
        #    self.cost = self.cost + 1e-5*tf.reduce_sum(self.z**2)
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.lr,
            beta1=self.params.adam_beta1,
            beta2=self.params.adam_beta2
        ).minimize(self.cost)
    def compute_embeddings(self,X,z=[],show_progress=True):
        if len(z)==0:
            z = np.zeros( (X.shape[0], self.n_z_id), dtype=np.float32 )
        if show_progress:
            bar = progressbar.ProgressBar()
        else:
            bar = lambda x: x
        for begin_idx in bar(xrange(0,X.shape[0],100)):
            end_idx = min(begin_idx+100, X.shape[0])
            batch_z = self.sess.run([self.z_id], feed_dict={self.x:X[begin_idx:end_idx]})[0]
            z[begin_idx:end_idx,:] = batch_z
        return z
    def compute_nid_embeddings(self,X,z=[],show_progress=True):
        if len(z)==0:
            z = np.zeros( (X.shape[0], self.n_z_nid), dtype=np.float32 )
        if show_progress:
            bar = progressbar.ProgressBar()
        else:
            bar = lambda x: x
        for begin_idx in bar(xrange(0,X.shape[0],100)):
            end_idx = min(begin_idx+100, X.shape[0])
            batch_z = self.sess.run([self.z_nid], feed_dict={self.x:X[begin_idx:end_idx]})[0]
            z[begin_idx:end_idx,:] = batch_z
        return z
    def compute_full_embeddings(self,X,z=[],show_progress=True):
        if len(z)==0:
            z = np.zeros( (X.shape[0], self.n_z), dtype=np.float32 )
        if show_progress:
            bar = progressbar.ProgressBar()
        else:
            bar = lambda x: x
        for begin_idx in bar(xrange(0,X.shape[0],100)):
            end_idx = min(begin_idx+100, X.shape[0])
            batch_z = self.sess.run([self.z], feed_dict={self.x:X[begin_idx:end_idx]})[0]
            z[begin_idx:end_idx,:] = batch_z
        return z
    def compute_recon_loss(self,X,show_progress=True):
        if show_progress:
            bar = progressbar.ProgressBar()
        else:
            bar = lambda x:x
        losses=[]
        for begin_idx in bar(xrange(0,X.shape[0],100)):
            end_idx = min(begin_idx+100, X.shape[0])
            batch_loss = self.sess.run([self.reconstruction_loss], feed_dict={self.x:X[begin_idx:end_idx]})[0]
            losses.append(batch_loss)
        return np.sum(losses) / X.shape[0]

def do_profile(X,identities,params,model):
    from tensorflow.python.client import timeline
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    ids_in_order = np.unique(identities)
    for batch_idx,losses in params.loss_methods.train_batches(model, X, identities,\
                                                    ids_in_order,params.batch_size,\
                                                    params.examples_per_identity, run_options=run_options, run_metadata=run_metadata):
        if batch_idx == 5: break
    tl = timeline.Timeline(run_metadata.step_stats)
    ctf = tl.generate_chrome_trace_format()
    with open(params.trace_file, 'w') as f: f.write(ctf)

def compute_between_within_dists(model, X_train, ids):
    embedding = model.compute_embeddings(X_train)
    return compute_between_within_dists_from_embedding(embedding,ids)

def compute_between_within_dists_from_embedding(embedding, ids):
    id_means=[]
    within_dists=[]
    max_withins=[]
    for identity in np.unique(ids):
        this_id = np.where(ids == identity)[0]
        samples = embedding[this_id]
        id_mean = samples.mean(axis=0)
        #id_dists = np.sum( np.abs(samples-id_mean), axis=1)
        id_dists = np.sum( (samples-id_mean)**2.0, axis=1)
        max_withins.append(np.max(id_dists))
        within_dists.extend(id_dists)
        id_means.append(id_mean)
    id_means = np.array(id_means)
    global_mean = np.mean(id_means,axis=0)
    #between_dists = np.sum(np.abs(id_means - global_mean),axis=1)
    #between_dists = np.sum((id_means - global_mean)**2.0,axis=1)
    between_dists=[]
    for id_mean_idx, id_mean_a in enumerate(id_means):
        for id_mean_b in id_means[id_mean_idx+1:]:
            between_dists.append( np.sum( (id_mean_a-id_mean_b)**2.0) / 2.0 )
    return map(np.array, [within_dists, between_dists, max_withins])

def monitor_train_dists(model, X_train, ids):
    within_dists, between_dists, max_withins = compute_between_within_dists(model, X_train, ids)
    between_dist_mean = np.mean(between_dists)
    between_dist_std = np.std(between_dists)
    within_dist_mean = np.mean(within_dists)
    within_dist_std = np.std(within_dists)
    #print between_dist_mean, between_dist_std
    #print within_dist_mean, within_dist_std
    #print (between_dist_mean/between_dist_std)**2.0, (within_dist_mean/within_dist_std)**2.0
    with open('%s/distance_log.txt' % model.params.model_prefix, 'a') as outfile:
        outfile.write("%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n" % \
                        (between_dist_mean,between_dist_std, \
                        within_dist_mean, within_dist_std, \
                        np.mean(max_withins),
                        np.min(between_dists)
                        ) )

def transform_multi_factor_to_single(identities):
    b = np.array([','.join(['%d'%identities[i,j] for j in xrange(identities.shape[1])]) for i in xrange(identities.shape[0]) ])
    unique_bs, ids_to_split = np.unique(b, return_inverse=True)
    print len(unique_bs),'unique identities'
    return ids_to_split

def train(X_all, identities, train_idx, val_idx, test_idx, params, model, saver):
    X_train = X_all[train_idx]
    X_val = X_all[val_idx]
    X_test = X_all[test_idx]
    ids_train = identities[train_idx]
    unique_users, user_counts = np.unique(ids_train,return_counts=True)
    clients_train = unique_users[user_counts > 1]
    ids_val = identities[val_idx]
    ids_test = identities[test_idx]
    size = X_train[0].shape
    global best_acc
    best_acc = -np.inf
    epoch=0
    last_epoch_with_loss_improvement=-1
    if params.train_log_file:
        if os.path.exists(params.train_log_file):
            df = pd.read_csv(params.train_log_file)
            epoch = df.epoch.max()+1
            last_epoch_with_loss_improvement = epoch
            if not np.isnan(epoch):
                print 'Restarting training at epoch %d' % epoch
            else:
                print 'Starting training at epoch 0'
                epoch = 0
            params.train_log_file = open(params.train_log_file,'a')
        else:
            params.train_log_file = open(params.train_log_file,'w')
            #params.train_log_file.write('epoch,train_acc,val_acc,test_acc,train_dis_loss\n')
            params.train_log_file.write('epoch,train_acc,val_acc,train_dis_loss')
            if params.autoencoder:
                params.train_log_file.write(',recon_loss')
            params.train_log_file.write('\n')
            params.train_log_file.flush()
    dis_losses=[]
    recon_losses=[]
    nid_emb_losses=[]
    global summary_idx
    summary_idx=0
    def epoch_end():
        global summary_idx
        #monitor_train_dists(model,X_train,ids_train)
        global best_acc
        print 'End of Epoch %d' % epoch

        z_id_val = model.compute_embeddings(X_val)
        if params.multi_factor:
            import explicitness
            exp_results = explicitness.calculate_explicitness(z_id_val, ids_val)
            mean_aucs = exp_results[0]
            val_id_acc = np.mean(mean_aucs)
            train_id_acc = 0.
        else:
            if not params.disable_monitor_train_acc:
                z_id_train = model.compute_embeddings(X_train)
            val_r_at_k = recall_at_k.evaluate(z_id_val, ids_val, params)
            val_id_acc = np.mean(val_r_at_k)
            print 'valr at ak'
            print_recall_k(val_r_at_k,params)
            print 'done'
            if params.disable_monitor_train_acc:
                train_id_acc =0.
            else:
                train_r_at_k = recall_at_k.evaluate(z_id_train, ids_train, params)
                train_id_acc = np.mean(train_r_at_k)

        print 'Training  : acc_id %.4f' % (train_id_acc)
        print 'Validation: acc_id %.4f' % (val_id_acc)

        if params.autoencoder:
            val_recon_acc = 1.- model.compute_recon_loss(X_val)
            train_recon_acc = 1.- model.compute_recon_loss(X_train)

        summary=tf.Summary()
        summary.value.add(tag='train_id_acc', simple_value = train_id_acc)
        summary.value.add(tag='val_id_acc', simple_value = val_id_acc)
        model.train_writer.add_summary(summary, summary_idx)
        #print 'Testing   : acc_id %.4f' % (test_id_acc)
        if params.train_log_file:
            params.train_log_file.write('%d' % (epoch+1))
            params.train_log_file.write(',%.4f' % train_id_acc)
            params.train_log_file.write(',%.4f' % val_id_acc)
            #params.train_log_file.write('%.4f,' % test_id_acc)
            params.train_log_file.write(',%.4E' % np.mean(dis_losses))
            if params.autoencoder:
                params.train_log_file.write(',%.4E' % (1.- val_recon_acc) )
            params.train_log_file.write('\n')
            params.train_log_file.flush()
        if params.autoencoder:
            result = val_recon_acc > best_acc
            if result:
                best_acc = val_recon_acc
            return result
        else:
            result = val_id_acc > best_acc
            if result:
                best_acc = val_id_acc
            return result
    #epoch_end()
    while True:
        del dis_losses[:]
        del recon_losses[:]
        del nid_emb_losses[:]
        client_user_order_train = np.random.permutation(len(clients_train))
        client_users_in_order_train = clients_train[client_user_order_train]
        start = time.time()
        print 'Epoch %d' % epoch
        objectives = [model.optimizer, model.disentangling_loss]
        if params.autoencoder:
            objectives.append(model.reconstruction_loss)
        if model.summary_var != None: objectives.append(model.summary_var)
        for batch_idx,feed_dict in params.loss_methods.train_batches(model, X_train, ids_train):
            results = model.sess.run(objectives, feed_dict=feed_dict)
            opt = results[0]
            dis_loss = results[1]
            if params.autoencoder:
                recon_losses.append(results[2])
            if model.summary_var is not None:
                summary = results[-1]
                model.train_writer.add_summary(summary, summary_idx)
            summary_idx+=1
            dis_losses.append(np.mean(dis_loss))
            if batch_idx % 10 == 0:
                end = time.time()
                elapsed = time.time() - start
                start = end
                status = 'batch %d %d-%d, dis %f, t=%.4f' % (summary_idx, batch_idx-10,batch_idx,
                                                               np.mean(dis_losses[-1:-11:-1]),
                                                               elapsed/10)
                if params.autoencoder:
                    status += ' recon %.4f' % np.mean(recon_losses[-1:-11:-1])
                print status
                #saver.save(model.sess, params.model_name,global_step=0)
        if epoch_end():
            print 'save', params.model_name
            saver.save(model.sess, params.model_name,global_step=epoch+1)
            last_epoch_with_loss_improvement = epoch
        print "Best Epoch", last_epoch_with_loss_improvement
        if epoch - last_epoch_with_loss_improvement >= params.patience:
            print 'Early Stopping'
            break
        epoch+=1

def get_xv_split(identities, params):
    if params.multi_factor:
        ids_to_split = transform_multi_factor_to_single(identities)
        assert len(ids_to_split) == identities.shape[0]
    else:
        ids_to_split = identities
    unique_ids = np.array(sorted(np.unique(ids_to_split)))
    print unique_ids
    kf = KFold(len(unique_ids), n_folds=params.cross_validation_n_splits, shuffle=True,random_state=0)
    train_val_uidx, test_uidx = [(train,test) for train,test in kf][params.cross_validation_split]
    train_val_ids = unique_ids[train_val_uidx]
    test_ids = unique_ids[test_uidx]
    train_ids, val_ids = train_test_split(train_val_ids, test_size=0.2, random_state=0)
    print 'train', train_ids[:5], 'val', val_ids[:5], 'test',test_ids[:5]
    print len(train_ids), len(val_ids), len(test_ids), len(unique_ids)
    test_idx = np.in1d(ids_to_split, test_ids)
    train_idx = np.in1d(ids_to_split, train_ids)
    val_idx = np.in1d(ids_to_split, val_ids)
    return train_idx, val_idx, test_idx

def get_model(params, load_data=True):
    import cuhk_model
    import market_model
    import norb_model
    import cub200
    import sprite_model
    if params.data_type == 'cuhk':
        model = cuhk_model.CUHKModel(n_id=params.n_id, params=params, lr=params.learning_rate)
        params.test_type = 'open_set'
    elif params.data_type == 'market':
        model = market_model.MarketModel(n_id=params.n_id, params=params, lr=params.learning_rate)
        params.test_type = 'open_set'
    elif params.data_type == 'cub200':
        model = cub200.CUB200Model(n_id=params.n_id, params=params, lr=params.learning_rate)
        params.test_type = 'open_set'
    elif params.data_type == 'sprites':
        model = sprite_model.SpriteModel(n_id=params.n_id, params=params, lr=params.learning_rate)
        params.test_type = 'open_set'
    elif params.data_type == 'norb':
        model = norb_model.NorbModel(n_id=params.n_id, params=params, lr=params.learning_rate)
        params.test_type = 'open_set'
    else:
        raise 'unknown data type', params.data_type

    # Load dataset THEN initialize -- do this so that the ae
    # can know stuff about the dataset, like the number of training identities
    if load_data:
        X, identities, train_idx, val_idx, test_idx = model.load_dataset(params)
        if params.cross_validation_n_splits > 0 and params.cross_validation_split > -1:
            train_idx, val_idx, test_idx = get_xv_split(identities, params)

    model.initialize()
    meta_name = '%s-%d.meta' % (params.model_name, params.model_epoch)
    saver = tf.train.Saver(max_to_keep=1)
    model.params = params
    if os.path.exists(meta_name):
        print 'Restoring weights from %s' % meta_name
        model_loc = '%s-%d' % (params.model_name,params.model_epoch)
        print saver.restore(model.sess, model_loc)
    if load_data:
        return model,saver, X, identities, train_idx, val_idx, test_idx
    else:
        return model,saver

def visualize_embedding(X,users,params, autoencoder,saver):
    import visualization
    with tf.device(params.device):
        visualization.visualize_embedding(autoencoder, X, users,
                                          '%s/id-%d.png' % (params.model_prefix, params.model_epoch),
                                          '%s/nid-%d.png' % (params.model_prefix, params.model_epoch) )

def get_params(params_func=None):
    class MyArgumentParser(argparse.ArgumentParser):
        def convert_arg_line_to_args(self, arg_line):
            return arg_line.split()
    parser = MyArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument('-c','--command',
        choices=('train','embedding','tsne_vis', 'visualize_dims',
                 'eval_ld', 'test', 'profile', 'test_gaussian', 'test_laplace', 'test_knn', 'test_linear',
                 'visualize_problems', 'tsne_simple', 'visualize_distance_histograms',
                 'visualize_dim_pairs', 'recons', 'embedding'
                 ), required=True)
    parser.add_argument('-ds','--dataset',default='../../lbpembedding/data/aa_2_19_16_filtered/')
    parser.add_argument('-d','--device',default='/cpu:0')
    parser.add_argument('-l','--loss',default='triplet_emb_loss')
    parser.add_argument('-oef','--output_embedding_file')
    parser.add_argument('-odi','--output_dims_file')
    parser.add_argument('-tl','--train_log_file',type=str)
    parser.add_argument('-s','--seed',type=int)
    parser.add_argument('-p','--patience',type=int,default=10)
    parser.add_argument('-bs','--batch_size',type=int,default=20,help='Number of classes sampled per minibatch')
    parser.add_argument('-dt','--data_type',choices=('cuhk','market', 'mnist', 'cars','cars2', 'synthetic_oriented_lines',
                                                     'yaleb', 'sprites', 'online_products', 'cub200', 'norb','celeba'), required=True)
    parser.add_argument('-epi','--examples_per_identity',type=int,default=1)
    parser.add_argument('-mn','--model_name',type=str,default='%s/model.ckpt'% (os.path.dirname(os.path.realpath(__file__))))
    parser.add_argument('-me','--model_epoch',type=int,default=0)
    parser.add_argument('-tf','--trace_file', type=str)
    parser.add_argument('-m', '--margin', type=float, default=0.1)
    parser.add_argument('-n_id',type=int, default=5)
    parser.add_argument('-max_k',type=int, default=20)
    parser.add_argument('-l2_normalize', type=int, default=1)
    parser.add_argument('-dist',type=str, choices=('cos','l2', 'l1', 'max_l1', 'max_l2', 'softmax_l1', 'p_norm', 'max_n'), default='cos')
    parser.add_argument('-p_norm_p', type=float, default=1.)
    parser.add_argument('-f_statistic_d',type=int,default=1)
    parser.add_argument('-binomial_dev_C', type=float,default=10.)
    parser.add_argument('-lr','--learning_rate', type=float, default=1e-4)
    parser.add_argument('-adam_beta1',type=float, default=0.9)
    parser.add_argument('-adam_beta2',type=float, default=0.999)
    parser.add_argument('-xvs', '--cross_validation_split', default=-1, type=int)
    parser.add_argument('-xvns', '--cross_validation_n_splits', default=-1, type=int)
    parser.add_argument('-as', '--advanced_summary',action='store_true')
    parser.add_argument('-ae', '--autoencoder',action='store_true')
    parser.add_argument('-ae_split_z', '--autoencoder_split_z',action='store_true')
    parser.add_argument('-ae_rlw', '--recon_loss_weight',type=float)
    parser.add_argument('-ae_clw', '--content_loss_weight',type=float,default=1.0)
    parser.add_argument('-beta_vae', type=float, default=0., help='Turns on beta-VAE and specifies the weight on the KL-divergence term')
    parser.add_argument('-ae_n_nid',type=int, default=0)
    parser.add_argument('-dmta','--disable_monitor_train_acc', action='store_true')
    parser.add_argument('-mf','--multi_factor', action='store_true')
    parser.add_argument('-norb_use_category_as_identity',action='store_true')
    parser.add_argument('-norb_use_10_levels_for_pose',action='store_true')
    if params_func: params_func(parser)
    params = parser.parse_args()
    if params.autoencoder and params.autoencoder_split_z and (params.ae_n_nid == 0):
        raise ValueError('If autoencoder is set, then n_nid must be set too and be nonzero')
    if params.dist == 'cos':
        params.dist_fun = lambda x,y: 1.- tf.reduce_sum( tf.multiply(x, y, axis=1 ))/2.0
    elif params.dist == 'l2':
        params.dist_fun = lambda x,y: tf.reduce_sum( (x-y)**2, axis=1 )
    elif params.dist == 'l1':
        params.dist_fun = lambda x,y: tf.reduce_sum( tf.abs(x-y), axis=1 )
    elif params.dist == 'max_l1':
        params.dist_fun = lambda x,y: tf.reduce_max( tf.abs(x-y), axis=1 )
    elif params.dist == 'max_l2':
        params.dist_fun = lambda x,y: tf.reduce_max( (x-y)**2, axis=1 )
    params.loss_methods = __import__(params.loss)
    if not params.model_name.startswith('/'):
        params.model_name = '%s/%s' % (os.getcwd(), params.model_name)
        params.model_prefix = os.path.dirname(params.model_name)
    if sys.argv[1][0] == '@':
        params.model_name = '%s/%s/m' % (os.getcwd(), os.path.dirname(sys.argv[1][1:]))
        params.model_prefix = os.path.dirname(params.model_name)
    if params.model_epoch == -1:
        models = os.listdir(params.model_prefix)
        models = filter(lambda m: '.meta' in m, models)
        models = sorted(models, key=lambda m: (len(m), m) )
        model_epoch = int(re.match(r'm-([\d]+).meta', models[-1]).groups()[0])
        params.model_epoch = model_epoch
    if params.seed:
        np.random.seed(params.seed)
    else:
        np.random.seed(1234)
    if not params.train_log_file:
        params.train_log_file = '%s/tl.txt' % os.path.dirname(params.model_name)
        print 'Using %s as train log file' % params.train_log_file
    return params

def print_recall_k(r_at_k,params):
    print ','.join( ['%.4f' % r_at_k for r_at_k in r_at_k] )

def main():
    params = get_params()
    print params
    model,saver, X, identities, train_idx, val_idx, test_idx = get_model(params)
    if params.command == 'train':
        train(X,identities,train_idx,val_idx,test_idx,params,model,saver)
    elif params.command =='tsne_vis':
        import tsne_visualization_2 as tsne_vis
        tsne_vis.visualize_embedding(model,X[train_idx], identities[train_idx], '%s/tsne-train.png' % (params.model_prefix))
        tsne_vis.visualize_embedding(model,X[val_idx], identities[val_idx], '%s/tsne-val.png' % (params.model_prefix) )
    elif params.command == 'embedding' and params.output_embedding_file:
        with tf.device(params.device):
            embedding = model.compute_nid_embeddings(X[test_idx])
            np.savez(params.output_embedding_file,
                     embedding=embedding,
                     identity=identities
                     )
    elif params.command == 'test':
        if model.params.test_type == 'open_set':
            test_embedding = model.compute_embeddings(X[test_idx] )
            test_recall_at_k = recall_at_k.evaluate(test_embedding, identities[test_idx], params)
            train_embedding = model.compute_embeddings(X[train_idx])
            train_recall_at_k = recall_at_k.evaluate(train_embedding, identities[train_idx], params)
            val_embedding = model.compute_embeddings(X[val_idx])
            val_recall_at_k = recall_at_k.evaluate(val_embedding, identities[val_idx], params)
            print 'test'
            print_recall_k(test_recall_at_k,params)
            print 'mean', np.mean(test_recall_at_k)
            print 'train'
            print_recall_k(train_recall_at_k,params)
            print 'mean', np.mean(train_recall_at_k)
            print 'val'
            print_recall_k(val_recall_at_k,params)
            print 'mean', np.mean(val_recall_at_k)
            with open('%s/test.txt' % params.model_prefix, 'w') as of:
                of.write('test,')
                of.write(','.join(map(str,test_recall_at_k)))
                of.write('\n')
                of.write('train,')
                of.write(','.join(map(str,train_recall_at_k)))
                of.write('\n')
                of.write('val,')
                of.write(','.join(map(str,val_recall_at_k)))
                of.write('\n')
        if params.autoencoder:
            test_recon_loss = model.compute_recon_loss(X[test_idx])
            train_recon_loss = model.compute_recon_loss(X[train_idx])
            val_recon_loss = model.compute_recon_loss(X[val_idx])
            print 'Reconstruction Train %.8f Valid %.8f Test %.8f' % (train_recon_loss, val_recon_loss, test_recon_loss)
    elif params.command == 'recons':
        n=40
        X_test = X[test_idx]
        examples = X_test[np.random.choice(range(X_test.shape[0]), n, replace=False)]
        recons, = model.sess.run([model.x_hat_rescaled], feed_dict={model.x:examples})
        import recon
        recon.save_recon_image(examples,recons,outfile='%s/recons-test.png' % params.model_prefix)

if __name__ == '__main__': main()
