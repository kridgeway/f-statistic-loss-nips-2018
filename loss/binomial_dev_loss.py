import tensorflow as tf
import numpy as np
import batch_pair_generator as gen

def create_loss(model, z):
    model.pos_comps = pos_comps = tf.placeholder(tf.int32, [None,2])
    model.neg_comps = neg_comps = tf.placeholder(tf.int32, [None,2])
    pos_comps_l = tf.reshape(pos_comps[:,0], (-1,))
    s_pos = tf.reduce_sum( tf.multiply(tf.gather(z,pos_comps[:,0]), tf.gather(z,pos_comps[:,1])), axis=1 )
    s_neg = tf.reduce_sum( tf.multiply(tf.gather(z,neg_comps[:,0]), tf.gather(z,neg_comps[:,1])), axis=1 )
    # Settings from "Deep Metric Learning for Practical Person
    # Re-Identification, Yi Dong, et al, 2014"
    alpha=2.0
    beta=0.5
    c = model.params.binomial_dev_C
    epsilon = 1e-5
    pos_loss = tf.log( tf.exp(-alpha*(s_pos - beta) ) + 1.0 )
    avg_pos_loss = tf.reduce_mean( pos_loss )
    neg_loss = tf.log( tf.exp(-alpha*(s_neg - beta)*(-c)) + 1.0 + epsilon )
    avg_neg_loss = tf.reduce_mean( neg_loss )
    return avg_pos_loss + avg_neg_loss

def train_batches(model,X_train, ids, run_options=None, run_metadata=None):
    for batch in gen.generate_batch_pairs(model.params,ids, generate_triplets=False):
            feed_dict = {model.x:X_train[batch['batch_samples']],
                         model.pos_comps:batch['pos_comps'],
                         model.neg_comps:batch['neg_comps'] }
            yield batch['batch_idx'], feed_dict
