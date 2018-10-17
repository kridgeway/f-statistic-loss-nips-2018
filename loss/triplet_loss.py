import tensorflow as tf
import numpy as np
import batch_pair_generator as gen

def create_loss(model, z):
    margin = model.params.margin
    model.pos_comps = pos_comps = tf.placeholder(tf.int32, [None,2], name='pos_comps')
    model.neg_comps = neg_comps = tf.placeholder(tf.int32, [None,2], name='neg_comps')
    d_pos = model.params.dist_fun(tf.gather(z,pos_comps[:,0]), tf.gather(z,pos_comps[:,1]) )
    d_neg = model.params.dist_fun(tf.gather(z,neg_comps[:,0]), tf.gather(z,neg_comps[:,1]) )
    model.trip_comps = trip_comps = tf.placeholder(tf.int32, [None,2], name='trip_comps')
    d_pos_trip = tf.gather(d_pos, trip_comps[:,0])
    d_neg_trip = tf.gather(d_neg, trip_comps[:,1])
    return tf.reduce_mean( tf.maximum(0., d_pos_trip + margin - d_neg_trip) )

def train_batches(model,X_train, ids, run_options=None, run_metadata=None):
    for batch in gen.generate_batch_pairs(model.params, ids, generate_triplets=True):
            feed_dict = {model.x:X_train[batch['batch_samples']],
                         model.pos_comps:batch['pos_comps'],
                         model.neg_comps:batch['neg_comps'],
                         model.trip_comps:batch['trip_comps']
                         }
            yield batch['batch_idx'], feed_dict

