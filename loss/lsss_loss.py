import tensorflow as tf
import numpy as np
import batch_pair_generator as gen

def create_loss(model, z):
    #Margin from "Deep Metric Learning via Lifted Structured Feature Embedding Query Retrieval"
    margin =1.0

    model.pos_comps = pos_comps = tf.placeholder(tf.int32, [None,2], name='pos_comps')
    model.neg_comps = neg_comps = tf.placeholder(tf.int32, [None,2], name='neg_comps')
    #model.n_pos_comps = n_pos_comps = tf.placeholder(tf.int32, (), 'n_pos_comps')
    n_pos_comps = tf.reduce_sum( tf.ones_like( pos_comps ) )

    #d_pos = 1.-tf.reduce_sum( tf.multiply(tf.gather(z,pos_comps[:,0]), tf.gather(z,pos_comps[:,1])), axis=1 )
    #d_neg = 1.-tf.reduce_sum( tf.multiply(tf.gather(z,neg_comps[:,0]), tf.gather(z,neg_comps[:,1])), axis=1 )
    d_pos = tf.reduce_sum( (tf.gather(z,pos_comps[:,0])- tf.gather(z,pos_comps[:,1]))**2, axis=1 )
    d_neg = tf.reduce_sum( (tf.gather(z,neg_comps[:,0])- tf.gather(z,neg_comps[:,1]))**2, axis=1 )

    def process_pos_comp(foo):
        pos_comp, pos = foo
        left = pos_comp[0]
        right = pos_comp[1]
        neg_locations_bool = tf.logical_or(
            tf.logical_or(
                tf.logical_or(
                    tf.equal(neg_comps[:,0], left),
                    tf.equal(neg_comps[:,0], right)
                ),
                tf.equal(neg_comps[:,1], left)
            ),
            tf.equal(neg_comps[:,1], right)
        )
        neg_locations = tf.reshape(tf.where(neg_locations_bool, name='neg_locations_where'), (-1,))
        negs = tf.gather(d_neg, neg_locations)
        neg_loss = tf.log(tf.reduce_sum( tf.exp(margin - negs) ))
        return tf.maximum(0.0, neg_loss + pos)

    losses = tf.map_fn(process_pos_comp, (pos_comps, d_pos), dtype=tf.float32)
    return 1./(2.0*tf.cast(n_pos_comps, tf.float32)) * tf.reduce_sum(losses)

def train_batches(model,X_train, ids, run_options=None, run_metadata=None):
    for batch in gen.generate_batch_pairs(model.params,ids, generate_triplets=False):
            feed_dict = {model.x:X_train[batch['batch_samples']],
                         model.pos_comps:batch['pos_comps'],
                         model.neg_comps:batch['neg_comps'],
                         #model.n_pos_comps:len(batch['pos_comps'])
                         }
            yield batch['batch_idx'], feed_dict
