import tensorflow as tf
import numpy as np
import batch_pair_generator as gen
import scipy.special as sc

def create_loss(model, z):
    if 'batch_ids' not in dir(model):
        model.batch_ids = batch_ids = tf.placeholder(tf.int32, (None,), name='batch_ids')
        model.n_classes = n_classes = tf.placeholder(tf.int32, (), name='n_classes')
    else:
        batch_ids = model.batch_ids
        n_classes = model.n_classes
    unique_classes, idx, unique_counts = tf.unique_with_counts(batch_ids)
    model.unique_classes = unique_classes

    z_nd = tf.shape(z)[1]

    def process_class_within(class_id):
        this_class_idx = tf.where(tf.equal(batch_ids,class_id))
        this_class = tf.gather(z, this_class_idx, name='this_class')
        this_class = tf.reshape(this_class, (-1, z_nd ))
        this_class_mean = tf.reduce_mean(this_class,axis=0,name='this_class_mean')
        within_class_diffs = (this_class - this_class_mean)**2.0
        #within class sum for each dimension
        within_class_diff_sum = tf.reduce_sum(within_class_diffs, axis=0)
        return this_class_mean, within_class_diff_sum

    class_means, within_diffs = tf.map_fn(process_class_within, unique_classes, dtype=(tf.float32,tf.float32))
    within_epsilon = 1e-4
    between_epsilon = 1e-9

    def compute_f_pairs(cidx):
        l_mean = class_means[cidx]
        r_means = class_means[cidx+1:]
        l_within = within_diffs[cidx]
        r_withins = within_diffs[cidx+1:]
        pair_global_means = (l_mean + r_means)/2.0
        l_between = (l_mean-pair_global_means)**2
        r_betweens = (r_means-pair_global_means)**2
        pair_within_diffs = l_within + r_withins
        l_count = unique_counts[cidx]
        r_counts = unique_counts[cidx+1:]
        pair_between_diffs = l_between * tf.cast(l_count, tf.float32) + r_betweens * tf.reshape(tf.cast(r_counts,tf.float32), [-1,1])
        pair_counts = l_count + r_counts
        def compute_pair_f(item):
            between,within,pair_count,r_mean = item
            d1 = 1.0 #(two classes) - 1
            d2 = tf.cast(pair_count, tf.float32) - 2.0 #n - n_classes
            x = (between / (between + within))
            # Clip the probabilities to prevent numerical instability
            xmin = 1.0e-37
            xmax = 1. - 1.0e-5
            x_limit = tf.clip_by_value(x, xmin, xmax)
            xbetainc = tf.betainc(d1/2.0, d2/2.0, x_limit)
            d = model.params.f_statistic_d
            top_k, top_k_ind = tf.nn.top_k(xbetainc, k=d, sorted=False)
            result = tf.reduce_sum(tf.log(top_k))
            return result
        pair_cdfs = tf.map_fn(compute_pair_f, (pair_between_diffs, pair_within_diffs,pair_counts, r_means),
                              dtype=tf.float32, name='pair_cdfs')
        return tf.reduce_sum(pair_cdfs)
    class_indices = (tf.cumsum(tf.ones_like(unique_classes)) - 1)[:-1]
    n_classes = tf.cast( tf.reduce_sum( tf.ones_like(unique_classes) ), tf.float32 )

    def compute_global_xbetainc():
        global_mean = tf.reduce_mean(class_means, axis=0, name='global_mean')
        class_diffs = (class_means - global_mean)**2.
        class_counts = tf.reshape(tf.cast(unique_counts,tf.float32), (-1, 1) )
        between = tf.reduce_sum(tf.multiply(class_counts, class_diffs), axis=0)
        within = tf.reduce_sum(within_diffs, axis=0)
        d1 = n_classes - 1
        d2 = tf.cast(tf.reduce_sum(unique_counts), tf.float32) - n_classes
        s = (between / d1) / (within / d2)
        xmin = 1.0e-20
        xmax = 1. - xmin
        return tf.clip_by_value(1.-tf.betainc(d1/2., d2/2., (d1*s) / (d1*s + d2) ), xmin, xmax)
    cdfs = tf.map_fn(compute_f_pairs, class_indices, dtype=tf.float32, name='compute_f_pairs')
    cdfs = tf.reshape(cdfs, (-1,) )
    tf.summary.scalar('F-statistic loss', -tf.reduce_sum(cdfs)  )
    return -tf.reduce_sum(cdfs)

def train_batches(model,X_train, ids, run_options=None, run_metadata=None):
    for batch in gen.generate_batch_pairs(model.params,ids, generate_triplets=False):
            feed_dict = {model.x:X_train[batch['batch_samples']],
                         model.batch_ids:batch['batch_ids'],
                         model.n_classes:len(np.unique(batch['batch_ids']))}
            yield batch['batch_idx'], feed_dict
