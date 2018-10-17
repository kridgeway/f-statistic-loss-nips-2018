import tensorflow as tf
import numpy as np
import batch_pair_generator as gen

# estimate the histogram using the assigments of points to grid bins
def getDistributionDensity(x, n, grid, grid_delta):
    grid_size = grid.get_shape().as_list()[0]
    #[-1.00, -0.90, -0.80, -0.70, -0.60, -0.50, -0.40, -0.30, -0.20, -0.10, \
        #-0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
    def process_grid_cell(i):
        def get_left_add():
            inds = tf.reshape( tf.where(tf.logical_and(x >= grid[i-1], x < grid[i]) ), (-1,) )
            left_dist = tf.gather(x, inds)
            left_add = tf.reduce_sum(left_dist - grid[i-1])
            return left_add
        def get_right_add():
            inds = tf.reshape(tf.where(tf.logical_and(x >= grid[i], x < grid[i+1])), (-1,) )
            right_dist = tf.gather(x, inds)
            right_add = tf.reduce_sum(grid[i+1] - right_dist)
            return right_add
        left_add = tf.cond( i > 0, get_left_add, lambda: tf.constant(0.0, dtype=tf.float32))
        right_add = tf.cond( i < grid_size-1, get_right_add, lambda: tf.constant(0.0, dtype=tf.float32))
        return left_add + right_add
    p_list = tf.map_fn(process_grid_cell, np.arange(grid_size, dtype=np.int32), dtype=tf.float32)
    p = tf.concat(p_list, axis=0)
    p = p / ( tf.cast(n, tf.float32) * grid_delta)
    """
    for i in xrange(len(grid)):
        left_add = 0
        if i > 0:
            d_i_list_left = np.array(bins_to_places[i])
            left_dist = np.array([x[ii] for ii in d_i_list_left])
            left_add = sum(left_dist - grid[i - 1])
        right_add = 0
        if i < len(grid) - 1:
            d_i_list_right = np.array(bins_to_places[i + 1])
            right_dist = np.array([x[ii] for ii in d_i_list_right])
            right_add = sum(grid[i + 1] - right_dist)
        p[i] = (left_add + right_add)
    p /= len(x) * grid_delta
    """
    return p

# Calculates probability of wrong order in pairs' similarities: positive pair less similar than negative one
# (this corresponds to 'simple' loss, other variants ('linear', 'exp') are generalizations that take into account
# not only the order but also the difference between the two similarity values).
# Can use histogram and beta-distribution to fit input data.
def create_loss(model, z):
    grid_delta = 0.01
    grid_arr = np.array([i for i in np.arange(-1., 1. + grid_delta, grid_delta)])
    grid_len = len(grid_arr)
    grid = tf.constant(grid_arr, dtype=tf.float32)
    L = np.ones((grid_len, grid_len))
    for i in xrange(grid_len):
        L[i] = grid_arr[i] <= grid_arr
    L = tf.constant(L, dtype=tf.float32)

    model.pos_comps = pos_comps = tf.placeholder(tf.int32, [None,2])
    model.n_pos_comps = n_pos_comps = tf.placeholder(tf.int32, ())
    model.neg_comps = neg_comps = tf.placeholder(tf.int32, [None,2])
    model.n_neg_comps = n_neg_comps = tf.placeholder(tf.int32, ())

    pos_comps_l = tf.reshape(pos_comps[:,0], (-1,))
    d_pos = tf.reduce_sum( tf.multiply(tf.gather(z,pos_comps[:,0]), tf.gather(z,pos_comps[:,1])), axis=1 )
    d_neg = tf.reduce_sum( tf.multiply(tf.gather(z,neg_comps[:,0]), tf.gather(z,neg_comps[:,1])), axis=1 )

    distr_pos = getDistributionDensity(d_pos, n_pos_comps, grid, grid_delta)
    distr_neg = getDistributionDensity(d_neg, n_neg_comps, grid, grid_delta)
    distr_pos = tf.reshape(distr_pos, (1,-1))
    distr_neg = tf.reshape(distr_neg, (-1,1))
    result = tf.matmul( tf.matmul(distr_pos, L), distr_neg )
    return result

def train_batches(model,X_train, ids, run_options=None, run_metadata=None):
    batch_idx=0
    losses = [model.optimizer, model.cost]
    for batch in gen.generate_batch_pairs(model.params, ids, generate_triplets=False):
        feed_dict = {model.x:X_train[batch['batch_samples']],
                        model.pos_comps:batch['pos_comps'],
                        model.neg_comps:batch['neg_comps'],
                        model.n_pos_comps:len(batch['pos_comps']),
                        model.n_neg_comps:len(batch['neg_comps'])
                        }
        yield batch['batch_idx'], feed_dict
