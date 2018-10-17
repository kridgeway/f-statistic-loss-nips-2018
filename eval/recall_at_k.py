import numpy as np
from scipy import stats
from joblib import Parallel, delayed
import scipy
import time
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import manhattan_distances
from scipy.spatial.distance import cdist

def find_l2_distances(X,Y):
    intersection = -2.* np.dot(X,Y.T)
    X_sum = np.sum(X**2,axis=1)
    Y_sum = np.sum(Y**2,axis=1)
    XY_sum = X_sum[:, np.newaxis] + Y_sum
    return XY_sum + intersection

def find_cos_distances(X,Y):
    return (1.-np.dot(X, Y.T))/2.0

def max_distances(X,Y, dist_fun):
    if dist_fun == 'max_l1':
        return cdist(X, Y, 'chebyshev')
    else: raise 'not implemented'

def p_norm_distances(X,Y, p):
    return cdist(X,Y,'minkowski',p=p)

def compute_softmax_dist(X,Y,X_idx,beta):
    l1_dists = np.abs(X[X_idx] - Y)
    exp_ax = np.exp(beta * l1_dists)
    numerator = np.sum(l1_dists * exp_ax, axis=1)
    denominator = np.sum(exp_ax, axis=1)
    result =  numerator / denominator
    return result

def softmax_distances(X,Y, beta):
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)
    results = Parallel(n_jobs=10, backend="threading", verbose=1)(
        delayed(compute_softmax_dist)(X,Y,X_idx,beta) for X_idx in xrange(X.shape[0])
    )
    output = np.zeros( (X.shape[0], Y.shape[0]), dtype=np.float32 )
    for X_idx in xrange(len(results)):
        output[X_idx,:] = results[X_idx]
    return output

def compute_max_n(X,Y,X_idx,n):
    l1_dists = np.abs(X[X_idx] - Y)
    max_n = np.argpartition(l1_dists,-n)[:,-n:]
    return np.sum(l1_dists[np.arange(l1_dists.shape[0])[:,None], max_n], axis=1)

def max_n_distances(X,Y,n):
    results = Parallel(n_jobs=10,backend="threading",verbose=1)(
        delayed(compute_max_n)(X,Y,X_idx,n) for X_idx in range(X.shape[0])
    )
    output = np.zeros( (X.shape[0], Y.shape[0]), dtype=np.float32 )
    for X_idx in xrange(len(results)):
        output[X_idx,:] = results[X_idx]
    return output

def evaluate(test_emb, test_id, params):
    unique_ids, unique_counts = np.unique(test_id,return_counts=True)
    unique_ids = unique_ids[unique_counts >= 2]
    good_test_indices = np.in1d(test_id,unique_ids)
    valid_test_embs = test_emb[good_test_indices]
    valid_test_ids = test_id[good_test_indices]
    n_correct_at_k = np.zeros(params.max_k)
    if len(test_emb) < 40000:
        if params.dist == 'cos':
            #distances = 1.-np.dot(valid_test_embs, test_emb.T)
            distances = find_cos_distances(valid_test_embs,test_emb)
        elif params.dist == 'l2':
            distances = find_l2_distances(valid_test_embs, test_emb)
        elif params.dist == 'l1':
            distances = manhattan_distances(valid_test_embs, test_emb)
        elif params.dist == 'max_l1' or params.dist == 'max_l2':
            distances = max_distances(valid_test_embs, test_emb, params.dist)
        elif params.dist == 'softmax_l1':
            distances = softmax_distances(valid_test_embs, test_emb, params.softmax_l1_beta)
        elif params.dist == 'p_norm':
            distances = p_norm_distances(valid_test_embs, test_emb, params.p_norm_p)
        elif params.dist == 'max_n':
            distances = max_n_distances(valid_test_embs, test_emb, params.max_n)
        for idx, valid_test_id in enumerate(valid_test_ids):
            k_sorted_indices = np.argsort(distances[idx])[1:]
            first_correct_position = np.where(test_id[k_sorted_indices] == valid_test_id)[0][0]
            if first_correct_position < params.max_k:
                n_correct_at_k[first_correct_position:] += 1
        return 100.*n_correct_at_k / len(valid_test_ids)
    else:
        #if params.dist == 'cos':
        #    metric='cosine'
        #else: metric = 'l2'
        metric = 'l2'
        nn = NearestNeighbors(n_neighbors=params.max_k+1, metric=metric, algorithm='kd_tree', n_jobs=-1).fit(test_emb)
        distances, indices = nn.kneighbors(valid_test_embs)
        for idx, valid_test_id in enumerate(valid_test_ids):
            k_sorted_indices = indices[idx]
            correct_positions = np.where(test_id[k_sorted_indices] == valid_test_id)[0][1:]
            first_correct_position = params.max_k
            if len(correct_positions)>0:
                first_correct_position = correct_positions[0] - 1
            if first_correct_position < params.max_k:
                n_correct_at_k[first_correct_position:] += 1
        return 100.*n_correct_at_k / len(valid_test_ids)
