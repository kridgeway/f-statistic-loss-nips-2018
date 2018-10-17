import numpy as np
import collections

def generate_for_batch(batch_ids, generate_triplets):
    pos_comps=[]
    neg_comps=[]
    triplet_pos_weights=[]
    triplet_neg_weights=[]
    triplet_comps=[] # index into pos_comps and neg_comps
    unique_batch_ids, unique_batch_id_counts = np.unique(batch_ids, return_counts=True)
    for batch_idx in xrange(len(batch_ids)):
        batch_id = batch_ids[batch_idx]
        negatives = np.where(batch_ids != batch_id)[0]
        positives = np.where(batch_ids == batch_id)[0]
        these_pos_comps = map(lambda idx: (batch_idx, idx), filter(lambda idx: idx > batch_idx, positives) )
        these_neg_comps = map(lambda idx: (batch_idx, idx), filter(lambda idx: idx > batch_idx, negatives) )
        pos_comps.extend(these_pos_comps)
        neg_comps.extend(these_neg_comps)
        if generate_triplets:
            # For each positive pair (a1,a2) , we can create 2*N triplets from N different-class samples
            # in both orders: (a1,a2,b1..N) and (a2,a1,b1..N)
            triplet_pos_weights.extend( [2.0*len(negatives)]  * len(these_pos_comps) )
            # Each negative pair (a1,b1) shows up in |a|-1 + |b|-1 triplets
            for negative_l, negative_r in these_neg_comps:
                n_r = unique_batch_id_counts[unique_batch_ids==batch_ids[negative_r]][0]
                triplet_neg_weights.append( float(len(positives)-1. + n_r-1.) )
    pos_comps = np.array(pos_comps)
    neg_comps = np.array(neg_comps)
    if generate_triplets:
        for idx, pos_comp in enumerate(pos_comps):
            a1,a2 = pos_comp
            if neg_comps.shape[0] > 0:
                negatives = (neg_comps[:,0]==a1)|(neg_comps[:,1]==a1)|(neg_comps[:,0]==a2)|(neg_comps[:,1]==a2)
                triplet_comps.extend([ (idx, negidx) for negidx in np.where(negatives)[0]])
    return pos_comps, neg_comps, triplet_comps, triplet_pos_weights, triplet_neg_weights

# Generates the "full" combinations of pairs for each minibatch
def generate_batch_pairs(params, ids, generate_triplets=True):
    if params.multi_factor:
        generators = [do_generate_batch_pairs(params,ids[:,factor_idx],generate_triplets) for factor_idx in xrange(ids.shape[1])]
        while len(generators)>0:
            for gidx,generator in enumerate(generators):
                gens_to_remove=[]
                try:
                    yield next(generator)
                except StopIteration:
                    gens_to_remove.append(gidx)
            for gidx in gens_to_remove[::-1]:
                del generators[gidx]
    else:
        for foo in do_generate_batch_pairs(params,ids,generate_triplets):
            yield foo

def do_generate_batch_pairs(params, ids, generate_triplets):
    batch_number=0
    min_samples_per_id = 2
    unique_ids = np.unique(ids)
    id_indices=collections.defaultdict(list)
    for idx,identity in enumerate(ids):
        id_indices[identity].append(idx)
    def randomize_order(id_indices, unique_ids):
        unique_ids = np.random.permutation(unique_ids)
        n_identities_left=0
        ids_to_remove = []
        for unique_id in unique_ids:
            if len(id_indices[unique_id]) >= min_samples_per_id:
                n_identities_left+=1
            else:
                ids_to_remove.append(unique_id)
            id_indices[unique_id] = np.random.permutation(id_indices[unique_id])
        unique_ids = unique_ids[ np.logical_not( np.in1d(unique_ids, ids_to_remove) ) ]
        return n_identities_left, id_indices, unique_ids
    n_identities_left, id_indices, unique_ids = randomize_order(id_indices,unique_ids)
    training=True
    cur_identity_idx=0
    n_samp_per_id = params.examples_per_identity
    while training:
        if cur_identity_idx >= len(unique_ids):
            cur_identity_idx = 0
            n_identities_left, id_indices, unique_ids = randomize_order(id_indices,unique_ids)
            if n_identities_left < 2:
                training = False
                continue
        batch_ids=[]
        batch_samples=[]
        for identity in unique_ids[cur_identity_idx:cur_identity_idx+params.batch_size]:
            samples = id_indices[identity][:n_samp_per_id]
            id_indices[identity] = np.delete(id_indices[identity], np.arange(n_samp_per_id))
            batch_ids.extend( [identity] * len(samples) )
            batch_samples.extend(samples)
        cur_identity_idx = cur_identity_idx + params.batch_size
        pos_comps, neg_comps, triplet_comps, triplet_pos_weights, triplet_neg_weights = \
            generate_for_batch(batch_ids, generate_triplets)
        batch_number+=1
        if len(pos_comps) > 0 and len(neg_comps) > 0 and ((len(triplet_comps) > 0) or not generate_triplets):
            #u,c = np.unique(batch_ids, return_counts=True)
            #print u,c
            yield {'batch_samples':batch_samples,
                'batch_ids': batch_ids,
                'pos_comps':pos_comps,
                'neg_comps':neg_comps,
                'batch_idx':batch_number,
                'triplet_pos_weights': triplet_pos_weights,
                'triplet_neg_weights': triplet_neg_weights,
                'trip_comps': triplet_comps
                }
        batch_samples=[]
        batch_ids=[]

