import numpy as np
import estimate_mi_cont_disc
import pandas as pd
from joblib import Parallel, delayed
import math
import embedding
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LogisticRegression
import scipy.stats
from sklearn.metrics import roc_auc_score

def add_params(parser):
    parser.add_argument('-use_whole_set',action='store_true')
    parser.add_argument('-use_ids',action='store_true')
    parser.add_argument('-style',action='store_true')
    parser.add_argument('-plot',action='store_true')
    parser.add_argument('-show',action='store_true')
    parser.add_argument('-wa', '--weighted_average_deviation',action='store_true')

def compute_deviations(mutual_infos,factor_names=None):
    n_z = mutual_infos.shape[0]
    n_l = mutual_infos.shape[1]
    deviations = np.zeros(n_z)
    thetas = np.zeros(n_z)
    for zidx in xrange(n_z):
        row = mutual_infos[zidx,:]
        max_mi_idx = np.argmax(row)
        theta = mutual_infos[zidx,max_mi_idx]
        template = np.zeros(n_l)
        template[max_mi_idx] = theta
        dist = np.sum((row-template)**2.) / np.sum( (theta**2.) *(n_l-1))
        if not (factor_names is None):
            print zidx, max_mi_idx, factor_names[max_mi_idx], row[max_mi_idx], dist, row
        else:
            print zidx, max_mi_idx, row[max_mi_idx], dist, row
        deviations[zidx] = dist
        thetas[zidx] = theta
    return deviations, thetas

def compute_mutual_infos(z, factors):
    n_z = z.shape[1]
    n_l = factors.shape[1]
    mutual_infos = np.zeros( (n_z, n_l), dtype=np.float32)
    for zidx in xrange(n_z):
        for lidx in xrange(n_l):
            code = z[:,zidx]
            vals = factors[:,lidx]
            mutual_infos[zidx,lidx] = estimate_mi_cont_disc.calculate_mi(code,vals,bins=20)
    return mutual_infos

def make_mi_csd_plot(factors,style_factors,params,model,mis):
    nsf = style_factors.shape[1]
    nf = factors.shape[1] - nsf
    csd_mi = np.array( (np.sum(mis[:,:nf],axis=1), np.sum(mis[:,nf:],axis=1) ) ).T
    csd_deviations, thetas = compute_deviations(csd_mi)
    sort_order = np.argsort(csd_mi[:,0])
    mis = mis[sort_order,:]
    csd_mi = csd_mi[sort_order,:]
    print 'csd deviations', csd_deviations
    print 'CSD Modularity', np.mean(csd_deviations)
    if params.plot:
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        plt.figure(figsize=(4.5,9))
        plt.subplot(1,2,1)
        plt.suptitle('%s - Mutual Information' % params.loss)
        plt.pcolor(mis)
        plt.xticks([nf/2,nf+nsf/2], ['identity','style'] )
        plt.ylabel('Embedding Dimension, sorted by sum(identity)')
        plt.subplot(1,2,2)
        plt.pcolor(csd_mi)
        plt.xticks([0.5,1.5], ['sum(identity)','sum(style)'])
        plt.subplots_adjust(left=0.12, right=0.98, top=0.92, bottom=0.05)
        plt.savefig('%s/modularity.png' % params.model_prefix,transparent=True)
        if params.show:
            plt.show()

def make_mi_plot(factors,factor_names,params,model,mis):
    nf = factors.shape[1]
    if params.plot:
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7,9))
        plt.suptitle('%s - Mutual Information' % params.loss)
        plt.pcolor(mis)
        #plt.xticks([nf/2,nf+nsf/2], ['identity','style'] )
        plt.xticks(np.arange(nf)+.5, factor_names,rotation='vertical')
        plt.ylabel('Embedding Dimension')
        plt.subplots_adjust(left=0.12, right=0.98, top=0.92, bottom=0.2)
        plt.savefig('%s/modularity.png' % params.model_prefix,transparent=True)
        if params.show:
            plt.show()


if __name__ == '__main__':
    params = embedding.get_params(add_params)
    model,saver, X, ids, train_idx, val_idx, test_idx = embedding.get_model(params)
    if params.use_ids:
        factors = ids.reshape( (-1, 1))
    else:
        factors = model.factors
    if not params.use_whole_set:
        print 'Using test set'
        X = X[test_idx]
        ids = ids[test_idx]
        factors = factors[test_idx]

    if params.style:
        style_factors = model.style_factors
        if not params.use_whole_set:
            style_factors = style_factors[test_idx]
        factors = np.concatenate( (factors, style_factors), axis=1 )

    z = model.compute_embeddings(X)

    mis = compute_mutual_infos(z,factors)
    if 'factor_names' in dir(model):
        deviations, thetas = compute_deviations(mis, model.factor_names)
    else:
        deviations, thetas = compute_deviations(mis)

    if params.plot:
        if params.style:
            make_mi_csd_plot(factors,style_factors,params,model,mis)
        else:
            make_mi_plot(factors,model.factor_names,params,model,mis)

    df = pd.DataFrame({'zdim':np.arange(z.shape[1]), 'modularity':1.-deviations})
    if not params.use_whole_set: fn = '%s/modularity_test.txt' % params.model_prefix
    else: fn = '%s/modularity.txt' % params.model_prefix
    df.to_csv(fn,index=False)
    print 'Overall Modularity', np.mean(1.-deviations)
