import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import math
import embedding
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LogisticRegression
import scipy.stats
from sklearn.metrics import roc_auc_score

def calculate_explicitness(z, factors):
    n_z = z.shape[1]
    n_f = factors.shape[1]
    mean_aucs = np.zeros(n_f)
    all_aucs = []
    all_aucs_factors=[]
    all_aucs_factor_vals=[]
    for factor_idx in xrange(n_f):
        model = LogisticRegression(C=1e10)
        model.fit(z, factors[:,factor_idx])
        preds = model.predict_proba(z)
        aucs=[]
        for val_idx, val in enumerate(model.classes_):
            y_true = factors[:,factor_idx] == val
            y_pred = preds[:,val_idx]
            auc = roc_auc_score(y_true,y_pred)
            aucs.append(auc)
            all_aucs_factor_vals.append(val)
        mean_aucs[factor_idx] = np.mean(aucs)
        all_aucs.extend(aucs)
        all_aucs_factors.extend([factor_idx] * len(aucs))
    return mean_aucs, all_aucs, all_aucs_factors, all_aucs_factor_vals

if __name__ == '__main__':
    def add_params(parser):
        parser.add_argument('-use_whole_set',action='store_true')
        parser.add_argument('-all_aucs',action='store_true')
        parser.add_argument('-style',action='store_true')
    params = embedding.get_params(add_params)
    model,saver, X, ids, train_idx, val_idx, test_idx = embedding.get_model(params)
    factors = model.factors
    if not params.use_whole_set:
        print 'Using test set'
        X = X[test_idx]
        ids = ids[test_idx]
        factors = factors[test_idx]
    z = model.compute_embeddings(X)
    n_f = factors.shape[1]
    mean_aucs, all_aucs, all_aucs_factors, all_aucs_factor_vals = calculate_explicitness(z,factors)
    if params.all_aucs:
        df = pd.DataFrame({'factor_idx':all_aucs_factors,
                           'factor_val':all_aucs_factor_vals,
                           'auc': all_aucs})
    else:
        df = pd.DataFrame({'factor_idx': np.arange(n_f), 'mean_auc': mean_aucs})
    if not params.use_whole_set:
        fn = '%s/explicitness_test.txt' % params.model_prefix
    else:
        fn = '%s/explicitness.txt' % params.model_prefix
    df.to_csv(fn,index=False)
    print 'content', mean_aucs, np.mean(mean_aucs)

    if params.style:
        style_factors = model.style_factors
        if not params.use_whole_set:
            style_factors = style_factors[test_idx]
        #z_style = z[:,params.n_id:] if params.autoencoder_split_z else z
        mean_aucs, all_aucs, all_aucs_factors, all_aucs_factor_vals = calculate_explicitness(z,style_factors)
        print 'style', mean_aucs, np.mean(mean_aucs)
        df = pd.DataFrame({'factor_idx':all_aucs_factors,
                           'factor_val':all_aucs_factor_vals,
                           'auc': all_aucs})
        if not params.use_whole_set: fn = '%s/explicitness_test_style.txt' % params.model_prefix
        else: fn = '%s/explicitness_style.txt' % params.model_prefix
        df.to_csv(fn,index=False)
