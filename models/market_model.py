import numpy as np
import tensorflow as tf
import os
import numpy as np
import embedding as emb
import sys
import h5py
import deep_metric_learning

class MarketModel(deep_metric_learning.DMLModel):
    def load_dataset(self,params):
        ds = np.load(params.dataset)
        ids = ds['identity']
        ids = np.array([int(i) for i in ids])
        imagedata = ds['imagedata']
        test_indices_bool = ds['test_indices_bool']
        train_indices_bool = ds['train_indices_bool']
        valid_indices_bool = ds['val_indices_bool']
        print '%d train %d val %d test' % (train_indices_bool.sum(), valid_indices_bool.sum(), test_indices_bool.sum())
        return imagedata, ids, train_indices_bool, valid_indices_bool, test_indices_bool
    def get_output_size(self):
        return [None, 128, 64, 3]
