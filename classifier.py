import calour as ca
import calour_utils as cu

import numpy as np
import glob
import os
import pandas as pd
import shutil
import calour

def classifier_performance_matrix(exp: ca.Experiment, use_subset_features=True, shuffle=False, shuffle_source=False):
    '''test the cross cohort classifier performance
    
    Parameters
    ----------
    exp: ca.Experiment with the different cohort subsets
    use_subset_features: bool, optional
        True to filter features for classifier before each cross-cohort classification (feature intersection of the 2 cohorts)
        False to use all the exp features for classifiers
    shuffle: bool, optional
        if True, randomize the HC/disease labels of validation cohort before predicting using the model (for null hypothesis values)
    shuffle_source: bool, optional
        if True, randomize the HC/disease labels of training cohort before predicting (for null hypothesis values)
    
    Returns
    -------
    ids1: the 'expid' field order for the prediction matrix rows
    ids2: the 'expid' field order for the prediction matrix columns
    roc_mat:
        2d numpy.array. row is the training experiment, column is the testing experiment, value is the ROC
    '''
    ca.set_log_level('ERROR')
    num_exp = len(exp.sample_metadata['expid'].unique())
    print('processing %d experiments' % num_exp)
    roc_mat = np.zeros([num_exp, num_exp])

    ids1 = []
    ids2 = []
    for idx1, (id1,exp1) in enumerate(exp.iterate('expid')):
        if shuffle_source:
            exp1.sample_metadata['type'] = exp1.sample_metadata['type'].sample(frac=1).values
        exp1 = exp1.filter_sum_abundance(0,strict=True)
        ids1.append(id1)
        # if not subset of features, train once on the exp1 dataset
        if not use_subset_features:
            model=cu.classify_fit(exp1,'type')
        for idx2, (id2, exp2) in enumerate(exp.iterate('expid')):
            ids2.append(id2)
            exp2 = exp2.filter_sum_abundance(0,strict=True)
            cexp1 = exp1.filter_ids(exp2.feature_metadata.index)
            cexp2 = exp2.filter_ids(cexp1.feature_metadata.index)
            # if shuffle, mix the HC/disease of exp2
            if shuffle:
                cexp2.sample_metadata['type'] = cexp2.sample_metadata['type'].sample(frac=1).values

            # if same experiment, so do training/validation
            if idx1 == idx2:
                # keep 2/3 of samples
                cexp1 = cexp1.downsample('expid',keep=2*int(len(cexp1.sample_metadata)/3))
                cexp2 = cexp2.filter_ids(cexp1.sample_metadata.index,axis='s',negate=True)

            # if using subset features, train the model after subsetting
            if use_subset_features:
                model=cu.classify_fit(cexp1,'type')
        
            # now predict on exp2 and measure performance
            res=cu.classify_predict(cexp2, 'type', model,plot_it=False)
            roc_auc = cu.classify_get_roc(res)
            roc_mat[idx1,idx2] = roc_auc
    ca.set_log_level('INFO')
    return ids1, ids2, roc_mat
