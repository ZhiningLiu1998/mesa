# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 14:32:27 2019
@author: v-zhinli
mailto: znliu19@mails.jlu.edu.cn / zhining.liu@outlook.com
"""

from imblearn.under_sampling import (
    ClusterCentroids, 
    NearMiss, 
    RandomUnderSampler, 
    EditedNearestNeighbours, 
    AllKNN, 
    TomekLinks, 
    OneSidedSelection, 
    RepeatedEditedNearestNeighbours, 
    CondensedNearestNeighbour, 
    NeighbourhoodCleaningRule,
)
from imblearn.over_sampling import (
    RandomOverSampler, SMOTE, ADASYN, BorderlineSMOTE,
)
from imblearn.combine import (
    SMOTEENN, SMOTETomek,
)

from sklearn.tree import DecisionTreeClassifier as DT
from collections import Counter
from time import clock
import pandas as pd

class Error(Exception):
    pass

class Resample_classifier(object):
    '''
    Re-sampling methods for imbalance classification, based on imblearn python package.
    imblearn url: https://github.com/scikit-learn-contrib/imbalanced-learn
    Hyper-parameters:
        base_estimator : scikit-learn classifier object
            optional (default=DecisionTreeClassifier)
            The base estimator used for training after re-sampling
    '''
    def __init__(self, base_estimator=DT(), resample_by='ORG'):
        self.base_estimator = base_estimator
        self.resample_by = resample_by

    def fit(self, X_train, y_train, verbose=False):
        start_time = clock()
        X_train_resampled, y_train_resampled = self.resample(X_train, y_train, by=self.resample_by)
        end_time = clock()
        self._last_resample_info = 'Resampling method: {}, class distribution from {} to {}, time used {}s'.format(
                self.resample_by, dict(Counter(y_train)), dict(Counter(y_train_resampled)), end_time - start_time,
                )
        if verbose:
            print (self._last_resample_info)
        self.base_estimator.fit(X_train_resampled, y_train_resampled)
    
    def predict(self, X):
        return self.base_estimator.predict(X)

    def predict_proba(self, X):
        return self.base_estimator.predict_proba(X)
    
    def resample(self, X, y, by, random_state=None):
        '''
        by: String
            The method used to perform re-sampling
            currently support: ['RUS', 'CNN', 'ENN', 'NCR', 'Tomek', 'ALLKNN', 'OSS',
                'NM', 'CC', 'SMOTE', 'ADASYN', 'BorderSMOTE', 'SMOTEENN', 'SMOTETomek',
                'ORG']
        '''
        if by == 'RUS':
            sampler = RandomUnderSampler(random_state=random_state)
        elif by == 'CNN':
            sampler = CondensedNearestNeighbour(random_state=random_state)
        elif by == 'ENN':
            sampler = EditedNearestNeighbours(random_state=random_state)
        elif by == 'NCR':
            sampler = NeighbourhoodCleaningRule(random_state=random_state)
        elif by == 'Tomek':
            sampler = TomekLinks(random_state=random_state)
        elif by == 'ALLKNN':
            sampler = AllKNN(random_state=random_state)
        elif by == 'OSS':
            sampler = OneSidedSelection(random_state=random_state)
        elif by == 'NM':
            sampler = NearMiss(random_state=random_state)
        elif by == 'CC':
            sampler = ClusterCentroids(random_state=random_state)
        elif by == 'ROS':
            sampler = RandomOverSampler(random_state=random_state)
        elif by == 'SMOTE':
            sampler = SMOTE(random_state=random_state)
        elif by == 'ADASYN':
            sampler = ADASYN(random_state=random_state)
        elif by == 'BorderSMOTE':
            sampler = BorderlineSMOTE(random_state=random_state)
        elif by == 'SMOTEENN':
            sampler = SMOTEENN(random_state=random_state)
        elif by == 'SMOTETomek':
            sampler = SMOTETomek(random_state=random_state)
        elif by == 'ORG':
            sampler = None
        else:
            raise Error('Unexpected \'by\' type {}'.format(by))
        
        if by != 'ORG':
            X_train, y_train = sampler.fit_resample(X, y)
        else:
            X_train, y_train = X, y

        return X_train, y_train