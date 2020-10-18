# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 02:27:20 2020
@author: ZhiningLiu1998
mailto: zhining.liu@outlook.com / v-zhinli@microsoft.com
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    f1_score, 
    average_precision_score, 
    matthews_corrcoef, 
    )
from sklearn.model_selection import train_test_split

class Rater():
    """Rater for evaluate classifiers performance on class imabalanced data.

    Parameters
    ----------
    metric :    {'aucprc', 'mcc', 'fscore'}, optional (default='aucprc')
        Specify the performance metric used for evaluation.
        If 'aucprc' then use Area Under Precision-Recall Curve.
        If 'mcc' then use Matthews Correlation Coefficient.
        If 'fscore' then use F1-score, also known as balanced F-score or F-measure.
        Passing other values raises an exception.

    threshold : float, optional (default=0.5)
        The threshold used for binarizing the predicted probability.
        It does not affect the AUCPRC score
    
    Attributes
    ----------
    metric_ : string
        The performance metric used for evaluation.

    threshold_ : float
        The predict threshold.
    """
    def __init__(self, metric='aucprc', threshold=0.5):

        if metric not in ['aucprc', 'mcc', 'fscore', 'bacc']:
            raise ValueError(f'Metric {metric} is not supported.\
                \nSupport metrics: [aucprc, mcc, fscore].')

        self.metric_ = metric
        self.threshold_ = threshold
        
    def score(self, y_true, y_pred):
        """Score function.

        Parameters
        ----------
        y_true : array-like of shape = [n_samples]
            The ground truth labels.

        y_pred : array-like of shape = [n_samples]
            The predict probabilities.

        Returns
        ----------
        score: float
        """
        if self.metric_ == 'aucprc':
            return average_precision_score(y_true , y_pred)
        elif self.metric_ == 'mcc':
            y_pred_b = y_pred.copy()
            y_pred_b[y_pred_b < self.threshold_] = 0
            y_pred_b[y_pred_b >= self.threshold_] = 1
            return matthews_corrcoef(y_true, y_pred_b)
        elif self.metric_ == 'fscore':
            y_pred_b = y_pred.copy()
            y_pred_b[y_pred_b < self.threshold_] = 0
            y_pred_b[y_pred_b >= self.threshold_] = 1
            return f1_score(y_true, y_pred_b)

def load_dataset(dataset_name):
    """Util function that load training/validation/test data from /data folder.

    Parameters
    ----------
    dataset_name : string
        Name of the target dataset.
        Train/validation/test data are expected to save in .csv files with 
        suffix _{train/valid/test}.csv. Labels should be at the last column 
        named with 'label'.

    Returns
    ----------
    X_train, y_train, X_valid, y_valid, X_test, y_test
        Pandas DataFrames / Series
    """
    df_train = pd.read_csv(f'data/{dataset_name}_train.csv')
    X_train = df_train[df_train.columns.tolist()[:-1]]
    y_train = df_train['label']
    df_valid = pd.read_csv(f'data/{dataset_name}_valid.csv')
    X_valid = df_valid[df_valid.columns.tolist()[:-1]] 
    y_valid = df_valid['label']
    df_test = pd.read_csv(f'data/{dataset_name}_test.csv')
    X_test = df_test[df_test.columns.tolist()[:-1]] 
    y_test = df_test['label']
    return  X_train.values, y_train.values, \
            X_valid.values, y_valid.values, \
            X_test.values, y_test.values

def histogram_error_distribution(y_true, y_pred, bins):
    """Util function that compute the error histogram.

    Parameters
    ----------
    y_true : array-like of shape = [n_samples]
        The ground truth labels.

    y_pred : array-like of shape = [n_samples]
        The predict probabilities.

    bins :   int, number of bins in the histogram

    Returns
    ----------
    hist :   array-like of shape = [bins]
    """
    error = np.absolute(y_true - y_pred)
    hist, _ = np.histogram(error, bins=bins)
    return hist

def gaussian_prob(x, mu, sigma):
    """The Gaussian function.

    Parameters
    ----------
    x :     float
        Input number.

    mu :    float
        Parameter mu of the Gaussian function.

    sigma : float
        Parameter sigma of the Gaussian function.

    Returns
    ----------
    output : float
    """
    return (1 / (sigma * np.sqrt(2*np.pi))) * np.exp(-0.5*np.power((x-mu)/sigma, 2))

def meta_sampling(y_pred, y_true, X, n_under_samples, mu, sigma, random_state=None):
    """The meta-sampling process in MESA.

    Parameters
    ----------
    y_pred : array-like of shape = [n_samples]
        The predict probabilities.

    y_true : array-like of shape = [n_samples]
        The ground truth labels.
    
    X :      array-like of shape = [n_samples, n_features]
        The original data to be meta-sampled.

    n_under_samples : int, <= n_samples
        The expected number of instances in the subset after meta-sampling. 

    mu :    float
        Parameter mu of the Gaussian function.

    sigma : float
        Parameter sigma of the Gaussian function.

    random_state : int or None, optional (default=None)
        If int, random_state is the seed used by the random number generator.
        If None, the random number generator is the RandomState instance used
        by np.random.

    Returns
    ----------
    X_subset : array-like of shape = [n_under_samples, n_features]
        The subset after meta-sampling.
    """
    sample_weights = gaussian_prob(np.absolute(y_true - y_pred), mu, sigma)
    X_subset = pd.DataFrame(X).sample(n_under_samples, weights=sample_weights, random_state=random_state)
    return X_subset

def imbalance_train_test_split(X, y, test_size, random_state=None):
    '''Train/Test split that guarantee same class distribution between split datasets.'''
    classes = np.unique(y)
    X_trains, y_trains, X_tests, y_tests = [], [], [], []
    for label in classes:
        inds = (y==label)
        X_label, y_label = X[inds], y[inds]
        X_train, X_test, y_train, y_test = train_test_split(
            X_label, y_label, test_size=test_size, random_state=random_state)
        X_trains.append(X_train)
        X_tests.append(X_test)
        y_trains.append(y_train)
        y_tests.append(y_test)
    X_train = np.concatenate(X_trains)
    X_test = np.concatenate(X_tests)
    y_train = np.concatenate(y_trains)
    y_test = np.concatenate(y_tests)
    return  X_train, X_test, y_train, y_test

def state_scale(state, scale):
    '''Scale up the meta-states.'''
    return state / state.sum() * 2 * scale

def memory_init_fulfill(args, memory):
    '''Initialize the memory.'''
    num_bins = args.num_bins
    memory_size = args.replay_size
    error_in_bins = np.linspace(0, 1, num_bins)
    mu = 0.3
    unfitted, midfitted, fitted = \
        gaussian_prob(error_in_bins, 1, mu), \
        gaussian_prob(error_in_bins, 0.5, mu), \
        gaussian_prob(error_in_bins, 0, mu)
    underfitting_state = state_scale(np.concatenate([unfitted, unfitted]), num_bins)
    learning_state = state_scale(np.concatenate([midfitted, midfitted]), num_bins)
    overfitting_state = state_scale(np.concatenate([fitted, midfitted]), num_bins)
    noise_scale = 0.5
    num_per_transitions = int(memory_size/3)
    for i in range(num_per_transitions):
        state = underfitting_state + np.random.rand(num_bins*2) * noise_scale
        next_state = underfitting_state + np.random.rand(num_bins*2) * noise_scale
        memory.push(state, [0.9], args.reward_coefficient * 0.05, next_state, 0)
    for i in range(num_per_transitions):
        state = learning_state + np.random.rand(num_bins*2) * noise_scale
        next_state = learning_state + np.random.rand(num_bins*2) * noise_scale
        memory.push(state, [0.5], args.reward_coefficient * 0.05, next_state, 0)
    for i in range(num_per_transitions):
        state = overfitting_state + np.random.rand(num_bins*2) * noise_scale
        next_state = overfitting_state + np.random.rand(num_bins*2) * noise_scale
        memory.push(state, [0.1], args.reward_coefficient * 0.05, next_state, 0)
    return memory
    
def transform(y):
    if y.ndim == 1:
        y = y[:, np.newaxis]
    if y.shape[1] == 1:
        y = np.append(1-y, y, axis=1)
    return y

def cross_entropy(y_pred, y_true, epsilon=1e-4):
    '''Cross-entropy error function.'''
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    y_pred = transform(y_pred)
    y_true = transform(y_true)
    return (-y_true*np.log(y_pred)).sum(axis=1)

def slide_mean(data, window_half):
    '''Slide mean for better visualization.'''
    result = []
    for i in range(len(data)):
        lower_bound = max(i-window_half, 0)
        upper_bound = min(i+window_half+1, len(data)-1)
        result.append(np.mean(data[lower_bound:upper_bound]))
    return result