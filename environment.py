# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 02:27:20 2020
@author: ZhiningLiu1998
mailto: zhining.liu@outlook.com / v-zhinli@microsoft.com
"""

import pandas as pd
import numpy as np
import sklearn
import warnings
warnings.filterwarnings("ignore")

from utils import (
    Rater, meta_sampling, histogram_error_distribution, imbalance_train_test_split,
    )

class Ensemble():
    """A basic ensemble learning framework.

    Parameters
    ----------
    base_estimator : object (scikit-learn classifier)
        The base estimator used to build ensemble classifiers.
        NO need to support sample weighting. 
        Built-in `fit()`, `predict()`, `predict_proba()` methods are required.
    
    Attributes
    ----------
    base_estimator_ : estimator
        The base estimator from which the ensemble is grown.

    estimators_ : list of estimators
        The collection of fitted sub-estimators.
    """
    def __init__(self, base_estimator):
        self.estimators_ = []
        if not sklearn.base.is_classifier(base_estimator):
            raise TypeError(f'Base estimator {base_estimator} is not a sklearn classifier.')
        self.base_estimator_ = base_estimator

    def fit_step(self, X, y):
        """Bulid a new base classifier from the training set (X, y).

        Parameters
        ----------
        y : array-like of shape = [n_samples]
            The training labels.

        X : array-like of shape = [n_samples, n_features]
            The training instances.

        Returns
        ----------
        self : object (Ensemble)
        """
        self.estimators_.append(
            sklearn.base.clone(self.base_estimator_).fit(X, y)
            )
        return self

    def predict_proba(self, X):
        """Predict class probabilities for X.

        The predicted class probabilities of an input sample is computed as the  
        mean predicted class probabilities of the classifiers in the ensemble.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input data instances.
        
        Returns
        ----------
        p : array-like of shape [n_samples, n_classes]
            The class probabilities of the input samples.
        """
        y_pred = np.array(
            [model.predict_proba(X)[:, 1] for model in self.estimators_]
            ).mean(axis=0)
        if y_pred.ndim == 1:
            y_pred = y_pred[:, np.newaxis]
        if y_pred.shape[1] == 1:
            y_pred = np.append(1-y_pred, y_pred, axis=1)
        return y_pred
    
    def predict(self, X):
        """Predict classes for X.

        The predicted class of an input sample is computed as the mean 
        prediction of the classifiers in the ensemble.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input data instances.
        
        Returns
        ----------
        y : array-like of shape = [n_samples]
            The predicted classes.
        """
        y_pred_binarized = sklearn.preprocessing.binarize(
            self.predict_proba(X)[:,1].reshape(1,-1), threshold=0.5)[0]
        return y_pred_binarized
    
    def score(self, X, y):
        """Return area under precision recall curve (AUCPRC) scores for X, y.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input data instances.

        y : array-like of shape = [n_samples]
            Labels for X.
        
        Yields
        ----------
        z : float
        """
        yield sklearn.metrics.average_precision_score(
            y, self.predict_proba(X)[:, 1])

class EnsembleTrainingEnv(Ensemble):
    """The ensemble training environment in MESA.

    Parameters
    ----------
    args : arguments
        See arguments.py for more information.

    base_estimator : object (scikit-learn classifier)
        The base estimator used to build ensemble classifiers.
        NO need to support sample weighting. 
        Built-in `fit()`, `predict()`, `predict_proba()` methods are required.
    
    Attributes
    ----------
    args : arguments

    rater : object (Rater)
        Rater for evaluate classifiers performance on class imabalanced data.
        See arguments.py for more information.

    base_estimator_ : object (scikit-learn classifier)
        The base estimator from which the ensemble is grown.

    estimators_ : list of classifiers
        The collection of fitted sub-estimators.
    """
    def __init__(self, args, base_estimator):

        super(EnsembleTrainingEnv, self).__init__(
            base_estimator=base_estimator)

        self.base_estimator_ = base_estimator
        self.args = args
        self.rater = Rater(metric=args.metric)
    
    def load_data(self, X_train, y_train, X_valid, y_valid, X_test=None, y_test=None, train_ratio=1):
        """Load and preprocess the train/valid/test data into the environment."""
        self.flag_use_test_set = False if X_test is None or y_test is None else True
        if train_ratio < 1:
            print ('Using {:.2%} random subset for meta-training.'.format(train_ratio))
            _, X_train, _, y_train = imbalance_train_test_split(X_train, y_train, test_size=train_ratio)
        self.X_train, self.y_train = pd.DataFrame(X_train), pd.Series(y_train)
        self.X_valid, self.y_valid = pd.DataFrame(X_valid), pd.Series(y_valid)
        self.X_test,  self.y_test  = pd.DataFrame(X_test),  pd.Series(y_test)
        self.mask_maj_train, self.mask_min_train = (y_train==0), (y_train==1)
        self.mask_maj_valid, self.mask_min_valid = (y_valid==0), (y_valid==1)
        self.n_min_samples = self.mask_min_train.sum()
        n_samples = int(self.n_min_samples*self.args.train_ir)
        if n_samples > self.mask_maj_train.sum():
            raise ValueError(f"\
                Argument 'train_ir' should be smaller than imbalance ratio,\n \
                Please set this parameter to < {self.mask_maj_train.sum()/self.mask_min_train.sum()}.\
                ")
        self.n_samples = n_samples

    def init(self):
        """Reset the environment."""
        self.estimators_ = []
        # buffer the predict probabilities for better efficiency
        # initialize 
        self.y_pred_train_buffer = np.zeros_like(self.y_train)
        self.y_pred_valid_buffer = np.zeros_like(self.y_valid)
        if self.flag_use_test_set:
            self.y_pred_test_buffer  = np.zeros_like(self.y_test)
        self._warm_up()
    
    def get_state(self):
        """Fetch the current state of the environment."""
        hist_train = histogram_error_distribution(
            self.y_train[self.mask_maj_train], 
            self.y_pred_train_buffer[self.mask_maj_train], 
            self.args.num_bins)
        hist_valid = histogram_error_distribution(
            self.y_valid[self.mask_maj_valid], 
            self.y_pred_valid_buffer[self.mask_maj_valid], 
            self.args.num_bins)
        hist_train = hist_train / hist_train.sum() * self.args.num_bins
        hist_valid = hist_valid / hist_valid.sum() * self.args.num_bins
        state = np.concatenate([hist_train, hist_valid])
        return state
    
    def step(self, action, verbose=False):
        """Perform an environment step.

        Parameters
        ----------
        action: float, in [0, 1]
            The action (mu) to execute in the environment.

        verbose: bool, optional (default=False)
            Whether to compute and return the information about the current ensemble.
        
        Returns
        ----------
        next_state : array-like of shape [state_size]
            The state of the environment after executing the action.
        
        reward : float
            The reward of taking the action.
        
        done : bool
            Indicates the end of an episode.
            True if the ensemble reaches the maximum number of base estimators.
        
        info : string
            Information about the current ensemble.
            Empty string if verbose == False.
        """
        # check action value
        if action < 0 or action > 1:
            raise ValueError("Action must be a float in [0, 1].")

        # perform meta-sampling
        X_maj_subset = meta_sampling(
            y_pred = self.y_pred_train_buffer[self.mask_maj_train], 
            y_true = self.y_train[self.mask_maj_train],
            n_under_samples = self.n_samples,
            X = self.X_train[self.mask_maj_train],
            mu = action, 
            sigma = self.args.sigma, 
            random_state = self.args.random_state,)
        # build training subset (X_train_iter, y_train_iter)
        X_train_iter = pd.concat([X_maj_subset, self.X_train[self.mask_min_train]]).values
        y_train_iter = np.concatenate([np.zeros(X_maj_subset.shape[0]), np.ones(self.n_min_samples)])
        
        score_valid_before = self.rater.score(self.y_valid, self.y_pred_valid_buffer)

        # build a new base classifier from (X_train_iter, y_train_iter)
        self.fit_step(X_train_iter, y_train_iter)
        self.update_all_pred_buffer()

        score_valid = self.rater.score(self.y_valid, self.y_pred_valid_buffer)

        # obtain return values
        next_state = self.get_state()
        reward = score_valid - score_valid_before
        done = True if len(self.estimators_) >= self.args.max_estimators else False
        info = ''

        # fetch environment information if verbose==True
        if self.args.meta_verbose is 'full' or verbose:
            score_train = self.rater.score(self.y_train, self.y_pred_train_buffer)
            score_test  = self.rater.score(self.y_test,  self.y_pred_test_buffer) if self.flag_use_test_set else 'NULL'
            info = 'k={:<3d}|{}| train {:.3f} | valid {:.3f} | '.format(
                len(self.estimators_)-1, self.args.metric, score_train, score_valid)
            info += 'test {:.3f}'.format(score_test) if self.flag_use_test_set else 'test NULL'
        
        return next_state, reward, done, info
        
    def update_all_pred_buffer(self):
        """Update all buffered predict probabilities."""
        n_clf = len(self.estimators_)
        self.y_pred_train_buffer = self._update_pred_buffer(n_clf, self.X_train, self.y_pred_train_buffer)
        self.y_pred_valid_buffer = self._update_pred_buffer(n_clf, self.X_valid, self.y_pred_valid_buffer)
        if self.flag_use_test_set:
            self.y_pred_test_buffer  = self._update_pred_buffer(n_clf, self.X_test,  self.y_pred_test_buffer)
        return

    def _update_pred_buffer(self, n_clf, X, y_pred_buffer):
        """Update buffered predict probabilities.

        Parameters
        ----------
        n_clf : int
            Current ensemble size.

        X : array-like of shape = [n_samples, n_features]
            The input data instances.
            
        y_pred_buffer : array-like of shape [n_samples]
            The buffered predict probabilities of X.

        Returns
        ----------
        y_pred_updated : array-like of shape [n_samples]
        """
        y_pred_last_clf = self.estimators_[-1].predict_proba(X)[:, 1]
        y_pred_buffer_updated = (y_pred_buffer * (n_clf-1) + y_pred_last_clf) / n_clf
        return y_pred_buffer_updated
    
    def _warm_up(self):
        """Train the first base classifier with random under-sampling."""
        X_maj = self.X_train[self.mask_maj_train]
        X_min = self.X_train[self.mask_min_train]
        X_maj_rus = X_maj.sample(n=self.n_samples, random_state=self.args.random_state)
        # X_maj_rus = X_maj
        X_train_rus = pd.concat([X_maj_rus, X_min]).values
        y_train_rus = np.concatenate([np.zeros(X_maj_rus.shape[0]), np.ones(X_min.shape[0])])
        self.fit_step(X_train_rus, y_train_rus)
        self.update_all_pred_buffer()
        return