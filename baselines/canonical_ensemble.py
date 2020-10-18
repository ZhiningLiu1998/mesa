# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 14:32:27 2020
@author: ZhiningLiu1998
mailto: zhining.liu@outlook.com

NOTE: The implementation of SMOTEBoost/RUSBoost/RAMOBoost was obtained from
imbalanced-algorithms: https://github.com/dialnd/imbalanced-algorithms
"""

import numpy as np
import sklearn
from sklearn.base import is_regressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble.forest import BaseForest
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from sklearn.tree.tree import BaseDecisionTree
from sklearn.utils import check_random_state
from sklearn.utils import check_X_y
from sklearn.utils import check_array
from sklearn.preprocessing import binarize
from utils import *
from collections import Counter
import warnings
warnings.filterwarnings("ignore")


class SMOTE(object):
    """Implementation of Synthetic Minority Over-Sampling Technique (SMOTE).
    SMOTE performs oversampling of the minority class by picking target 
    minority class samples and their nearest minority class neighbors and 
    generating new samples that linearly combine features of each target 
    sample with features of its selected minority class neighbors [1].
    Parameters
    ----------
    k_neighbors : int, optional (default=5)
        Number of nearest neighbors.
    random_state : int or None, optional (default=None)
        If int, random_state is the seed used by the random number generator.
        If None, the random number generator is the RandomState instance used
        by np.random.
    References
    ----------
    .. [1] N. V. Chawla, K. W. Bowyer, L. O. Hall, and P. Kegelmeyer. "SMOTE:
           Synthetic Minority Over-Sampling Technique." Journal of Artificial
           Intelligence Research (JAIR), 2002.
    """

    def __init__(self, k_neighbors=5, random_state=None):
        self.k = k_neighbors
        self.random_state = random_state

    def sample(self, n_samples):
        """Generate samples.
        Parameters
        ----------
        n_samples : int
            Number of new synthetic samples.
        Returns
        -------
        S : array, shape = [n_samples, n_features]
            Returns synthetic samples.
        """
        np.random.seed(seed=self.random_state)

        S = np.zeros(shape=(n_samples, self.n_features))
        # Calculate synthetic samples.
        for i in range(n_samples):
            j = np.random.randint(0, self.X.shape[0])

            # Find the NN for each sample.
            # Exclude the sample itself.
            nn = self.neigh.kneighbors(self.X[j].reshape(1, -1),
                                       return_distance=False)[:, 1:]
            nn_index = np.random.choice(nn[0])

            dif = self.X[nn_index] - self.X[j]
            gap = np.random.random()

            S[i, :] = self.X[j, :] + gap * dif[:]

        return S

    def fit(self, X):
        """Train model based on input data.
        Parameters
        ----------
        X : array-like, shape = [n_minority_samples, n_features]
            Holds the minority samples.
        """
        self.X = X
        self.n_minority_samples, self.n_features = self.X.shape

        # Learn nearest neighbors.
        self.neigh = NearestNeighbors(n_neighbors=self.k + 1)
        self.neigh.fit(self.X)

        return self

class SMOTEBoost(AdaBoostClassifier):
    """Implementation of SMOTEBoost.
    SMOTEBoost introduces data sampling into the AdaBoost algorithm by
    oversampling the minority class using SMOTE on each boosting iteration [1].
    This implementation inherits methods from the scikit-learn 
    AdaBoostClassifier class, only modifying the `fit` method.
    Parameters
    ----------
    n_samples : int, optional (default=100)
        Number of new synthetic samples per boosting step.
    k_neighbors : int, optional (default=5)
        Number of nearest neighbors.
    base_estimator : object, optional (default=DecisionTreeClassifier)
        The base estimator from which the boosted ensemble is built.
        Support for sample weighting is required, as well as proper `classes_`
        and `n_classes_` attributes.
    n_estimators : int, optional (default=50)
        The maximum number of estimators at which boosting is terminated.
        In case of perfect fit, the learning procedure is stopped early.
    learning_rate : float, optional (default=1.)
        Learning rate shrinks the contribution of each classifier by
        ``learning_rate``. There is a trade-off between ``learning_rate`` and
        ``n_estimators``.
    algorithm : {'SAMME', 'SAMME.R'}, optional (default='SAMME.R')
        If 'SAMME.R' then use the SAMME.R real boosting algorithm.
        ``base_estimator`` must support calculation of class probabilities.
        If 'SAMME' then use the SAMME discrete boosting algorithm.
        The SAMME.R algorithm typically converges faster than SAMME,
        achieving a lower test error with fewer boosting iterations.
    random_state : int or None, optional (default=None)
        If int, random_state is the seed used by the random number generator.
        If None, the random number generator is the RandomState instance used
        by np.random.
    References
    ----------
    .. [1] N. V. Chawla, A. Lazarevic, L. O. Hall, and K. W. Bowyer.
           "SMOTEBoost: Improving Prediction of the Minority Class in
           Boosting." European Conference on Principles of Data Mining and
           Knowledge Discovery (PKDD), 2003.
    """

    def __init__(self,
                 n_samples=100,
                 k_neighbors=5,
                 base_estimator=None,
                 n_estimators=50,
                 learning_rate=1.,
                 algorithm='SAMME.R',
                 random_state=None):

        self.n_samples = n_samples
        self.algorithm = algorithm
        self.smote = SMOTE(k_neighbors=k_neighbors,
                           random_state=random_state)

        super(SMOTEBoost, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state)

    def fit(self, X, y, sample_weight=None, minority_target=None):
        """Build a boosted classifier/regressor from the training set (X, y),
        performing SMOTE during each boosting step.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR. The dtype is
            forced to DTYPE from tree._tree if the base classifier of this
            ensemble weighted boosting classifier is a tree or forest.
        y : array-like of shape = [n_samples]
            The target values (class labels in classification, real numbers in
            regression).
        sample_weight : array-like of shape = [n_samples], optional
            Sample weights. If None, the sample weights are initialized to
            1 / n_samples.
        minority_target : int
            Minority class label.
        Returns
        -------
        self : object
            Returns self.
        Notes
        -----
        Based on the scikit-learn v0.18 AdaBoostClassifier and
        BaseWeightBoosting `fit` methods.
        """
        # Check that algorithm is supported.
        if self.algorithm not in ('SAMME', 'SAMME.R'):
            raise ValueError("algorithm %s is not supported" % self.algorithm)

        # Check parameters.
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")

        if (self.base_estimator is None or
                isinstance(self.base_estimator, (BaseDecisionTree,
                                                 BaseForest))):
            DTYPE = np.float64  # from fast_dict.pxd
            dtype = DTYPE
            accept_sparse = 'csc'
        else:
            dtype = None
            accept_sparse = ['csr', 'csc']

        X, y = check_X_y(X, y, accept_sparse=accept_sparse, dtype=dtype,
                         y_numeric=is_regressor(self))

        if sample_weight is None:
            # Initialize weights to 1 / n_samples.
            sample_weight = np.empty(X.shape[0], dtype=np.float64)
            sample_weight[:] = 1. / X.shape[0]
        else:
            sample_weight = check_array(sample_weight, ensure_2d=False)
            # Normalize existing weights.
            sample_weight = sample_weight / sample_weight.sum(dtype=np.float64)

            # Check that the sample weights sum is positive.
            if sample_weight.sum() <= 0:
                raise ValueError(
                    "Attempting to fit with a non-positive "
                    "weighted number of samples.")

        if minority_target is None:
            # Determine the minority class label.
            stats_c_ = Counter(y)
            maj_c_ = max(stats_c_, key=stats_c_.get)
            min_c_ = min(stats_c_, key=stats_c_.get)
            self.minority_target = min_c_
        else:
            self.minority_target = minority_target

        # Check parameters.
        self._validate_estimator()

        # Clear any previous fit results.
        self.estimators_ = []
        self.estimator_weights_ = np.zeros(self.n_estimators, dtype=np.float64)
        self.estimator_errors_ = np.ones(self.n_estimators, dtype=np.float64)

        random_state = check_random_state(self.random_state)

        self.total_training_instances = 0
        self.total_training_instances_list = []
        for iboost in range(self.n_estimators):
            # SMOTE step.
            X_min = X[np.where(y == self.minority_target)]
            self.smote.fit(X_min)
            X_syn = self.smote.sample(self.n_samples)
            y_syn = np.full(X_syn.shape[0], fill_value=self.minority_target,
                            dtype=np.int64)

            # Normalize synthetic sample weights based on current training set.
            sample_weight_syn = np.empty(X_syn.shape[0], dtype=np.float64)
            sample_weight_syn[:] = 1. / X.shape[0]

            # print ('Boosting Iter: {} n_train: {} n_smote: {}'.format(
            #     iboost, len(X_min), len(y_syn)))

            # Combine the original and synthetic samples.
            X = np.vstack((X, X_syn))
            y = np.append(y, y_syn)

            self.total_training_instances = self.total_training_instances + len(y)
            self.total_training_instances_list.append(self.total_training_instances)
            print(f'SMOTEBoost total training size: {self.total_training_instances}')

            # Combine the weights.
            sample_weight = \
                np.append(sample_weight, sample_weight_syn).reshape(-1, 1)
            sample_weight = \
                np.squeeze(normalize(sample_weight, axis=0, norm='l1'))

            # X, y, sample_weight = shuffle(X, y, sample_weight,
            #                              random_state=random_state)

            # Boosting step.
            sample_weight, estimator_weight, estimator_error = self._boost(
                iboost,
                X, y,
                sample_weight,
                random_state)

            # Early termination.
            if sample_weight is None:
                print('sample_weight: {}'.format(sample_weight))
                break

            self.estimator_weights_[iboost] = estimator_weight
            self.estimator_errors_[iboost] = estimator_error

            # Stop if error is zero.
            # if estimator_error == 0:
            #     print('error: {}'.format(estimator_error))
            #     break

            sample_weight_sum = np.sum(sample_weight)

            # Stop if the sum of sample weights has become non-positive.
            if sample_weight_sum <= 0:
                print('sample_weight_sum: {}'.format(sample_weight_sum))
                break

            if iboost < self.n_estimators - 1:
                # Normalize.
                sample_weight /= sample_weight_sum

        return self

class RankedMinorityOversampler(object):
    """Implementation of Ranked Minority Oversampling (RAMO).
    Oversample the minority class by picking samples according to a specified
    sampling distribution.
    Parameters
    ----------
    k_neighbors_1 : int, optional (default=5)
        Number of nearest neighbors used to adjust the sampling probability of
        the minority examples.
    k_neighbors_2 : int, optional (default=5)
        Number of nearest neighbors used to generate the synthetic data
        instances.
    alpha : float, optional (default=0.3)
        Scaling coefficient.
    random_state : int or None, optional (default=None)
        If int, random_state is the seed used by the random number generator.
        If None, the random number generator is the RandomState instance used
        by np.random.
    """

    def __init__(self, k_neighbors_1=5, k_neighbors_2=5, alpha=0.3,
                 random_state=None):
        self.k_neighbors_1 = k_neighbors_1
        self.k_neighbors_2 = k_neighbors_2
        self.alpha = alpha
        self.random_state = random_state

    def sample(self, n_samples):
        """Generate samples.
        Parameters
        ----------
        n_samples : int
            Number of new synthetic samples.
        Returns
        -------
        S : array, shape = [n_samples, n_features]
            Returns synthetic samples.
        """
        np.random.seed(seed=self.random_state)

        S = np.zeros(shape=(n_samples, self.n_features))
        # Calculate synthetic samples.
        for i in range(n_samples):
            # Choose a sample according to the sampling distribution, r.
            j = np.random.choice(self.n_minority_samples, p=self.r)

            # Find the NN for each sample.
            # Exclude the sample itself.
            nn = self.neigh_2.kneighbors(self.X_min[j].reshape(1, -1),
                                         return_distance=False)[:, 1:]
            nn_index = np.random.choice(nn[0])

            dif = self.X_min[nn_index] - self.X_min[j]
            gap = np.random.random()

            S[i, :] = self.X_min[j, :] + gap * dif[:]

        return S

    def fit(self, X, y, sample_weight=None, minority_target=None):
        """Train model based on input data.
        Parameters
        ----------
        X : array-like, shape = [n_total_samples, n_features]
            Holds the majority and minority samples.
        y : array-like, shape = [n_total_samples]
            Holds the class targets for samples.
        sample_weight : array-like of shape = [n_samples], optional
            Sample weights multiplier. If None, the multiplier is 1.
        minority_target : int, optional (default=None)
            Minority class label.
        """
        if minority_target is None:
            # Determine the minority class label.
            stats_c_ = Counter(y)
            maj_c_ = max(stats_c_, key=stats_c_.get)
            min_c_ = min(stats_c_, key=stats_c_.get)
            self.minority_target = min_c_
        else:
            self.minority_target = minority_target

        self.X_min = X[y == self.minority_target]
        self.n_minority_samples, self.n_features = self.X_min.shape

        neigh_1 = NearestNeighbors(n_neighbors=self.k_neighbors_1 + 1)
        neigh_1.fit(X)
        nn = neigh_1.kneighbors(self.X_min, return_distance=False)[:, 1:]

        if sample_weight is None:
            sample_weight_min = np.ones(shape=(len(self.minority_target)))
        else:
            assert(len(y) == len(sample_weight))
            sample_weight_min = sample_weight[y == self.minority_target]

        self.r = np.zeros(shape=(self.n_minority_samples))
        for i in range(self.n_minority_samples):
            majority_neighbors = 0
            for n in nn[i]:
                if y[n] != self.minority_target:
                    majority_neighbors += 1

            self.r[i] = 1. / (1 + np.exp(-self.alpha * majority_neighbors))

        self.r = (self.r * sample_weight_min).reshape(1, -1)
        self.r = np.squeeze(normalize(self.r, axis=1, norm='l1'))

        # Learn nearest neighbors.
        self.neigh_2 = NearestNeighbors(n_neighbors=self.k_neighbors_2 + 1)
        self.neigh_2.fit(self.X_min)

        return self


class RAMOBoost(AdaBoostClassifier):
    """Implementation of RAMOBoost.
    RAMOBoost introduces data sampling into the AdaBoost algorithm by
    oversampling the minority class according to a specified sampling
    distribution on each boosting iteration [1].
    This implementation inherits methods from the scikit-learn 
    AdaBoostClassifier class, only modifying the `fit` method.
    Parameters
    ----------
    n_samples : int, optional (default=100)
        Number of new synthetic samples per boosting step.
    k_neighbors_1 : int, optional (default=5)
        Number of nearest neighbors used to adjust the sampling probability of
        the minority examples.
    k_neighbors_2 : int, optional (default=5)
        Number of nearest neighbors used to generate the synthetic data
        instances.
    alpha : float, optional (default=0.3)
        Scaling coefficient.
    base_estimator : object, optional (default=DecisionTreeClassifier)
        The base estimator from which the boosted ensemble is built.
        Support for sample weighting is required, as well as proper `classes_`
        and `n_classes_` attributes.
    n_estimators : int, optional (default=50)
        The maximum number of estimators at which boosting is terminated.
        In case of perfect fit, the learning procedure is stopped early.
    learning_rate : float, optional (default=1.)
        Learning rate shrinks the contribution of each classifier by
        ``learning_rate``. There is a trade-off between ``learning_rate`` and
        ``n_estimators``.
    algorithm : {'SAMME', 'SAMME.R'}, optional (default='SAMME.R')
        If 'SAMME.R' then use the SAMME.R real boosting algorithm.
        ``base_estimator`` must support calculation of class probabilities.
        If 'SAMME' then use the SAMME discrete boosting algorithm.
        The SAMME.R algorithm typically converges faster than SAMME,
        achieving a lower test error with fewer boosting iterations.
    random_state : int or None, optional (default=None)
        If int, random_state is the seed used by the random number generator.
        If None, the random number generator is the RandomState instance used
        by np.random.
    References
    ----------
    .. [1] S. Chen, H. He, and E. A. Garcia. "RAMOBoost: Ranked Minority
           Oversampling in Boosting". IEEE Transactions on Neural Networks,
           2010.
    """

    def __init__(self,
                 n_samples=100,
                 k_neighbors_1=5,
                 k_neighbors_2=5,
                 alpha=0.3,
                 base_estimator=None,
                 n_estimators=50,
                 learning_rate=1.,
                 algorithm='SAMME.R',
                 random_state=None):

        self.n_samples = n_samples
        self.algorithm = algorithm
        self.ramo = RankedMinorityOversampler(k_neighbors_1, k_neighbors_2,
                                              alpha, random_state=random_state)

        super(RAMOBoost, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state)

    def fit(self, X, y, sample_weight=None, minority_target=None):
        """Build a boosted classifier/regressor from the training set (X, y),
        performing random undersampling during each boosting step.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR. The dtype is
            forced to DTYPE from tree._tree if the base classifier of this
            ensemble weighted boosting classifier is a tree or forest.
        y : array-like of shape = [n_samples]
            The target values (class labels in classification, real numbers in
            regression).
        sample_weight : array-like of shape = [n_samples], optional
            Sample weights. If None, the sample weights are initialized to
            1 / n_samples.
        minority_target : int
            Minority class label.
        Returns
        -------
        self : object
            Returns self.
        Notes
        -----
        Based on the scikit-learn v0.18 AdaBoostClassifier and
        BaseWeightBoosting `fit` methods.
        """
        # Check that algorithm is supported.
        if self.algorithm not in ('SAMME', 'SAMME.R'):
            raise ValueError("algorithm %s is not supported" % self.algorithm)

        # Check parameters.
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")

        if (self.base_estimator is None or
                isinstance(self.base_estimator, (BaseDecisionTree,
                                                 BaseForest))):
            DTYPE = np.float64  # from fast_dict.pxd
            dtype = DTYPE
            accept_sparse = 'csc'
        else:
            dtype = None
            accept_sparse = ['csr', 'csc']

        X, y = check_X_y(X, y, accept_sparse=accept_sparse, dtype=dtype,
                         y_numeric=is_regressor(self))

        if sample_weight is None:
            # Initialize weights to 1 / n_samples.
            sample_weight = np.empty(X.shape[0], dtype=np.float64)
            sample_weight[:] = 1. / X.shape[0]
        else:
            sample_weight = check_array(sample_weight, ensure_2d=False)
            # Normalize existing weights.
            sample_weight = sample_weight / sample_weight.sum(dtype=np.float64)

            # Check that the sample weights sum is positive.
            if sample_weight.sum() <= 0:
                raise ValueError(
                    "Attempting to fit with a non-positive "
                    "weighted number of samples.")

        if minority_target is None:
            # Determine the minority class label.
            stats_c_ = Counter(y)
            maj_c_ = max(stats_c_, key=stats_c_.get)
            min_c_ = min(stats_c_, key=stats_c_.get)
            self.minority_target = min_c_
        else:
            self.minority_target = minority_target

        # Check parameters.
        self._validate_estimator()

        # Clear any previous fit results.
        self.estimators_ = []
        self.estimator_weights_ = np.zeros(self.n_estimators, dtype=np.float64)
        self.estimator_errors_ = np.ones(self.n_estimators, dtype=np.float64)

        random_state = check_random_state(self.random_state)

        self.total_training_instances = 0
        self.total_training_instances_list = []
        for iboost in range(self.n_estimators):
            # RAMO step.
            self.ramo.fit(X, y, sample_weight=sample_weight)
            X_syn = self.ramo.sample(self.n_samples)
            y_syn = np.full(X_syn.shape[0], fill_value=self.minority_target,
                            dtype=np.int64)

            # Combine the minority and majority class samples.
            X = np.vstack((X, X_syn))
            y = np.append(y, y_syn)

            self.total_training_instances = self.total_training_instances + len(y)
            self.total_training_instances_list.append(self.total_training_instances)
            print (f'RAMOBoost total training size: {self.total_training_instances}')

            # Normalize synthetic sample weights based on current training set.
            sample_weight_syn = np.empty(X_syn.shape[0], dtype=np.float64)
            sample_weight_syn[:] = 1. / X.shape[0]

            # Combine the weights.
            sample_weight = \
                np.append(sample_weight, sample_weight_syn).reshape(-1, 1)
            sample_weight = \
                np.squeeze(normalize(sample_weight, axis=0, norm='l1'))

            # X, y, sample_weight = shuffle(X, y, sample_weight,
            #                              random_state=random_state)

            # Boosting step.
            sample_weight, estimator_weight, estimator_error = self._boost(
                iboost,
                X, y,
                sample_weight,
                random_state)

            # Early termination.
            # if sample_weight is None:
            #     break

            self.estimator_weights_[iboost] = estimator_weight
            self.estimator_errors_[iboost] = estimator_error

            # Stop if error is zero.
            # if estimator_error == 0:
            #     break

            sample_weight_sum = np.sum(sample_weight)

            # Stop if the sum of sample weights has become non-positive.
            # if sample_weight_sum <= 0:
            #     break

            if iboost < self.n_estimators - 1:
                # Normalize.
                sample_weight /= sample_weight_sum

        return self

class RandomUnderSampler(object):
    """Implementation of random undersampling (RUS).
    Undersample the majority class(es) by randomly picking samples with or
    without replacement.
    Parameters
    ----------
    with_replacement : bool, optional (default=True)
        Undersample with replacement.
    return_indices : bool, optional (default=False)
        Whether or not to return the indices of the samples randomly selected
        from the majority class.
    random_state : int or None, optional (default=None)
        If int, random_state is the seed used by the random number generator.
        If None, the random number generator is the RandomState instance used
        by np.random.
    """

    def __init__(self, with_replacement=True, return_indices=False,
                 random_state=None):
        self.return_indices = return_indices
        self.with_replacement = with_replacement
        self.random_state = random_state

    def sample(self, n_samples):
        """Perform undersampling.
        Parameters
        ----------
        n_samples : int
            Number of samples to remove.
        Returns
        -------
        S : array, shape = [n_majority_samples - n_samples, n_features]
            Returns synthetic samples.
        """
        np.random.seed(seed=self.random_state)

        if self.n_majority_samples <= n_samples:
            n_samples = self.n_majority_samples

        idx = np.random.choice(self.n_majority_samples,
                            #    size=self.n_majority_samples - n_samples,
                               size=self.n_minority_samples,
                               replace=self.with_replacement)

        if self.return_indices:
            return (self.X_maj[idx], idx)
        else:
            return self.X_maj[idx]

    def fit(self, X_maj, X_min):
        """Train model based on input data.
        Parameters
        ----------
        X : array-like, shape = [n_majority_samples, n_features]
            Holds the majority samples.
        """
        self.X_maj = X_maj
        self.X_min = X_min
        self.n_majority_samples, self.n_features = self.X_maj.shape
        self.n_minority_samples = self.X_min.shape[0]

        return self

import pandas as pd

class RUSBoost(AdaBoostClassifier):
    """Implementation of RUSBoost.
    RUSBoost introduces data sampling into the AdaBoost algorithm by
    undersampling the majority class using random undersampling (with or
    without replacement) on each boosting iteration [1].
    This implementation inherits methods from the scikit-learn 
    AdaBoostClassifier class, only modifying the `fit` method.
    Parameters
    ----------
    n_samples : int, optional (default=100)
        Number of new synthetic samples per boosting step.
    min_ratio : float (default=1.0)
        Minimum ratio of majority to minority class samples to generate.
    with_replacement : bool, optional (default=True)
        Undersample with replacement.
    base_estimator : object, optional (default=DecisionTreeClassifier)
        The base estimator from which the boosted ensemble is built.
        Support for sample weighting is required, as well as proper `classes_`
        and `n_classes_` attributes.
    n_estimators : int, optional (default=50)
        The maximum number of estimators at which boosting is terminated.
        In case of perfect fit, the learning procedure is stopped early.
    learning_rate : float, optional (default=1.)
        Learning rate shrinks the contribution of each classifier by
        ``learning_rate``. There is a trade-off between ``learning_rate`` and
        ``n_estimators``.
    algorithm : {'SAMME', 'SAMME.R'}, optional (default='SAMME.R')
        If 'SAMME.R' then use the SAMME.R real boosting algorithm.
        ``base_estimator`` must support calculation of class probabilities.
        If 'SAMME' then use the SAMME discrete boosting algorithm.
        The SAMME.R algorithm typically converges faster than SAMME,
        achieving a lower test error with fewer boosting iterations.
    random_state : int or None, optional (default=None)
        If int, random_state is the seed used by the random number generator.
        If None, the random number generator is the RandomState instance used
        by np.random.
    References
    ----------
    .. [1] C. Seiffert, T. M. Khoshgoftaar, J. V. Hulse, and A. Napolitano.
           "RUSBoost: Improving Classification Performance when Training Data
           is Skewed". International Conference on Pattern Recognition
           (ICPR), 2008.
    """

    def __init__(self,
                 n_samples=100,
                 min_ratio=1.0,
                 with_replacement=True,
                 base_estimator=None,
                 n_estimators=10,
                 learning_rate=1.,
                 algorithm='SAMME.R',
                 random_state=None):

        self.n_samples = n_samples
        self.min_ratio = min_ratio
        self.algorithm = algorithm
        self.rus = RandomUnderSampler(with_replacement=with_replacement,
                                      return_indices=True,
                                      random_state=random_state)

        super(RUSBoost, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state)

    def fit(self, X, y, sample_weight=None, minority_target=None, verbose=False):
        """Build a boosted classifier/regressor from the training set (X, y),
        performing random undersampling during each boosting step.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR. The dtype is
            forced to DTYPE from tree._tree if the base classifier of this
            ensemble weighted boosting classifier is a tree or forest.
        y : array-like of shape = [n_samples]
            The target values (class labels in classification, real numbers in
            regression).
        sample_weight : array-like of shape = [n_samples], optional
            Sample weights. If None, the sample weights are initialized to
            1 / n_samples.
        minority_target : int
            Minority class label.
        Returns
        -------
        self : object
            Returns self.
        Notes
        -----
        Based on the scikit-learn v0.18 AdaBoostClassifier and
        BaseWeightBoosting `fit` methods.
        """
        # Check that algorithm is supported.
        if self.algorithm not in ('SAMME', 'SAMME.R'):
            raise ValueError("algorithm %s is not supported" % self.algorithm)

        # Check parameters.
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")

        if (self.base_estimator is None or
                isinstance(self.base_estimator, (BaseDecisionTree,
                                                 BaseForest))):
            DTYPE = np.float64  # from fast_dict.pxd
            dtype = DTYPE
            accept_sparse = 'csc'
        else:
            dtype = None
            accept_sparse = ['csr', 'csc']

        X, y = check_X_y(X, y, accept_sparse=accept_sparse, dtype=dtype,
                         y_numeric=is_regressor(self))

        if sample_weight is None:
            # Initialize weights to 1 / n_samples.
            sample_weight = np.empty(X.shape[0], dtype=np.float64)
            sample_weight[:] = 1. / X.shape[0]
        else:
            sample_weight = check_array(sample_weight, ensure_2d=False)
            # Normalize existing weights.
            sample_weight = sample_weight / sample_weight.sum(dtype=np.float64)

            # Check that the sample weights sum is positive.
            if sample_weight.sum() <= 0:
                raise ValueError(
                    "Attempting to fit with a non-positive "
                    "weighted number of samples.")

        if minority_target is None:
            # Determine the minority class label.
            stats_c_ = Counter(y)
            maj_c_ = max(stats_c_, key=stats_c_.get)
            min_c_ = min(stats_c_, key=stats_c_.get)
            self.minority_target = min_c_
        else:
            self.minority_target = minority_target

        # Check parameters.
        self._validate_estimator()

        # Clear any previous fit results.
        self.estimators_ = []
        self.estimator_weights_ = np.zeros(self.n_estimators, dtype=np.float64)
        self.estimator_errors_ = np.ones(self.n_estimators, dtype=np.float64)

        random_state = check_random_state(self.random_state)

        for iboost in range(self.n_estimators):
            # Random undersampling step.
            X_maj = X[np.where(y != self.minority_target)]
            X_min = X[np.where(y == self.minority_target)]
            self.rus.fit(X_maj, X_min)
            # self.rus.fit(X_maj)

            n_maj = X_maj.shape[0]
            n_min = X_min.shape[0]
            if n_maj - self.n_samples < int(n_min * self.min_ratio):
                self.n_samples = n_maj - int(n_min * self.min_ratio)
            X_rus, X_idx = self.rus.sample(self.n_samples)

            if verbose:
                print ('{:<12s} | Iter: {} X_maj: {} X_rus: {} X_min: {}'.format(
                    'RUSBoost', iboost, len(X_maj), len(X_rus), len(X_min)))

            y_rus = y[np.where(y != self.minority_target)][X_idx]
            y_min = y[np.where(y == self.minority_target)]

            sample_weight_rus = \
                sample_weight[np.where(y != self.minority_target)][X_idx]
            sample_weight_min = \
                sample_weight[np.where(y == self.minority_target)]

            # Combine the minority and majority class samples.
            X_train = np.vstack((X_rus, X_min))
            y_train = np.append(y_rus, y_min)

            # Combine the weights.
            sample_weight_train = \
                np.append(sample_weight_rus, sample_weight_min).reshape(-1, 1)
            sample_weight_train = \
                np.squeeze(normalize(sample_weight_train, axis=0, norm='l1'))

            # Boosting step.
            _, estimator_weight_train, estimator_error = self._boost(
                iboost,
                X_train, y_train,
                sample_weight_train,
                random_state)
            
            y_predict_proba = self.estimators_[-1].predict_proba(X)
            y_predict = self.classes_.take(np.argmax(y_predict_proba, axis=1),
                                        axis=0)
            # Instances incorrectly classified
            incorrect = y_predict != y
            # Error fraction
            estimator_error = np.mean(
                np.average(incorrect, weights=sample_weight, axis=0))
            n_classes = self.n_classes_
            classes = self.classes_
            y_codes = np.array([-1. / (n_classes - 1), 1.])
            y_coding = y_codes.take(classes == y[:, np.newaxis])
            estimator_weight = (-1. * self.learning_rate
                    * ((n_classes - 1.) / n_classes)
                    * (y_coding * (y_predict_proba)).sum(axis=1))
            
            if not iboost == self.n_estimators - 1:
                # Only boost positive weights
                sample_weight *= np.exp(estimator_weight * ((sample_weight > 0) | (estimator_weight < 0)))

            # Early termination.
            if sample_weight is None:
                break

            self.estimator_weights_[iboost] = estimator_weight_train
            self.estimator_errors_[iboost] = estimator_error

            # Stop if error is zero.
            # if estimator_error == 0:
                # print('error: {}'.format(estimator_error))
                # break

            sample_weight_sum = np.sum(sample_weight)

            # Stop if the sum of sample weights has become non-positive.
            if sample_weight_sum <= 0:
                break

            if iboost < self.n_estimators - 1:
                # Normalize.
                sample_weight /= sample_weight_sum

        return self

import pandas as pd
from imblearn.over_sampling import SMOTE as SMOTE_IMB
from sklearn.tree import DecisionTreeClassifier as DT

class SMOTEBagging():
    def __init__(self,
                 n_samples=100,
                 min_ratio=1.0,
                 with_replacement=True,
                 base_estimator=None,
                 n_estimators=10,
                 learning_rate=1.,
                 algorithm='SAMME.R',
                 random_state=None):

        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.estimators_ = []
    
    def fit(self, X, y, verbose=False):

        self.total_training_instances = 0
        self.total_training_instances_list = []
        self.estimators_ = []
        df = pd.DataFrame(X); df['label'] = y
        df_maj = df[df['label']==0]; n_maj = len(df_maj)
        df_min = df[df['label']==1]; n_min = len(df_min)
        cols = df.columns.tolist(); cols.remove('label')

        for ibagging in range(self.n_estimators):
            b = min(0.1*((ibagging%10)+1), 1)
            train_maj = df_maj.sample(frac=1, replace=True)
            train_min = df_min.sample(frac=(n_maj/n_min)*b, replace=True)
            n_min_train = train_min.shape[0]
            N = int((n_maj/n_min_train)*(1-b)*100)
            ratio = min((n_min_train + N) / n_maj, 1)
            df_k = train_maj.append(train_min)

            if N > 0:
                X_train, y_train = SMOTE_IMB(
                    k_neighbors=min(5, len(train_min)-1), 
                    ratio=ratio,
                    random_state=self.random_state,
                ).fit_resample(
                    df_k[cols], df_k['label']
                )
            else:
                X_train, y_train = df_k[cols], df_k['label']

            self.total_training_instances = self.total_training_instances + len(y_train)
            self.total_training_instances_list.append(self.total_training_instances)
            if verbose:
                print ('{:<12s} | Iter: {} |b: {:.1f}|n_train: {}|n_smote: {}|n_total_train: {}'.format(
                    'SMOTEBagging', ibagging, b, len(y_train), len(y_train)-len(df_k), self.total_training_instances))
            model = clone(self.base_estimator).fit(X_train, y_train)
            self.estimators_.append(model)

        return self
    
    def predict_proba(self, X):

        y_pred = np.array([model.predict_proba(X)[:, 1] for model in self.estimators_]).mean(axis=0)
        if y_pred.ndim == 1:
            y_pred = y_pred[:, np.newaxis]
        if y_pred.shape[1] == 1:
            y_pred = np.append(1-y_pred, y_pred, axis=1)
        return y_pred
    
    def predict(self, X):

        y_pred_binarazed = binarize(self.predict_proba(X)[:,1].reshape(1,-1), threshold=0.5)[0]
        return y_pred_binarazed


import pandas as pd
from sklearn.tree import DecisionTreeClassifier as DT

class UnderBagging():
    def __init__(self,
                 n_samples=100,
                 min_ratio=1.0,
                 with_replacement=True,
                 base_estimator=None,
                 n_estimators=10,
                 learning_rate=1.,
                 algorithm='SAMME.R',
                 random_state=None):

        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.estimators_ = []
    
    def fit(self, X, y, verbose=False):

        self.estimators_ = []
        df = pd.DataFrame(X); df['label'] = y
        df_maj = df[df['label']==0]; n_maj = len(df_maj)
        df_min = df[df['label']==1]; n_min = len(df_min)
        cols = df.columns.tolist(); cols.remove('label')

        for ibagging in range(self.n_estimators):
            train_maj = df_maj.sample(n=int(n_min), random_state=self.random_state)
            train_min = df_min
            if verbose:
                print ('{:<12s} | Iter: {} X_maj: {} X_rus: {} X_min: {}'.format(
                    'UnderBagging', ibagging, len(df_maj), len(train_maj), len(train_min)))
            df_k = train_maj.append(train_min)
            X_train, y_train = df_k[cols], df_k['label']
            model = clone(self.base_estimator).fit(X_train, y_train)
            self.estimators_.append(model)

        return self

    def predict_proba(self, X):

        y_pred = np.array([model.predict_proba(X)[:, 1] for model in self.estimators_]).mean(axis=0)
        if y_pred.ndim == 1:
            y_pred = y_pred[:, np.newaxis]
        if y_pred.shape[1] == 1:
            y_pred = np.append(1-y_pred, y_pred, axis=1)
        return y_pred
    
    def predict(self, X):

        y_pred_binarazed = binarize(self.predict_proba(X)[:,1].reshape(1,-1), threshold=0.5)[0]
        return y_pred_binarazed


from sklearn.base import clone
class BalanceCascade():
    """
    The implementation of BalanceCascade.
    Hyper-parameters:
        base_estimator : scikit-learn classifier object
            optional (default=DecisionTreeClassifier)
            The base estimator from which the ensemble is built.
        n_estimators:       Number of iterations / estimators
        k_bins:             Number of hardness bins
    """
    def __init__(self, base_estimator=DT(), n_estimators=10, random_state=None):

        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.estimators_ = []
        # Will be set in the fit function
        self.feature_cols = None

    def _fit_baselearner(self, df_train):

        model = clone(self.base_estimator)
        return model.fit(df_train[self.feature_cols], df_train['label'])

    def fit(self, X, y, verbose=False, visualize=False):

        self.estimators_ = []
        # Initialize majority & minority set
        df = pd.DataFrame(X); df['label'] = y
        df_maj = df[y==0]; n_maj = df_maj.shape[0]
        df_min = df[y==1]; n_min = df_min.shape[0]
        self.feature_cols = df.columns.tolist()
        self.feature_cols.remove('label')

        ir = n_min / n_maj
        keep_fp_rate = np.power(ir, 1/(self.n_estimators-1))

        # Algorithm start
        for ibagging in range(1, self.n_estimators):
            df_train = df_maj.sample(n=n_min).append(df_min)
            if visualize:
                df_train.plot.scatter(x=0, y=1, s=3, c='label', colormap='coolwarm', title='Iter {} training set'.format(ibagging))
            if verbose:
                print ('{:<12s} | Iter: {} X_maj: {} X_rus: {} X_min: {}'.format(
                    'Cascade', ibagging, len(df_maj), len(df_min), len(df_min)))
            self.estimators_.append(self._fit_baselearner(df_train))
            # drop "easy" majority samples
            df_maj['pred_proba'] = self.predict(df_maj[self.feature_cols])
            df_maj = df_maj.sort_values(by='pred_proba', ascending=False)[:int(keep_fp_rate*len(df_maj)+1)]
        
        return self

    def predict_proba(self, X):

        y_pred = np.array([model.predict_proba(X)[:, 1] for model in self.estimators_]).mean(axis=0)
        if y_pred.ndim == 1:
            y_pred = y_pred[:, np.newaxis]
        if y_pred.shape[1] == 1:
            y_pred = np.append(1-y_pred, y_pred, axis=1)
        return y_pred
    
    def predict(self, X):

        y_pred_binarazed = binarize(self.predict_proba(X)[:,1].reshape(1,-1), threshold=0.5)[0]
        return y_pred_binarazed

class SelfPacedEnsemble():
    """ Self-paced Ensemble (SPE)

    Parameters
    ----------
    base_estimator : object, optional (default=sklearn.Tree.DecisionTreeClassifier())
    |   The base estimator to fit on self-paced under-sampled subsets of the dataset. 
    |   NO need to support sample weighting. 
    |   Built-in `fit()`, `predict()`, `predict_proba()` methods are required.

    hardness_func :  function, optional 
    |   (default=`lambda y_true, y_pred: np.absolute(y_true-y_pred)`)
    |   User-specified classification hardness function
    |   |   Parameters:
    |   |   |   y_true: 1-d array-like, shape = [n_samples] 
    |   |   |   y_pred: 1-d array-like, shape = [n_samples] 
    |   |   Returns:
    |   |   |   hardness: 1-d array-like, shape = [n_samples]

    n_estimators :  integer, optional (default=10)
    |   The number of base estimators in the ensemble.

    k_bins :        integer, optional (default=10)
    |   The number of hardness bins that were used to approximate hardness distribution.

    random_state :  integer / RandomState instance / None, optional (default=None)
    |   If integer, random_state is the seed used by the random number generator; 
    |   If RandomState instance, random_state is the random number generator; 
    |   If None, the random number generator is the RandomState instance used by 
    |   `numpy.random`.

    Attributes
    ----------
    base_estimator_ : estimator
    |   The base estimator from which the ensemble is grown.

    estimators_ : list of estimator
    |   The collection of fitted base estimators.


    Example:
    ```
    import numpy as np
    from sklearn import datasets
    from sklearn.tree import DecisionTreeClassifier
    from src.self_paced_ensemble import SelfPacedEnsemble
    from src.utils import (
        make_binary_classification_target, imbalance_train_test_split)

    X, y = datasets.fetch_covtype(return_X_y=True)
    y = make_binary_classification_target(y, 7, True)
    X_train, X_test, y_train, y_test = imbalance_train_test_split(
            X, y, test_size=0.2, random_state=42)

    def absolute_error(y_true, y_pred):
        # Self-defined classification hardness function
        return np.absolute(y_true - y_pred)

    spe = SelfPacedEnsemble(
        base_estimator=DecisionTreeClassifier(),
        hardness_func=absolute_error,
        n_estimators=10,
        k_bins=10,
        random_state=42,
    ).fit(
        X=X_train,
        y=y_train,
    )
    print('auc_prc_score: {}'.format(spe.score(X_test, y_test)))
    ```

    """
    def __init__(self, 
            base_estimator=DecisionTreeClassifier(), 
            hardness_func=cross_entropy,
            n_estimators=10, 
            k_bins=10, 
            random_state=None):
        self.base_estimator = base_estimator
        self.estimators_ = []
        self._hardness_func = hardness_func
        self._n_estimators = n_estimators
        self._k_bins = k_bins
        self._random_state = random_state

    def _fit_base_estimator(self, X, y):
        """Private function used to train a single base estimator."""
        return sklearn.base.clone(self.base_estimator).fit(X, y)

    def _random_under_sampling(self, X_maj, y_maj, X_min, y_min):
        """Private function used to perform random under-sampling."""
        np.random.seed(self._random_state)
        idx = np.random.choice(len(X_maj), len(X_min), replace=False)
        X_train = np.concatenate([X_maj[idx], X_min])
        y_train = np.concatenate([y_maj[idx], y_min])
        return X_train, y_train

    def _self_paced_under_sampling(self, 
            X_maj, y_maj, X_min, y_min, i_estimator):
        """Private function used to perform self-paced under-sampling."""
        # Update hardness value estimation
        y_pred_maj = self.predict_proba(X_maj)[:, 1]
        hardness = self._hardness_func(y_maj, y_pred_maj)

        # If hardness values are not distinguishable, perform random smapling
        if hardness.max() == hardness.min():
            X_train, y_train = self._random_under_sampling(X_maj, y_maj, X_min, y_min)
        # Else allocate majority samples into k hardness bins
        else:
            step = (hardness.max()-hardness.min()) / self._k_bins
            bins = []; ave_contributions = []
            for i_bins in range(self._k_bins):
                idx = (
                    (hardness >= i_bins*step + hardness.min()) & 
                    (hardness < (i_bins+1)*step + hardness.min())
                )
                # Marginal samples with highest hardness value -> kth bin
                if i_bins == (self._k_bins-1):
                    idx = idx | (hardness==hardness.max())
                bins.append(X_maj[idx])
                ave_contributions.append(hardness[idx].mean())

            # Update self-paced factor alpha
            alpha = np.tan(np.pi*0.5*(i_estimator/(self._n_estimators-1)))
            # Caculate sampling weight
            weights = 1 / (ave_contributions + alpha)
            weights[np.isnan(weights)] = 0
            # Caculate sample number from each bin
            n_sample_bins = len(X_min) * weights / weights.sum()
            n_sample_bins = n_sample_bins.astype(int)+1
            
            # Perform self-paced under-sampling
            sampled_bins = []
            for i_bins in range(self._k_bins):
                if min(len(bins[i_bins]), n_sample_bins[i_bins]) > 0:
                    np.random.seed(self._random_state)
                    idx = np.random.choice(
                        len(bins[i_bins]), 
                        min(len(bins[i_bins]), n_sample_bins[i_bins]), 
                        replace=False)
                    sampled_bins.append(bins[i_bins][idx])
            X_train_maj = np.concatenate(sampled_bins, axis=0)
            y_train_maj = np.full(X_train_maj.shape[0], y_maj[0])
            X_train = np.concatenate([X_train_maj, X_min])
            y_train = np.concatenate([y_train_maj, y_min])

        return X_train, y_train

    def fit(self, X, y, label_maj=0, label_min=1, verbose=False):
        """Build a self-paced ensemble of estimators from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        y : array-like, shape = [n_samples]
            The target values (class labels).
        
        label_maj : int, bool or float, optional (default=0)
            The majority class label, default to be negative class.
            
        label_min : int, bool or float, optional (default=1)
            The minority class label, default to be positive class.
        
        Returns
        ------
        self : object
        """
        self.estimators_ = []
        # Initialize by spliting majority / minority set
        X_maj = X[y==label_maj]; y_maj = y[y==label_maj]
        X_min = X[y==label_min]; y_min = y[y==label_min]

        # Random under-sampling in the 1st round (cold start)
        X_train, y_train = self._random_under_sampling(
            X_maj, y_maj, X_min, y_min)
        self.estimators_.append(
            self._fit_base_estimator(
                X_train, y_train))

        # Loop start
        for i_estimator in range(1, self._n_estimators):
            X_train, y_train = self._self_paced_under_sampling(
                X_maj, y_maj, X_min, y_min, i_estimator,)
            if verbose:
                print ('{:<12s} | Iter: {} X_maj: {} X_min: {} alpha: {:.3f}'.format(
                    'SPEnsemble', i_estimator, len(X_maj), len(X_min), np.tan(np.pi*0.5*(i_estimator/(self._n_estimators-1)))))
            self.estimators_.append(
                self._fit_base_estimator(
                    X_train, y_train))

        return self

    def predict_proba(self, X):
        """Predict class probabilities for X.

        The predicted class probabilities of an input sample is computed as
        the mean predicted class probabilities of the base estimators in the
        ensemble. If base estimators do not implement a ``predict_proba``
        method, then it resorts to voting and the predicted class probabilities
        of an input sample represents the proportion of estimators predicting
        each class.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        Returns
        -------
        p : array of shape = [n_samples, n_classes]
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
        """Predict class for X.

        The predicted class of an input sample is computed as the class with
        the highest mean predicted probability. If base estimators do not
        implement a ``predict_proba`` method, then it resorts to voting.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes.
        """
        y_pred_binarized = sklearn.preprocessing.binarize(
            self.predict_proba(X)[:,1].reshape(1,-1), threshold=0.5)[0]
        return y_pred_binarized
    
    def score(self, X, y):
        """Returns the average precision score (equivalent to the area under 
        the precision-recall curve) on the given test data and labels.
        
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples.

        y : array-like, shape = (n_samples) or (n_samples, n_outputs)
            True labels for X.

        Returns
        -------
        score : float
            Average precision of self.predict_proba(X)[:, 1] wrt. y.
        """
        return sklearn.metrics.average_precision_score(
            y, self.predict_proba(X)[:, 1])