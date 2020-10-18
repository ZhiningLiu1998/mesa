# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 02:27:20 2020
@author: ZhiningLiu1998
mailto: zhining.liu@outlook.com / v-zhinli@microsoft.com
"""

import os
import torch
import pandas as pd
import numpy as np
from gym import spaces
from sac_src.sac import SAC
from sac_src.replay_memory import ReplayMemory
from environment import EnsembleTrainingEnv
from utils import *

class Mesa(EnsembleTrainingEnv):
    """The ensemble imbalanced learning framework MESA.

    Parameters
    ----------
    args : arguments
        See arguments.py for more information.

    base_estimator : scikit-learn classifier object
        The base estimator used to build ensemble classifiers.
        NO need to support sample weighting. 
        Built-in `fit()`, `predict()`, `predict_proba()` methods are required.
    
    n_estimators : int, optional (default=10)
        The number of base estimators used to form an MESA ensemble.
    
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
        
    n_estimators : int
        The number of base estimators used to form an MESA ensemble.
    
    meta_sampler : object (SAC)
        The meta-sampler in MESA.
    
    env : object (EnsembleTrainingEnv)
        The ensemble training environment in MESA.
    
    memory : object (ReplayMemory)
        The replay memory for Soft Actor-Critic training.
    """
    def __init__(self, args, base_estimator, n_estimators=10):

        super(Mesa, self).__init__(args, base_estimator)

        # state-size = 2 x num_bins
        state_size = int(args.num_bins*2)
        action_space = spaces.Box(low=0.0, high=1.0, shape=[1], dtype=np.float32)

        self.args = args
        self.n_estimators = n_estimators
        self.base_estimator_ = base_estimator
        self.meta_sampler = SAC(state_size, action_space, self.args)
        self.env = EnsembleTrainingEnv(args, base_estimator)
        self.memory = ReplayMemory(self.args.replay_size)
    
    def meta_fit(self, X_train, y_train, X_valid, y_valid, X_test=None, y_test=None):
        """Meta-training process of MESA.
        
        Parameters
        ----------
        X_train : array-like of shape = [n_training_samples, n_features]
            The training data instances.

        y_train : array-like of shape = [n_training_samples]
            Labels for X_train.
            
        X_valid : array-like of shape = [n_validation_samples, n_features]
            The validation data instances.

        y_valid : array-like of shape = [n_validation_samples]
            Labels for X_valid.
            
        X_test : array-like of shape = [n_training_samples, n_features], optional (default=None)
            The test data instances.

        y_train : array-like of shape = [n_training_samples], optional (default=None)
            Labels for X_test.
        
        Returns
        ----------
        self : object (Mesa)
        """
        # initialize replay memory and environment
        self.env.load_data(X_train, y_train, X_valid, y_valid, X_test, y_test, train_ratio=self.args.train_ratio)
        self.memory = memory_init_fulfill(self.args, ReplayMemory(self.args.replay_size))
        
        self.scores = []
        total_steps = self.args.update_steps + self.args.start_steps
        num_steps, num_updates, num_episodes = 0, 0, 0

        # start meta-training
        while num_steps < total_steps:
            self.env.init()
            state = self.env.get_state()
            done = False

            # for each episode
            while not done:
                num_steps += 1

                # take an action
                if num_steps >= self.args.start_steps:
                    action, by = self.meta_sampler.select_action(state), 'mesa'
                else:
                    action, by = self.meta_sampler.action_space.sample(), 'rand'

                # store transition
                next_state, reward, done, info = self.env.step(action[0])
                reward = reward * self.args.reward_coefficient
                self.memory.push(state, action, reward, next_state, float(done))

                # update meta-sampler parameters
                if num_steps > self.args.start_steps:
                    for i in range(self.args.updates_per_step):
                        _, _, _, _, _ = self.meta_sampler.update_parameters(
                            self.memory, self.args.batch_size, num_updates)
                        num_updates += self.args.updates_per_step

                # print log to stdout
                if self.args.meta_verbose is 'full':
                    print ('Epi.{:<4d} updates{:<4d}| {} | {} by {}'.format(num_episodes, num_updates, info, action[0], by))

                if done:
                    num_episodes += 1
                    self.record_scores()
                    # record print mean score of latest args.meta_verbose_mean_episodes to stdout
                    self.verbose_mean_scores(num_episodes, num_updates, by)

        return self
    
    def record_scores(self):
        """Record the training/validation/test performance scores."""
        train_score = self.env.rater.score(self.env.y_train, self.env.y_pred_train_buffer)
        valid_score = self.env.rater.score(self.env.y_valid, self.env.y_pred_valid_buffer)
        test_score  = self.env.rater.score(self.env.y_test,  self.env.y_pred_test_buffer) if self.env.flag_use_test_set else 'NULL'
        self.scores.append([train_score, valid_score, test_score] if self.env.flag_use_test_set else [train_score, valid_score])
        return

    def verbose_mean_scores(self, num_episodes, num_updates, by):
        """Print mean score of latest n episodes to stdout.

        n = args.meta_verbose_mean_episodes
        
        Parameters
        ----------
        num_episodes : int
            The number of finished meta-training episodes. 

        num_updates : int
            The number of finished meta-sampler updates. 
        
        by : {'rand', 'mesa'}, string
            The way of selecting actions in the current episode.
        """
        if self.args.meta_verbose is 'full' or (self.args.meta_verbose != 0 and num_episodes % self.args.meta_verbose == 0):
            view_bound = max(-self.args.meta_verbose_mean_episodes, -len(self.scores))
            recent_scores_mean = np.array(self.scores)[view_bound:].mean(axis=0)
            print ('Epi.{:<4d} updates {:<4d} |last-{}-mean-{}| train {:.3f} | valid {:.3f} | test {:.3f} | by {}'.format(
                num_episodes, num_updates, self.args.meta_verbose_mean_episodes, self.args.metric, 
                recent_scores_mean[0], recent_scores_mean[1], recent_scores_mean[2], by))
        return

    def fit(self, X, y, X_valid, y_valid, n_estimators=None, verbose=False):
        """Build a MESA ensemble from training set (X, y) and validation set (X_valid, y_valid).

        Parameters
        ----------
        X : array-like of shape = [n_training_samples, n_features]
            The training data instances.

        y : array-like of shape = [n_training_samples]
            Labels for X.
            
        X_valid : array-like of shape = [n_validation_samples, n_features]
            The validation data instances.

        y_valid : array-like of shape = [n_validation_samples]
            Labels for X_valid.
        
        n_estimators : int, optional (default=self.n_estimators)
            The number of base estimators used to form an MESA ensemble.
        
        verbose: bool, optional (default=False)
            Whether to print progress messages to stdout.

        Returns
        ----------
        self : object (Mesa)
        """
        n_estimators = self.n_estimators if n_estimators is None else n_estimators
        self.load_data(X, y, X_valid, y_valid)
        self.init()
        self.actions_record = []
        for i in range(n_estimators-1):
            state = self.get_state()
            action = self.meta_sampler.select_action(state)
            self.actions_record.append(action[0])
            _, _, _, info = self.step(action[0], verbose)
            if verbose: 
                print ('{:<12s} | action: {} {}'.format('Mesa', action, info))
        return self
    
    def save_meta_sampler(self, directory='save_model', suffix='meta_sampler'):
        """Save trained meta-sampler to files.

        Parameters
        ----------
        directory : string, optional (default='save_model')
            The directory to save files.
            Create the directory if it does not exist.

        suffix : string, optional (default='meta_sampler')
            The actor network will be saved in {directory}/actor_{suffix}.
            The critic network will be saved in {directory}/critic_{suffix}.
        """
        directory_path = f'{directory}/'
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        actor_path = f'{directory_path}actor_{suffix}'
        critic_path = f'{directory_path}critic_{suffix}'
        self.meta_sampler.save_model(actor_path, critic_path)
        return
        
    def load_meta_sampler(self, directory='save_model', suffix='meta_sampler'):
        """Load trained meta-sampler from files.
        
        Parameters
        ----------
        directory : string, optional (default='save_model')
            The directory to load files.

        suffix : string, optional (default='meta_sampler')
            The actor network will be loaded from {directory}/actor_{suffix}.
            The critic network will be loaded from {directory}/critic_{suffix}.
        """
        directory_path = f'{directory}/'
        actor_path = f'{directory_path}actor_{suffix}'
        critic_path = f'{directory_path}critic_{suffix}'
        self.meta_sampler.load_model(actor_path, critic_path)
        return self