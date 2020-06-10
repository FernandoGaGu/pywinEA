# Module containing the fitness functions used to evaluate individuals.
#
# Author: Fernando García <ga.gu.fernando@gmail.com>
#
# External dependencies
import numpy as np
from sklearn.model_selection import BaseCrossValidator, cross_val_score
from sklearn.base import BaseEstimator
# Module dependencies
from ..interface.fitness import FitnessStrategy


class MonoObjectiveCV(FitnessStrategy):
    """
    Class that calculates an individual's fitness using an estimator and a cross-validation strategy.
    """
    def __init__(self, estimator: BaseEstimator, cv: BaseCrossValidator, score: str, n_jobs: int = 1):
        """
        __init__(estimator: BaseEstimator, cv: BaseCrossValidator, score: str, n_jobs: int)

        """
        self.estimator = estimator
        self.cv = cv
        self.n_jobs = n_jobs
        self._score = score
        self._score_repr = score

    def __repr__(self):
        return f"MonoObjectiveCV(estimator={str(self.estimator)} cv={str(self.cv)} " \
               f"n_jobs={self.n_jobs} score={self._score}"

    def __str__(self):
        return self.__repr__()

    @property
    def score(self):
        """
        Return scores representation.

        Returns
        ---------
        :return str
        """
        return self._score_repr

    def eval_fitness(self, X, y, num_feats):
        """
        Function that evaluates the quality of a solution using the estimator and the cross-validation strategy.

        Parameters
        ------------
        :param X: 2d-array
            Predictor variables
        :param y: 1d-array
            Class labels
        :param num_feats: int
            Total number of features

        Returns
        ---------
        :return float
        """
        scores = cross_val_score(estimator=self.estimator, X=X, y=y, cv=self.cv, n_jobs=self.n_jobs, scoring=self._score)

        return np.mean(scores)

    def fit(self, X, y):
        """
                Fit the estimator.

        Parameters
        -----------
        :param X: 2d-array
            Predictor variables.
        :param y: 1d-array
            Class labels

        Returns
        ---------
        :return sklearn.base.BaseEstimator
        """

        return self.estimator.fit(X, y)
