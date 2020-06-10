# Module that defines the interface of the fitness functions used to evaluate the quality of the solutions.
#
# Author: Fernando García <ga.gu.fernando@gmail.com>
#
import numpy as np
from abc import ABCMeta, abstractmethod


class FitnessStrategy:
    """
    Base class for all fitness functions.
    """
    __metaclass__ = ABCMeta

    def __repr__(self):
        return "FitnessStrategy"

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        if not isinstance(other, __class__):
            return False
        return True

    @property
    @abstractmethod
    def score(self):
        """
        Function that return the score used to assess fitness
        """
        pass

    @abstractmethod
    def eval_fitness(self, X: np.ndarray, y: np.ndarray, num_feats: list):
        """
        Function that takes a population and return a list of scores related with individual
        fitness.
        """
        pass

    @abstractmethod
    def fit(self, X, y):
        """
        Function that fit the algorithm to a given dataset and return the model fitted.
        """
        pass
