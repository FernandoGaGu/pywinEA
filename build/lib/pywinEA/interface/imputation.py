# Module that defines the interface that will be common to all module imputation strategies
#
# Author: Fernando García <ga.gu.fernando@gmail.com>
#
import numpy as np
from abc import ABCMeta, abstractmethod


class ImputationStrategy:
    """
    Class that defines the interface that the imputation techniques must implement to be incorporated into a
    genetic algorithm. All subclasses derived from this class must implement the method impute().
    """
    __metaclass__ = ABCMeta

    def __repr__(self):
        return "ImputationStrategy"

    def __str__(self):
        return self.__repr__()

    @abstractmethod
    def impute(self, data: np.ndarray, y: np.ndarray):
        """
        Receive the data whose missing values are to be imputed.

        Parameters
        ------------
        :param data: 2d-array
            Predictor variables with missing values.
        :param y: 1d-array
            Class labels.

        Returns
        -----------
        :returns data: 2d-array
        """
        return np.ndarray
