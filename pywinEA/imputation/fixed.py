# Module that defines the strategies defined for the handling of missing values.
#
# Author: Fernando García <ga.gu.fernando@gmail.com>
#
# External dependencies
import numpy as np
# Module dependencies
from ..interface.imputation import ImputationStrategy


class FixedValue(ImputationStrategy):
    """
    Imputation technique that replaces missing values with a fixed value specified by the user.
    """
    def __init__(self, value: int = -1):
        """
        __init__(self, value: int = -1)
        """
        self.value = value

    def __repr__(self):
        return f"FixedValue(value={self.value})"

    def __str__(self):
        return self.__repr__()

    def impute(self, data, y):
        """
        Function that receives a dataset with missing values and replaces the values with the value provided
        by the user.

        Parameters
        -------------
        :param data: 2d-array
            Predictor variables with missing values
        :param y: 1d-array
            Class labels

        Returns
        ----------
        :return: 2d-array
        Dataset without imputation values.
        """
        data[np.isnan(data)] = self.value

        return data
