# Module that defines the strategies defined for the handling of missing values.
#
# Author: Fernando García <ga.gu.fernando@gmail.com>
#
# External dependencies
import numpy as np
# Module dependencies
from ..interface.imputation import ImputationStrategy


class DynamicValue(ImputationStrategy):
    """
    Imputation technique that replaces missing values dynamically using a backward/forward fill with randomization.
    """
    def __init__(self, seed=None):
        """
        __init__(self, seed=None)
        """
        if seed is not None and isinstance(seed, int):
            np.random.seed(seed)

    def __repr__(self):
        return "DynamicValue"

    def __str__(self):
        return self.__repr__()

    def impute(self, data: np.ndarray, y: np.ndarray):
        """
        Function that receives a dataset with missing values and replaces the values filling first in a forward
        way, that is, replacing missing values with the previous known value. Additionally, it may be the case
        in which a missing value is in the first row, therefore, after filling it using a forward strategy, it
        applies a backward filling to avoid the possible presence of missing values. Each time the impute()
        method is called the rows are shuffled randomly. The dataset is returned in the same order in which
        it is received.

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
        # Create an index
        idx = [n for n in range(data.shape[0])]

        # Append index as column
        data = np.hstack((data, np.array(idx).reshape(len(idx), 1)))

        # Handle imputation values using forward/backward fill with randomization
        np.random.shuffle(data)

        # Forward fill
        data = DynamicValue._forward_fill(data)

        # Backward fill (When the inverted dataset is provided, It's like performing a backward filling)
        data = DynamicValue._forward_fill(data[::-1])

        # Sort values by index
        data = data[np.argsort(data[:,data.shape[1]-1])]

        # Return data excluding the index
        return data[:, :data.shape[1]-1]

    @staticmethod
    def _forward_fill(data: np.ndarray):
        """
        Function that replaces missing values with the value of the previous instance.

        Parameters
        -------------
        :param data: 2d-array
            Dataset with missing values.

        Returns
        -----------
        :return: 2d-array
            Dataset filling using a forward strategy
        """
        last_values = None

        for row in data:
            if last_values is not None:
                # Get NaN values index
                idx = np.isnan(row)
                # Fill NaN values using last seen values
                row[idx] = last_values[idx]

            # Update last seen values
            last_values = row

        return data
