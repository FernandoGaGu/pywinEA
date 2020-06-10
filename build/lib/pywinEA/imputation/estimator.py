# Module that defines the strategies defined for the handling of missing values.
#
# Author: Fernando García <ga.gu.fernando@gmail.com>
#
# External dependencies
import numpy as np
# Module dependencies
from ..interface.imputation import ImputationStrategy


class SklearnImputer(ImputationStrategy):
	"""
	Class that allows to incorporate scikit-learn imputers to the genetic algorithm. With this class the values
	are imputed every time a dataset is generated from a new individual.
	"""

	def __init__(self, imputer):
		"""
		__init__(imputer: sklearn.preprocessing.Imputer):
		"""
		self.imputer = imputer

	def __repr__(self):
		return f"SklearnImputer(imputer={str(self.imputer)})"

	def __str__(self):
		return self.__repr__()

	def impute(self, data: np.ndarray, y: np.ndarray):
		"""
		Function that returns the imputed dataset using the scikit-learn imputer.

		Parameters
		-------------
		:param data: 2d-array
			Predictor variables with missing values
		:param y: 1d-array
			Class labels

		Returns
		----------
		:return: 2d-array
			Dataset without missing values.
		"""
		return self.imputer.fit_transform(data, y)[0]
