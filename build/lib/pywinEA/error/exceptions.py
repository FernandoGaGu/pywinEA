#  Module that defines the types of exceptions that the module can throw.
#
# Author: Fernando García <ga.gu.fernando@gmail.com>
#


class UnfittedAlgorithm(Exception):
    """
    Exception thrown when trying to access parameters or methods with untrained algorithm
    """
    pass


class MinimumNumberOfFeaturesAchieved(Exception):
    """
    Exception thrown when the algorithm has converged to less than two features.
    """
    pass


class ImpossibleProcessLabels(Exception):
    """
    Exception thrown when algorithm cannot process class labels.
    """


class FileExists(Exception):
    """
    Exception thrown when trying to overwrite existing files.
    """
    pass


class ImpossibleHandleMissing(Exception):
    """
    Exception thrown when data contains missing values and no way has been provided to handle it.
    """


class InconsistentParameters(Exception):
    """
    Exception thrown when the parameters provided to the algorithm are inconsistent.
    """


class FeaturesFunctionError(Exception):
    """
    Exception thrown when there is a failure due to the function to evaluate the number of features of a solution.
    """


class UnsuitableClassifier(Exception):
    """
    Exception thrown when a classifier is not suitable to perform a certain task.
    """
