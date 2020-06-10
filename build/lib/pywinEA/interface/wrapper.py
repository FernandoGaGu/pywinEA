# Module that defines the basic interface that wrappers must implement.
#
# Author: Fernando García <ga.gu.fernando@gmail.com> 
#
from abc import ABCMeta, abstractmethod


class WrapperBase:
    """
    Base class for wrapper classes.
    """
    __metaclass__ = ABCMeta

    def __repr__(self):
        return "WrapperBase"

    def __str__(self):
        return self.__repr__()

    @property
    @abstractmethod
    def algorithms(self):
        """
        Return all fitted algorithms.
        """
        pass

