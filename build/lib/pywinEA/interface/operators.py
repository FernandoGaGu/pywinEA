# Module that defines the interface that will be common to all operators of the module.
#
# Author: Fernando García <ga.gu.fernando@gmail.com>
#
from abc import ABCMeta, abstractmethod
from ..population.population import Population


class AnnihilationStrategy:
    """
    Base class for annihilation strategies.
    """
    __metaclass__ = ABCMeta

    def __repr__(self):
        return "AnnihilationStrategy"

    def __str__(self):
        return self.__repr__()

    @staticmethod
    @abstractmethod
    def annihilate(population: Population, annihilation: float):
        """
        Method that takes a Population and eliminates the worst individuals using their fitness value.

        Parameters
        ------------
        :param population: Population
        :param annihilation: float
        """
        pass


class CrossOverStrategy:
    """
    Base class for cross-over strategies.
    """
    __metaclass__ = ABCMeta

    def __repr__(self):
        return "CrossOverStrategy"

    def __str__(self):
        return self.__repr__()

    @staticmethod
    @abstractmethod
    def cross_population(population: Population):
        """
        Method that receives a population and carry out the cross-over between the individuals.

        Parameters
        ------------
        :param population: Population
        """
        pass


class ElitismStrategy:
    """
    Base class for elitism strategies.
    """
    __metaclass__ = ABCMeta

    def __repr__(self):
        return "ElitismStrategy"

    def __str__(self):
        return self.__repr__()

    @staticmethod
    @abstractmethod
    def select_elite(population: Population, elitism: float):
        """
        Method that takes a population and select the best individuals (elite)

        Parameters
        ------------
        :param population: Population
        :param elitism: float
        """
        pass


class MutationStrategy:
    """
    Base class for mutation strategies.
    """
    __metaclass__ = ABCMeta

    def __repr__(self):
        return "MutationStrategy"

    def __str__(self):
        return self.__repr__()

    @staticmethod
    @abstractmethod
    def mutate(population: Population, mutation_rate: float):
        """
        Method that receives a population and mutation rate and introduce variations in the individuals.

        Parameters
        ------------
        :param population: Population
        :param mutation_rate: float
        """
        pass


class SelectionStrategy:
    """
    Base class for selection strategies.
    """
    __metaclass__ = ABCMeta

    def __repr__(self):
        return "SelectionStrategy"

    def __str__(self):
        return self.__repr__()

    @abstractmethod
    def select(self, population: Population, new_pop_length: int):
        """
        This method receives a population, and the size of the new population that will be generated

        Parameters
        ------------
        :param population: Population
        :param new_pop_length: int

        Returns
        ----------
        :return: Population
        """
        pass


