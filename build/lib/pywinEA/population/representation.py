#  Module that contains the implementation of the populations used in the algorithms
#
# Author: Fernando García <ga.gu.fernando@gmail.com>
#


class Individual:
    """
    Class representing an individual. The representation of its genotype consists of an
    array of integer values of variable length.
    """

    def __init__(self, features):
        """
        __init__( features)
        """
        self.features = features
        self._fitness = None

    def __repr__(self):
        return f"Individual(features={self.features} fitness={self._fitness}"

    def __str__(self):
        return self.__repr__()

    def __getitem__(self, item):
        """
        DESCRIPTION

        Parameters
        -----------
        :param item: int
            Feature index.

        Returns
        --------
        :return int
            Feature as integer value.
        """
        return self.features[item]

    def __len__(self):
        """
        Returns the length of the individual (number of features).

        Returns
        --------
        :return: int
        """
        return len(self.features)

    @property
    def fitness(self):
        """
        Returns the fitness value of the individual

        Returns
        --------
        :return: float (GA) / Solution (MOAs)
            Fitness value
        """
        return self._fitness

    @fitness.setter
    def fitness(self, new_fitness):
        """
        Assign the fitness value to the individual

        Parameters
        -----------
        :param new_fitness: float (GA) / Solution (MOAs)
        """
        self._fitness = new_fitness
