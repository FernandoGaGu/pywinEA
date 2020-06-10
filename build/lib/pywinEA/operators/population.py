# "pywin"
#
# Module that defines the operators used by the algorithms to modify the populations of solutions.
#
# Author: Fernando García <ga.gu.fernando@gmail.com>
#
# External dependencies
import numpy as np
from random import uniform
from functools import reduce
#  Module dependencies
from ..population.population import Population
from ..interface.operators import SelectionStrategy
from ..interface.operators import AnnihilationStrategy
from ..interface.operators import ElitismStrategy


class TournamentSelection(SelectionStrategy):
    """
    Class that implements the tournament selection strategy applicable to genetic algorithms.
    """

    def __init__(self, k: int, replacement: bool = False, winners: int = 1):
        """
        __init__(k: int, replacement: bool = False, winners: int = 1)

        Parameters
        ------------
        :param k: int
            Number of the individuals selected for tournament.
        :param replacement: bool
            It indicates if the selection of individuals is going to be done with replacement (an individual can
            be selected multiple times) or without replacement.
        :param winners: int
            Number of individuals that are selected as tournament winners.
        """
        self.k = k
        self.replacement = replacement
        self.winners = winners

    def __repr__(self):
        return f"TournamentSelection(k={self.k} replacement={self.replacement} winners={self.winners})"

    def __str__(self):
        return self.__repr__()

    def select(self, population: Population, new_pop_length: int):
        """
        Function that applies the tournament selection algorithm. It takes a total of k individuals randomly
        (without replacement / with replacement) from the population and returns the best of those k individuals.

        Parameters
        ------------
        :param population Population
        :param new_pop_length int
            Size of the individuals that will form the new population

        Returns
        ----------
        :return: Population
            Tournament winners
        """
        winners = []
        for n in range(int(new_pop_length / self.winners)):
            # Get individual indices with or without replacement
            gladiators_idx = np.random.choice(
                range(population.length), size=self.k, replace=self.replacement).tolist()

            # Get individuals from population
            gladiators = [population.get_individual(idx) for idx in gladiators_idx]

            # Get fitness
            fitness = [individual.fitness for individual in gladiators]

            # Get winners
            for best in range(self.winners):
                idx = fitness.index(max(fitness))
                winners.append(gladiators.pop(idx))
                fitness.pop(idx)

        # Create a population using the winners
        best_individuals = population.create_new_population(size=len(winners), individuals=winners)

        return best_individuals


class RouletteWheel(SelectionStrategy):
    """
    Class that implements the roulette wheel selection strategy applicable to genetic algorithms.
    Calculate the cumulative probability of survival of each individual based on their fitness.
    Individuals with a higher probability will have more opportunities to be randomly selected.
    If the fitness is not of the float type (for example in multi-objective algorithms) consider as
    fitness the sum of the values of the objective functions. (For the stability of the algorithm
    these values should be scaled between 0 and 1 for all target functions)
    """

    def __init__(self):
        pass

    def __repr__(self):
        return "RouletteWheel"

    def __str__(self):
        return self.__repr__()

    def select(self, population: Population, new_pop_length: int):
        """
        Select individuals from the population. The best individuals are more likely to survive.

        Parameters
        ------------
        :param population Population
        :param new_pop_length int
            Size of the individuals that will form the new population

        Returns
        ----------
        :return: Population
        """
        # Get individuals fitness
        fitness = [individual.fitness for individual in population.individuals]

        # Get individuals probability
        survival_prob = list(map(RouletteWheel._prob_solution, fitness))

        # Get total probability
        total_prob = reduce(lambda a, b: a + b, survival_prob)

        # Calculate individual probability (accumulative)
        survival_prob = list(map(lambda a: a / total_prob, survival_prob))
        survival_prob = np.cumsum(survival_prob)

        selected_individuals = []

        while len(selected_individuals) < new_pop_length:
            #  Get position
            individual_idx = np.where(survival_prob >= uniform(0, 1))[0][0]
            #  Append to new individuals
            selected_individuals.append(population.get_individual(individual_idx))

        #  Create the new population
        new_population = population.create_new_population(
            size=len(selected_individuals), individuals=selected_individuals
        )

        return new_population

    @staticmethod
    def _prob_solution(fitness):
        """
        Function that adds the values of various objective functions when the algorithm is multi-objective.
        If the fitness is not a numerical value it must be a Solution or "something" that implement the attribute
        values.
        """
        if not isinstance(fitness, float):
            return reduce(lambda a, b: a + b, fitness.values)

        return fitness


class AnnihilateWorse(AnnihilationStrategy):
    """
    Eliminate individuals who have a lower fitness value.
    """

    def __init__(self):
        pass

    def __repr__(self):
        return "AnnihilateWorse"

    def __str__(self):
        return self.__repr__()

    @staticmethod
    def annihilate(population: Population, annihilation: float):
        """
        This function eliminates the worst individuals according to the selected "annihilation" parameter.

        Parameters
        ------------
        :param population: Population
        :param annihilation: float
        """
        annihilation_size = int(population.length * annihilation)

        individuals = population.individuals

        # Sort the individuals
        sorted_by_fitness = sorted(individuals, key=lambda individual: individual.fitness)

        # Eliminate the worst individuals
        new_population = sorted_by_fitness[annihilation_size:]

        # Create new population
        surviving_population = population.create_new_population(
            size=population.size - annihilation_size, individuals=new_population
        )

        return surviving_population


class BestFitness(ElitismStrategy):
    """
    Create a new population using the individuals with the highest fitness value.
    """

    def __init__(self):
        pass

    def __repr__(self):
        return "BestFitness"

    def __str__(self):
        return self.__repr__()

    @staticmethod
    def select_elite(population: Population, elitism: float):
        """ FIX DESCRIPTION
        Function that selects the best individuals based on the elitism parameter.

        Parameters
        ------------
        :param population: Population
        :param elitism: float
            Percentage of individuals in the population that will form the elite.
        Returns
        ----------
        :return: Population
            Elite population
        """
        # Determine the elite size taking into account the elitism and population size.
        elite_length = int(population.length * elitism)

        individuals = population.individuals

        # Sort list and use the indices to get the best individuals (descending)
        sorted_by_fitness = sorted(individuals, key=lambda individual: individual.fitness, reverse=True)

        # Get elite
        elite_individuals = sorted_by_fitness[:elite_length]

        # Create elite population
        elite = population.create_new_population(size=elite_length, individuals=elite_individuals)

        return elite
