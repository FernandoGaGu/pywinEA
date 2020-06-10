# Module that defines the operators used by the algorithms to modify the populations of solutions.
#
# Author: Fernando García <ga.gu.fernando@gmail.com>
#
# External dependencies
import numpy as np
import random
#  Module dependencies
from ..population.population import Population
from ..population.representation import Individual
from ..interface.operators import CrossOverStrategy
from ..interface.operators import MutationStrategy


class OnePoint(CrossOverStrategy):
    """
    Cross two individuals by randomly selecting a cut point and assembling the resulting halves together.
    """

    def __init__(self):
        pass

    def __repr__(self):
        return "OnePoint"

    def __str__(self):
        return self.__repr__()

    @staticmethod
    def cross_population(population: Population):
        """
        Function that selects individuals from the population (without replacement) to perform the crossover.
        When the population is odd an individual must be selected with replacement to maintain the population size.

        Parameters
        ------------
        :param population: Population

        Returns
        ----------
        :return: Population
            Offspring.
        """
        offspring = []
        odd = False

        # If the population is odd a random individual will participate in the crossover twice
        if population.length % 2 != 0:
            odd = True

        while population.length != 0:
            if odd:
                # Without removing individual from the population
                parent_1 = population.get_random_individual()
                odd = False
            else:
                # Removing individual from population
                parent_1 = population.pop_random_individual()

            parent_2 = population.pop_random_individual()

            # Generate two children
            son_1_feats, son_2_feats = OnePoint._generate_children(parent_1, parent_2, population.features_num)

            # Shuffle features and append it to offspring
            np.random.shuffle(son_1_feats)
            np.random.shuffle(son_2_feats)
            offspring.append(Individual(son_1_feats))
            offspring.append(Individual(son_2_feats))

        # Create a new population with the offspring
        population_offspring = population.create_new_population(size=len(offspring), individuals=offspring)

        return population_offspring

    @staticmethod
    def _generate_children(parent_1: Individual, parent_2: Individual, features: list):
        """
            Function that randomly cuts the chromosomes of the parents and mixes the four fragments generating
            the offspring.

            Parameters
            -------------
            :param parent_1: 1d-array
            :param parent_2: 1d-array
            :param features: 1d-array

            Returns
            ---------
            :return: tuple
                Two children.
        """
        parent_1_length = len(parent_1)
        parent_2_length = len(parent_1)

        #  If either parent is less than two features, a new one must be added randomly.
        if parent_1_length < 2 or parent_2_length < 2:
            print("WARNING: The length of one of the parents is less than 2. In order to participate "
                  "in the cross-over a random feature will be added")
            if parent_1_length < 2:
                parent_1.features = np.append(parent_1.features, np.random.choice(features))
            if parent_2_length < 2:
                parent_2.features = np.append(parent_2.features, np.random.choice(features))

        #  If one of the two parents has a length of 2 the only possible cut point is 1
        if parent_1_length == 2:
            cut_parent_1 = 1
        else:
            cut_parent_1 = random.randrange(1, parent_1_length - 1)

        if parent_2_length == 2:
            cut_parent_2 = 1
        else:
            cut_parent_2 = random.randrange(1, parent_2_length - 1)

        # Offspring without repetition of features
        son_1_feats, reps_1 = np.unique(
            np.array(
                parent_1.features[cut_parent_1:].tolist() + parent_2.features[cut_parent_2:].tolist()),
            return_counts=True
        )

        son_2_feats, reps_2 = np.unique(
            np.array(parent_1.features[:cut_parent_1].tolist() + parent_2.features[:cut_parent_2].tolist()),
            return_counts=True
        )

        # Fill repeated features with new ones
        reps_1 = sum(reps_1 > 1)
        reps_2 = sum(reps_2 > 1)

        if reps_1 > 0:
            not_used_features_son_1 = np.setdiff1d(np.array(features), son_1_feats)
            if not len(not_used_features_son_1) < reps_1:
                son_1_feats = np.append(
                    son_1_feats, np.random.choice(not_used_features_son_1, size=reps_1, replace=False)
                )

        if reps_2 > 0:
            not_used_features_son_2 = np.setdiff1d(np.array(features), son_2_feats)
            if not len(not_used_features_son_2) < reps_2:
                son_2_feats = np.append(
                    son_2_feats, np.random.choice(not_used_features_son_2, size=reps_2, replace=False)
                )

        return son_1_feats, son_2_feats


class RandomMutation(MutationStrategy):
    """
    Introduce random mutations.
    """

    def __init__(self):
        pass

    def __repr__(self):
        return "RandomMutation"

    def __str__(self):
        return self.__repr__()

    @staticmethod
    def mutate(population: Population, mutation_rate: float):
        """
        Function that introduces mutations inside the population based on the established mutation rate using an
        uniform random distribution.
        """
        for individual in population.individuals:

            # Generates an array of the same length as the individual with probabilities between 0 and 1.
            prob = np.random.uniform(low=0.0, high=1.0, size=len(individual))
            positions = np.where(prob <= mutation_rate)[0]

            # If there aren't positions to mutate go to the next individual
            if len(positions) == 0:
                continue

            # Get not used features
            not_used_features = np.setdiff1d(np.array(population.features_num), individual.features)

            # If the individual has all features, mutations cannot be introduced
            if len(not_used_features) == 0:
                continue

            # If there are more positions to mutate than features, mutate with replacement
            if len(not_used_features) < len(positions):
                replace = True
            else:
                replace = False

            # Mutate individual
            individual.features[positions] = np.random.choice(not_used_features, size=len(positions), replace=replace)

            # If mutations have occurred with replacement, eliminate repeated features
            if replace:
                individual.features = np.unique(individual.features)

            # Shuffle individual to avoid bias
            np.random.shuffle(individual.features)

        return population
