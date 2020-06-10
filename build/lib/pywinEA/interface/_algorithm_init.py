# Module that contains all the genetic algorithms implemented in the pywin module.
#
# Author: Fernando García <ga.gu.fernando@gmail.com>
#
# Module dependencies
from .operators import MutationStrategy
from .operators import SelectionStrategy
from .operators import CrossOverStrategy
from .fitness import FitnessStrategy
from .imputation import ImputationStrategy
from ..population.population import Population
from ..error.exceptions import *

# Module dependencies (default values)
from ..operators.individual import RandomMutation
from ..operators.population import TournamentSelection
from ..operators.individual import OnePoint


def _GAbase_init(kwargs: dict):
    """
    Function that verifies that all the necessary arguments for the initialization of the GAbase
    algorithm have been provided.
    """
    if (kwargs.get('population', None) is not None) and (not isinstance(kwargs.get('population'), Population)):
        raise InconsistentParameters(
            "Incorrect population parameter, please provide a valid population (using Population), or, "
            "instead, define parameters: population_size and optionally max_feat_init/min_feat_init.")

    if not isinstance(kwargs.get('population'), Population):
        if kwargs.get('population_size', None) is None:
            raise InconsistentParameters(
                "Parameter population_size must be defined as an integer greater than 10 or, "
                "instead, provide a population using Population instance.")

        elif kwargs['population_size'] < 10:
            raise InconsistentParameters(
                "Parameter population_size must be defined as an integer greater than 10 or, "
                "instead, provide a population using Population instance.")

        else:
            # Create a population
            kwargs['population'] = Population(size=kwargs['population_size'],
                                              min_feat_init=kwargs.get('min_feat_init', 2),
                                              max_feat_init=kwargs.get('max_feat_init', None))

    # Check if the imputation strategy is valid.
    if kwargs.get('imputer', None) is not None:
        if not isinstance(kwargs['imputer'], ImputationStrategy):
            raise InconsistentParameters("Invalid imputation strategy type. Use a valid ImputationStrategy class.")

    # Check number of generations
    if kwargs.get('generations', 0) <= 0:
        raise InconsistentParameters("Incorrect generations parameter, it must be an positive integer greater than 0.")

    # Check random state
    if not isinstance(kwargs.get('random_state', 1), int) or kwargs.get('random_state', 1) < 1:
        raise InconsistentParameters("random_state must be an integer greater than 1.")

    return kwargs


def _MOAbase_init(kwargs: dict):
    """
    Function that verifies that all the necessary arguments for the initialization of the MOAbase
    algorithm have been provided.
    """
    if kwargs.get('fitness', None) is None:
        raise InconsistentParameters(
            "You must provide a fitness value (for example MonoObjectiveCV from pywin.fitness).")
    else:
        if not isinstance(kwargs['fitness'], list):
            # Only one metric to optimize
            if not kwargs.get('optimize_features', False):
                raise InconsistentParameters("Only one metric has been provided for optimization.")
            # Wrong optimize_features parameter
            elif not isinstance(kwargs['optimize_features'], bool):
                raise InconsistentParameters("Parameter optimize_features must be True or False.")
            # Add fitness function to list
            else:
                kwargs['fitness'] = [kwargs['fitness']]
        else:
            if len(kwargs['fitness']) == 1 and not kwargs.get('optimize_features', False):
                raise InconsistentParameters("Only one metric has been provided for optimization.")

            for fitness_func in kwargs['fitness']:
                if not isinstance(fitness_func, FitnessStrategy):
                    raise InconsistentParameters(
                        f"Invalid fitness strategy, required FitnessStrategy provided: {type(fitness_func)}")

    # Check selection parameter
    if kwargs.get('selection', None) is None:
        # Default selection strategy TournamentSelection
        kwargs['selection'] = TournamentSelection(k=2, replacement=False, winners=1)
    else:
        if not isinstance(kwargs['selection'], SelectionStrategy):
            raise InconsistentParameters("Invalid selection technique. Use a valid SelectionStrategy class.")

    # Check mutation parameter
    if kwargs.get('mutation_rate', None) is not None:
        if isinstance(kwargs['mutation_rate'], float):
            if kwargs['mutation_rate'] >= 1 or kwargs['mutation_rate'] < 0:
                raise InconsistentParameters("mutation_rate must be a number between 0 and 1")
            else:
                kwargs['mutation'] = RandomMutation
        else:
            if not isinstance(kwargs.get('mutation', None), MutationStrategy):
                kwargs['mutation'] = RandomMutation

    # Check cross-over parameter
    if kwargs.get('crossover', None) is None:
        # Default selection strategy TournamentSelection
        kwargs['crossover'] = OnePoint()
    else:
        if not isinstance(kwargs['crossover'], CrossOverStrategy):
            raise InconsistentParameters("Invalid cross-over strategy type. Use a valid CrossOverStrategy class.")

    #  Check the user-defined function for optimizing the number of features.
    if kwargs.get('features_function', None) is not None and callable(kwargs['features_function']):
        individual = [1, 2, 3]
        try:
            #  Check if the function receives a variable called "individual"
            return_value = kwargs['features_function'](individual, 1)
        except:
            raise FeaturesFunctionError(
                "Impossible to evaluate the number of features of a solution with the provided function. Provide a "
                "valid function for parameter features_function or leave it as default. The function must receive an"
                " \"individual\" parameter and return a single numerical value to be maximized")

        #  Check if the return value is a single value
        if not (isinstance(return_value, int) or isinstance(return_value, float)):
            raise FeaturesFunctionError(
                "Impossible to evaluate the number of features of a solution with the provided function. Provide a "
                "valid function for parameter features_function or leave it as default. The function must receive an"
                " \"individual\" parameter and return a single numerical value to be maximized")

    return kwargs
