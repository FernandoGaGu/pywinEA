# Module that contains all the genetic algorithms implemented in the pywin module.
#
# Author: Fernando García <ga.gu.fernando@gmail.com>
#
import warnings
from scipy.spatial.distance import euclidean, minkowski, cosine, canberra
# Module dependencies
from ..interface.operators import ElitismStrategy
from ..interface.operators import AnnihilationStrategy
from ..interface.operators import SelectionStrategy
from ..interface.operators import CrossOverStrategy
from ..interface.fitness import FitnessStrategy
from ..error.exceptions import *
# Default
from ..operators.population import AnnihilateWorse
from ..operators.population import TournamentSelection
from ..operators.population import BestFitness
from ..operators.individual import OnePoint
from ..operators.individual import RandomMutation
from ..fitness import MonoObjectiveCV


def _GA_init(kwargs: dict):
    """
    Function that verifies that all the necessary arguments for the initialization of the GA
    algorithm have been provided.
    """
    # Check elitism parameter
    if kwargs.get('elitism', None) is not None:
        if isinstance(kwargs['elitism'], float) and (0 < kwargs['elitism'] < 1):
            kwargs['elitism_rate'] = kwargs['elitism']
            kwargs['elitism'] = BestFitness
        else:
            if not isinstance(kwargs['elitism'], ElitismStrategy):
                raise InconsistentParameters(
                    "Incorrect elitism parameter, please provide a valid elitism strategy (ElitismStrategy). Or "
                    "indicates a percentage (float). You can also enter a percentage using the elitism_rate parameter")

            elif kwargs.get('elitism_rate', None) is None:
                warnings.warn("Elitism strategy provided but no elitism rate. Select a valid elitism_rate")

            elif kwargs['elitism_rate'] <= 0 or kwargs['elitism_rate'] >= 1:
                raise InconsistentParameters(
                    "Percentage of elitism out of bounds. Select an elitism rate between 0 and 1.")

    # Check annihilation parameter
    if kwargs.get('annihilation', None) is not None:

        if isinstance(kwargs['annihilation'], AnnihilationStrategy):
            if kwargs.get('annihilation_rate', None) is not None:
                if kwargs['annihilation_rate'] < 0 or kwargs['annihilation_rate'] > 1:
                    raise InconsistentParameters("The annihilation parameter must be within the range 0 - 1")
            else:
                warnings.warn(
                    "Annihilation strategy provided but no annihilation rate. Select a valid annihilation_rate")

        else:
            if isinstance(kwargs['annihilation'], float):
                if 1 > kwargs['annihilation'] > 0:
                    kwargs['annihilation_rate'] = kwargs['annihilation']
                    kwargs['annihilation'] = AnnihilateWorse
                else:
                    raise InconsistentParameters(
                        "Provides a valid annihilation strategy (for example AnnihilateWorse() from pywin.operators) "
                        "or select a suitable value for the annihilation or annihilation_rate parameter within the "
                        "range 0 - 1")
            else:
                raise InconsistentParameters(
                    "Provides a valid annihilation strategy (for example AnnihilateWorse() from pywin.operators) "
                    "or select a suitable value for the annihilation or annihilation_rate parameter within the "
                    "range 0 - 1")

        if kwargs.get('fill_with_elite', None) is None:
            kwargs['fill_with_elite'] = 0

        elif not 0 <= kwargs['fill_with_elite'] <= 1:
            raise InconsistentParameters("The parameter fill_with_elite must be within the range 0 - 1")

    # Check mutation parameter
    if kwargs.get('mutation_rate', None) is not None:

        if isinstance(kwargs['mutation_rate'], int):
            raise InconsistentParameters("Parameter mutation_rate must be a number between 0 and 1")

        if isinstance(kwargs['mutation_rate'], float):

            if not (1 > kwargs['mutation_rate'] > 0):
                raise InconsistentParameters("Parameter mutation_rate must be a number between 0 and 1")
            else:
                kwargs['mutation'] = RandomMutation
        else:
            if isinstance(kwargs.get('mutation', None), float):
                kwargs['mutation_rate'] = kwargs['mutation']
                kwargs['mutation'] = RandomMutation

    # Check selection parameter
    if kwargs.get('selection', None) is None:
        # Default selection strategy TournamentSelection
        kwargs['selection'] = TournamentSelection(k=2, replacement=False, winners=1)
    else:
        if not isinstance(kwargs['selection'], SelectionStrategy):
            raise InconsistentParameters("Invalid selection technique. Use a valid SelectionStrategy class.")

    # Check cross-over parameter
    if kwargs.get('crossover', None) is None:
        # Default crossover strategy OnePoint
        kwargs['crossover'] = OnePoint()
    else:
        if not isinstance(kwargs['crossover'], CrossOverStrategy):
            raise InconsistentParameters("Invalid cross-over strategy type. Use a valid CrossOverStrategy class.")

    # Check fitness parameter
    if not isinstance(kwargs.get('fitness', None), FitnessStrategy):
        if kwargs.get('cv', None) is None:
            raise InconsistentParameters("You must provide a valid cross-validatior iterator "
                                         "in cv parameter (sklearn.BaseCrossValidator).")

        if kwargs.get('fitness', None) is None:
            raise InconsistentParameters("You must provide a valid fitness function or a FitnessStrategy in fitness "
                                         "parameter (sklearn.base.BaseEstimator or pywin.operators.FitnessStrategy).")

        # Create MonoObjectiveCV using accuracy as default metric
        kwargs['fitness'] = MonoObjectiveCV(
            estimator=kwargs['fitness'], cv=kwargs['cv'], score=kwargs.get('score', 'accuracy'),
            n_jobs=kwargs.get('n_jobs', 1))

    return kwargs


def _SPEA2_init(kwargs: dict):
    """
    Function that verifies that all the necessary arguments for the initialization of the SPEA2
    algorithm have been provided.
    """
    AVAILABLE_DISTANCES = ['euclidean', 'minkowski', 'cosine', 'canberra']

    if kwargs.get('distance', None) is not None:
        if isinstance(kwargs['distance'], str):
            input_distance = kwargs['distance'].lower()

            if input_distance == 'euclidean':
                kwargs['distance'] = euclidean
            elif input_distance == 'minkowski':
                kwargs['distance'] = minkowski
            elif input_distance == 'cosine':
                kwargs['distance'] = cosine
            elif input_distance == 'canberra':
                kwargs['distance'] = canberra
            else:
                raise InconsistentParameters("Distance to estimate the density of solutions, invalid, "
                                             "available distances: %s" % ", ".join(AVAILABLE_DISTANCES))
        else:
            raise InconsistentParameters("Distance to estimate the density of solutions, invalid, "
                                         "available distances: %s" % ", ".join(AVAILABLE_DISTANCES))
    else:
        kwargs['distance'] = euclidean

    if kwargs.get('archive_length', None) is not None:
        if kwargs['archive_length'] < 5:
            raise InconsistentParameters("The length of the file cannot be less than 5")

    return kwargs
