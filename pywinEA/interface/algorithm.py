# Module that defines the interface that will be common to all the algorithms of the module.
#
# Author: Fernando García <ga.gu.fernando@gmail.com>
#
# Dependencies
import numpy as np
import random
from abc import ABCMeta, abstractmethod
# Module dependencies
from ._algorithm_init import _GAbase_init
from ._algorithm_init import _MOAbase_init
from ..population.representation import Individual
from ..error.exceptions import *


class GAbase:
    """
    Base class for all genetic algorithms.
    """
    __metaclass__ = ABCMeta

    def __init__(self, **kwargs):
        """
            __init__(**kwargs)
        """
        kwargs = _GAbase_init(kwargs)

        # Algorithm initialization
        self._population = kwargs['population']
        self.generations = kwargs['generations']
        self._current_generation = 0
        self.imputer = kwargs.get('imputer', None)
        self.positive_class = kwargs.get('positive_class', None)
        self.random_state = kwargs.get('random_state', None)
        self.id = kwargs.get('id', id(self))

        self._X, self._y = None, None

        # Select random state
        if self.random_state is not None:
            random.seed(self.random_state)
            np.random.seed(self.random_state)

    def __eq__(self, other: object):
        """
        Comparison operator.

        Parameters
        -------------
        :param other: object

        Returns
        ---------
        :return bool
        """
        if not isinstance(other, __class__):
            return False
        return True

    @property
    def get_current_generation(self):
        """
        Returns the current generation.

        Returns
        ----------
        :return: int
            Current generation.
        """
        return self._current_generation

    @property
    def population(self):
        """
        Returns the population of the last generation explored by the algorithm. If the algorithm has not
        been fitted  it will return None.

        Returns
        ---------
        :return: pywin.population.Population
            Last population.
        """
        return self._population

    @property
    def population_fitness(self):
        """
        Returns the population fitness of the last generation explored by the algorithm. If the algorithm has
        not been fitted it will return None.

        Returns
        ---------
        :return: 1d-array / None
            Last population fitness.
        """
        return [individual.fitness for individual in self._population.individuals]

    def set_population(self, new_individuals: list):
        """
        It allows selecting a new population replacing the current population.

            set_population(new_individuals: list)

        Parameters
        -------------
        :param new_individuals: list
            New population.
        """
        if isinstance(new_individuals, list):
            self._population.set_new_individuals(new_individuals)
        else:
            raise TypeError("Invalid type for new population. Provided type (%s) must be Population" % new_individuals)

    def set_features(self, features: list):
        """
        Function that assigns to the numbers with which the genes (features) in the algorithm are encoded
        the real name of the feature. If no features are provided, the default names will be the column number.

            set_features(features: list)

        Parameters
        -------------
        :param features: list
            Features in the same order as the columns of the predictor variables passed to the fit method (see fit()).
        """
        self._population.init_features(features)

    def _label_processing(self, label_names: list):
        """
        Function that transforms the multi-class problem into a mono-class problem. It displays a warning message
        if less than 20% of the features belong to the target class (unbalanced problem).

        Parameters
        -------------
        :param label_names: list / 1d-array
            Array with class labels.

        Returns
        ---------
        :return: 1d-array
            Binary class labels.
        """
        labels, counts = np.unique(label_names, return_counts=True)
        dict_frequency = dict(zip(labels, counts))
        positive_class_freq = dict_frequency.setdefault(self.positive_class, None)

        if positive_class_freq is None:
            raise ImpossibleProcessLabels(
                "Label is not present. Available labels are: %r" % np.unique(label_names))

        elif positive_class_freq < (counts.sum() / 4):
            print("WARNING: Unbalanced problem.\n\tPositive class: %d\n\tNegative class: %d" %
                  (positive_class_freq, counts.sum() - positive_class_freq))

        # Transform labels
        label_names[np.where(label_names != self.positive_class)] = 0
        label_names[np.where(label_names == self.positive_class)] = 1

        self.positive_class = 1

        return label_names

    def _create_dataset(self, individual: Individual, X_data: np.ndarray):
        """
        Function that returns a dataset built using individual features. Additionally this function is responsible
        for handle the presence of missing values using a pywin.imputation.ImputationStrategy provided by the user
        applying transformations on the original variables.

        Parameters
        ------------
        :param individual: list
            Features
        :param X_data: 2d-array
            Original predictor variables.

        Returns
        -----------
        :return: 2d-array
            Dataset built from the individual.
        """
        X_data = X_data[:, individual.features]

        if np.isnan(X_data.sum()):
            # Throw and exception if the dataset contains imputation values and a strategy for imputation
            # hasn't been provided.
            if self.imputer is None:
                raise ImpossibleHandleMissing("Data contains missing values. Provide a fill strategy.")

            # Impute imputation values based on a pywin imputation strategy
            X_data = self.imputer.impute(X_data, self._y)

        return X_data

    def _restart_population(self):
        """
        Function that allows to re-initialize the population.
        """
        if len(self._population.features) < self._population.max_feat_init:
            # If the maximum number of features required for initialization is greater than the number of features
            # reduce the maximum number of features to the number of features.
            self._population.max_feat_init = len(self._population.features)
            # Restart population
            self._population.init()

        if len(self._population.features) <= 2:
            # Save model
            self.save(file_name=f"{str(self.id)}.pywin")
            # Raise Exception.
            raise MinimumNumberOfFeaturesAchieved(
                "It has not been possible to re-initialize the population because the number of features "
                "reached by the algorithm is 2. Model saved as: %s.pywin" % str(self.id))

        self._population.init()

    @property
    @abstractmethod
    def best_features(self):
        """
        It returns the best combination of features found by the algorithm.

        Returns
        --------
        :return: list
        """
        return list()

    @property
    @abstractmethod
    def best_performance(self):
        """
        It returns the best performance achieved by the algorithm.

        Returns
        ---------
        :return: float or list
        """
        return list()

    @abstractmethod
    def training_evolution(self):
        """
        Method that returns the data collected throughout the algorithm search.

        Returns
        ---------
        :return dict
        """
        return dict()

    @abstractmethod
    def get_dataset(self):
        """
        Function that returns a dataset with the selected features and associated class labels.
        """
        pass

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Method that starts the execution of the algorithm.

            fit(X: np.ndarray, y: np.ndarray)

        Parameters
        ------------
        :param X: 2d-array
            Predictor variables.
        :param y: 1d-array
            Class labels.
        """
        pass

    @abstractmethod
    def continue_training(self, generations: int):
        """
        This method allows to continue training the algorithm since the last generation, over the indicated
        number of generations.

            continue_training(generations: int)

        Parameters
        -----------
        :param generations: int
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray):
        """
        Method that returns the predictions made by the best combination of model and features found by
        the algorithm.

            predict(X: np.ndarray)

        Parameters
        ------------
        :param X: 2d-array
            Predictor variables.
        """
        pass

    def save(self, file_name: str, dir_name: str = './_PyWinModels', overwrite: bool = False):
        """
        Method that allows to serialize (using pickle) the model to later recover it and be able to continue
        the training process or explore the solutions. If a directory is not specified the model will be created
        and stored in _PyWinModels directory (If the folder does't exists it will be created in the current
        directory). With overwrite parameter if there is a file with the same name it will be overwritten.

            save(file_name: str, dir_name: str = './_PyWinModels', overwrite: bool = False)

        Parameters
        ------------
        :param file_name: str
            File name.
        :param dir_name: <optional> str
            Directory where the file is located.
        :param overwrite: bool
            True for overwrite files (False by default.)
        """
        import os
        import pickle

        # If the directory to store models doesn't exists it is created
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

        relative_path = "%s/%s" % (dir_name, file_name)

        # If the directory already exists but is empty or overwrite
        # parameters is true overrides the previous model
        if file_name not in os.listdir(dir_name) or overwrite:
            with open(relative_path, 'wb') as out:
                pickle.dump(self, out)
        else:
            raise TypeError("%s already exists, select overwrite=True to overwrite file %s"
                            % (relative_path, file_name))

    @classmethod
    def load(cls, file_name: str, dir_name: str = './_PyWinModels'):
        """
        Class method that allows retrieving models stored in pickle files. By default the files will
        be recovered from _PyWinModels directory.

            load(file_name: str, dir_name: str = './_PyWinModels')

        Parameters
        -------------
        :param file_name: str
            File name.
        :param dir_name: <optional> (str)
            Directory where the file is located.

        Returns
        ---------
        :return: pywin.algorithms.interface.GenAlgBase
            Genetic algorithm instance.
        """
        import pickle
        relative_path = '%s/%s' % (dir_name, file_name)
        return pickle.load(open(relative_path, 'rb'))


def features_function(individual, total_feats: int = None):
    """
    Default function to optimize the number of features: 1 - (num ind. Feat. / num total feat.)
    """
    return 1 - (len(individual) / total_feats)


class MOAbase(GAbase):
    """
     Base class for all multi-objective algorithms
    """
    __metaclass__ = ABCMeta

    def __init__(self, **kwargs):
        """
            __init__(**kwargs)
        """
        # Call superclass
        super().__init__(**kwargs)

        # Check parameters consistency
        kwargs = _MOAbase_init(kwargs)

        self.fitness = kwargs['fitness']
        self.selection = kwargs['selection']
        self.mutation = kwargs.get('mutation', None)
        self.mutation_rate = kwargs.get('mutation_rate', None)
        self.crossover = kwargs['crossover']
        self.optimize_features = kwargs.get('optimize_features', False)
        self.features_function = kwargs.get('features_function', features_function)

        # Monitoring parameters
        self._evolution = {
            "hypervolume": {},
            "num_solutions_front": {},
            "best_values": {}
        }

    @property
    def best_features(self):
        """
        Get features from non dominated Pareto front.

        Returns
        ---------
        :return: list
            List with the selected features in each of the solutions of the non-dominated front.
        """
        # Get individuals in Pareto front
        pareto_front = [self._population.individuals[idx] for idx in range(self._population.length)
                        if self._population.fitness[idx].rank == 0]

        # Get feature names
        selected_features = [[self._population.features[idx] for idx in individual] for individual in pareto_front]

        return selected_features

    @property
    def best_performance(self):
        """
        Returns the best values achieved for each of the objective functions.

        Returns
        --------
        :return: list
            List with the best function values in each of the solutions of the non-dominated front.
        """
        scores_ = [solution.values for solution in self._population.fitness if solution.rank == 0]

        #  Separate scores into lists
        scores_ = list(zip(*scores_))

        # Add score name
        scores = {f"{self.fitness[n].score}({n})": scores_[n] for n in range(len(self.fitness))}

        # If the features have been optimized add them
        if self.optimize_features:
            scores['feature_scores'] = scores_[-1]
            scores['num_features'] = [len(features) for features in self.best_features]

        return scores

    def predict(self, X: np.ndarray):
        """
        Method NOT available for multi-objective algorithms
        """
        print("Method not available for MultiObjective algorithms.")
        pass

    @abstractmethod
    def get_dataset(self):
        """
        Function that returns a dataset with the selected features and associated class labels.
        """
        pass

    @abstractmethod
    def training_evolution(self):
        """
        Method that returns the data collected throughout the algorithm search.

        Returns
        ---------
        :return dict
        """
        return dict()

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Method that starts the execution of the algorithm.

            fit(X: np.ndarray, y: np.ndarray)

        Parameters
        ------------
        :param X: 2d-array
            Predictor variables.
        :param y: 1d-array
            Class labels.
        """
        pass

    @abstractmethod
    def continue_training(self, generations: int):
        """
        This method allows to continue training the algorithm since the last generation, over the indicated
        number of generations.

            continue_training(generations: int)

        Parameters
        -----------
        :param generations: int
        """
        pass