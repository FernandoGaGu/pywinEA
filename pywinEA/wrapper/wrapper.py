#  Module that defines wrappers that allow executing several algorithms in parallel.
#  
# Author: Fernando García <ga.gu.fernando@gmail.com> 
#
# External dependencies
import warnings
import multiprocessing as mp
# Module dependencies
from ..interface.algorithm import MOAbase
from ..interface.algorithm import GAbase
from ..interface.wrapper import WrapperBase
from ..error.exceptions import *


class Parallel(GAbase, WrapperBase):
    """
    Wrapper that allows executing various genetic algorithms provided as parameters in parallel. This class follows
    the basic interface defined for genetic algorithms so all its methods are available.
    """

    def __init__(self, *algorithms):
        """
            __init__(*algorithms)

        Notes
        ------
        A variable number of genetic algorithms must be provided (greater than 2).

        Parameters
        -------------
        :param subclass of GenAlgBase
        """
        # Check parameters
        Parallel.__check_init(algorithms)

        # Save algorithms
        self._algorithms = {n: algorithm for n, algorithm in enumerate(algorithms)}

    def __repr__(self):
        return f"Parallel({list(self._algorithms.values())})"

    def __str__(self):
        return self.__repr__()

    @property
    def algorithms(self):
        """
        Returns all algorithms.

        Returns
        --------
        :return list
        """
        return list(self._algorithms.values())

    @algorithms.setter
    def algorithms(self, new_algorithms: list):
        """
        Select new algorithms.

        Parameters
        ------------
        :param new_algorithms: list
        """
        if not isinstance(new_algorithms, list):
            raise TypeError("You must provide a list of algorithms.")
        if len(new_algorithms) < 2:
            raise TypeError("You must provide two or more algorithms.")

        self._algorithms = new_algorithms

    @property
    def best_performance(self):
        """
        Returns a list with the performance achieved by each algorithm

        Returns
        ---------
        :return: list
        """
        return [algorithm.best_performance for algorithm in self._algorithms.values()]

    @property
    def best_features(self):
        """
        Returns a list with the features selected by each algorithm.

        Returns
        --------
        :return list
        """
        return [algorithm.best_features for algorithm in self._algorithms.values()]

    @property
    def get_current_generation(self):
        """
        Returns the current generation for each algorithm.

        Returns
        ----------
        :return list
        """
        return [algorithm.get_current_generation for algorithm in self._algorithms.values()]

    def training_evolution(self):
        """
        Function implemented for compatibility with the interface.
        """
        print("Unimplemented function for wrappers.")

        return dict()

    def set_features(self, features):
        """
        Function that assigns to the numbers with which the genes (features) in the algorithm are encoded the real
        name of the feature. If no features are provided, the default names will be the column number.

        Parameters
        -------------
        :param features: 1d-array
            Features in the same order as the columns of the predictor variables.
        """
        for algorithm in self._algorithms.values():
            algorithm.set_features(features)

    def fit(self, X, y):
        """
        Function that calls the fit() method of each of the algorithms provided as parameters. Consult the
        documentation of each of the algorithms for more information.

        Parameters
        ------------
        :param X: 2d-array
            Predictor variables
        :param y: 1d-array
            Class labels

        Returns
        ----------
        :return: Parallel
            Models fitted.
        """

        def fit_algorithm(algorithm_id, algorithm, X, y, return_dict):
            """ Function to be mapped during parallelization """
            return_dict[algorithm_id] = algorithm.fit(X, y)

        # Initializes Manager to save object states
        manager = mp.Manager()
        return_dict = manager.dict()

        # Create the processes that will be parallelize
        processes = [mp.Process(target=fit_algorithm, args=(algorithm_id, algorithm, X, y, return_dict))
                     for algorithm_id, algorithm in self._algorithms.items()]

        # Start parallelization
        for p in processes:
            p.start()

        # Join results
        for p in processes:
            p.join()

        # Save updated algorithm
        for algorithm_id, algorithm in return_dict.items():
            self._algorithms[algorithm_id] = algorithm

        return self

    def continue_training(self, generations):
        """
        Function that calls the continue_training() method of each of the previously trained algorithms.

        Parameters
        -------------
        :param generations: int
            Extra number of generations.

        Returns
        -----------
        :return: Parallel
            Parallel with the models fitted using the best feature combination.
        """
        if not isinstance(generations, int):
            raise InconsistentParameters("Parameter generations must be an integer.")

        self.__check_fitted()

        def continue_alg(est_id, est, generations, return_dict):
            """ Function to be mapped during parallelization """
            return_dict[est_id] = est.continue_training(generations)

        # Initializes Manager to save object states
        manager = mp.Manager()
        return_dict = manager.dict()

        # Create the processes that will be parallelize
        processes = [mp.Process(target=continue_alg, args=(algorithm_id, algorithm, generations, return_dict))
                     for algorithm_id, algorithm in self._algorithms.items()]

        # Start parallelization
        for p in processes:
            p.start()

        # Join results
        for p in processes:
            p.join()

        # Save updated algorithm
        for algorithm_id, algorithm in return_dict.items():
            self._algorithms[algorithm_id] = algorithm

        return self

    def predict(self, X):
        """
        Function not available for wrappers.
        """
        print("Function not available for wrappers. You can extract the algorithms "
              "from the wrapper using the \"algorithms\" property")

        return None

    def get_dataset(self):
        """
        Implemented for compatibility with GenAlgBase interface
        """
        print("Method not available for wrappers. Use get_dataset_from() instead.")

    def get_dataset_from(self, key: int):
        """
        Function that returns a dataset with the selected features and associated class labels.
        In this case the dataset generated using the best performing model will be returned.

        Parameters
        ------------
        :param key: int
            Algorithm index.

        Returns
        ---------
        :return: X (2d-array)
            Dataset
        :return y (1d-array)
            Labels
        """
        self.__check_fitted()

        return self._algorithms[key].get_dataset()

    @staticmethod
    def __check_init(algorithms):
        """
        Function that checks the consistency of the parameters provided to the constructor.
        """
        for algorithm in algorithms:
            if not isinstance(algorithm, GAbase):
                raise InconsistentParameters("Models must inherit from GenAlgBase. Provided: %s" % type(algorithm))
        if len(algorithms) < 2:
            raise InconsistentParameters("You have to provide two or more GenAlgBase subclasses.")

    def __check_fitted(self):
        """
        Function that checks if the algorithms have been fitted.
        """
        for algorithm in self._algorithms.values():
            if algorithm is None or algorithm.get_current_generation == 0:
                raise UnfittedAlgorithm("It is necessary to fit the algorithm before calling this function.")

    def save(self, model=None, file_name=None, dir_name='./_PyWinModels', overwrite=False):
        """
        Function that saves trained models in the specified directory (by default _PyWinModels). The name of the
        files will consist of the id of the model preceded by par(N)_ where N indicates the number of the model
        within the Parallel instance. With overwrite parameter if there is a file with the same name it will be
        overwritten. Internally this method will call the superclass save() method. For more information consult
        GenAlgBase class.

        Parameters
        ------------
        :param model: None
            Implemented for compatibility. No effects.
        :param file_name: None
            Implemented for compatibility. No effects.
        :param dir_name: <optional> str
            Directory where the file is located.
        :param overwrite: bool
            True for overwrite files (False by default.)
        """
        # Save models using his ids as names
        import os
        # If the directory to store models doesn't exists it is created.
        if os.path.exists(dir_name):
            for algorithm in self._algorithms.values():
                if str(algorithm.id) in os.listdir(dir_name) and not overwrite:
                    raise FileExists("The model already exists in the selected directory. You can:\n"
                                     "- Change the model id.\n- Select override true.\n- Change the "
                                     "directory.\n- Delete the file %s", str(algorithm.id))
        for key, algorithm in self._algorithms.items():
            algorithm.save(file_name="par(%d)_%s" % (key, str(algorithm.id)),
                           dir_name=dir_name, overwrite=overwrite)


class IslandModel(Parallel):
    """
    Algorithm that allows executing several algorithms in parallel, bringing together the best solutions every
    certain number of generations (called batches) and reinitializing the populations using all the combination of
    the features present in the best solutions.
    """

    def __init__(self, *algorithms, num_batches: int = 1, verbose: bool = False):
        """
            __init__(*algorithms, num_batches: int = 1, verbose: bool = False)

        Note
        -----
        The number of batches determines the number of times the best solutions achieved by the algorithms will
        intersect.
        """
        super().__init__(*algorithms)

        # Number of intersections between populations
        self.num_batches = num_batches

        # To select the generation batches
        self.batch_size = int(self._algorithms[0].generations / num_batches)

        for algorithm in self._algorithms.values():
            algorithm.generations = self.batch_size

        self.verbose = verbose

    def __repr__(self):
        return f"IslandModel({list(self._algorithms.values())})"

    def __str__(self):
        return self.__repr__()

    def fit(self, X, y):
        """
        Function that calls the fit() method of each of the algorithms provided as parameters. Consult the
        documentation of each of the algorithms for more information.

        Parameters
        ------------
        :param X: 2d-array
            Predictor variables
        :param y: 1d-array
            Class labels

        Returns
        ----------
        :return: Parallel
            Models fitted.
        """
        # Fit first batch
        super().fit(X, y)

        for batch in range(self.num_batches - 1):
            if self.__select_features():
                super().continue_training(self.batch_size)
            else:
                print("ALGORITHM CONVERGENCE: The number of features has been reduced to 2. "
                      "Impossible to continue training.")
                return self

        return self

    @property
    def best_performance(self):
        """
        Returns a list with the performance achieved by each algorithm

        Returns
        ---------
        :return: list
        """
        return [algorithm.best_performance for algorithm in self.algorithms]

    def continue_training(self, generations):
        """
        Function that calls the continue_training() method of each of the previously trained algorithms.
        """
        warnings.warn("WARNING: The algorithm will continue training in parallel without any intersection.")
        super().continue_training(generations)

    def __select_features(self):
        """
        Private method that obtains the best solutions reached by each algorithm and selects the total of different
        features as new features of the individual algorithms by re-initializing their populations.
        """
        # Get the features of the best solutions found by each of the algorithms.

        selected_feats = list()

        for algorithm in self._algorithms.values():

            for features in algorithm.best_features:
                #  If the algorithm is multi-objective, it is necessary to reduce the
                # number of features by one dimension
                if isinstance(algorithm, MOAbase):
                    for feature in features:
                        selected_feats.append(feature)
                else:
                    selected_feats.append(features)

        # Remove repeated features
        selected_feats = list(set(selected_feats))

        #  If the number of features is equal to or less than two, the algorithm ends
        if len(selected_feats) <= 2:
            return False

        # Create new features subset
        pop_feat_keys = list(self._algorithms[0].population.features.keys())
        pop_feat_values = list(self._algorithms[0].population.features.values())

        new_features = {
            pop_feat_values.index(feature): feature for feature in selected_feats if feature in pop_feat_values}

        # Console output
        if self.verbose:
            print("Number of different features: %d" % len(new_features))
            print("Features: %r" % list(new_features.values()))

        # Update features
        for algorithm in self._algorithms.values():
            algorithm.population.features = new_features
            algorithm._restart_population()

        return True
