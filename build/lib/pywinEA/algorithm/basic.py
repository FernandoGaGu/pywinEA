# Module that contains all the genetic algorithms implemented in the pywin module.
#
# Author: Fernando García <ga.gu.fernando@gmail.com>
#
# Dependencies
import numpy as np
import random
from tqdm import tqdm
# Module dependencies
from ..interface.algorithm import GAbase
from ._algorithm_init import _GA_init
from ..population.population import Population
from ..error.exceptions import *


class GA(GAbase):
    """
    Basic implementation of a mono-objective genetic algorithm (although the algorithm
    is mono-objective, multi-objective metrics can be coupled as long as they return a single value).
    In its initialization, you must specify a fitness function (pywin.fitness module) or a scikit-learn
    classifier and a Scikit-learn cross validation iterator. Additionally it is necessary to specify a
    population model (pywin.population.Population). Alternatively it is possible to indicate the population
    size, in this case a "simple" population will be created (pywin.population.Population).
    Additionally, it is necessary to provide the basic operators required by the genetic algorithm
    indicated below.


    Parameters
    ------------
    fitness: sklearn.base.BaseEstimator or pywin.fitness.interface.FitnessStrategy
        If a scikit-learn estimator is provided it is necessary to supply the cv parameter.

    cv: <optional> sklearn.BaseCrossValidator (Splitter class)
        Scikit-learn cross validator iterable.

    population: <optional> pywin.population.Population
        If a population is not provided it is necessary to specify the population size using argument
        population_size.

    population_size: <optionally> int
        Only necessary if a population is not provided. By default the algorithm will create a basic
        population (pywin.population.Population).

    selection: <optional> pywin.selection.interface.SelectionStrategy
        Individual selection strategy. By default pywin.selection.TournamentSelection (with two gladiators,
        a single winner and sampling without replacement).

    elitism: <optional> float or pywin.operators.interface.ElitismStrategy
        If a float (0-1) is provided, the algorithm will use pywin.operators.BestFitness to select the
        proportion of individuals best fitted according to the indicated percentage. Otherwise it is
        necessary to provide an operator that inherits from the ElitismStrategy interface.

    elitism_rate: <optional> float
        Percentage of individuals that will form the elite. Only required if an ElitismStrategy operator
        has been provided in the elitism parameter.

    annihilation: <optional> float or pywin.operators.interface.AnnihilationStrategy
        If a float number is provided, the worst individuals will be removed from the population by the
        indicated percentage and the population will be filled with new random or elite individuals (The
        percentage can be modified using the fill_with_elite parameter). In this case the algorithm will use the
        AnnihilateWorse annihilation strategy.
        If an Annihilation instance is provided it is necessary to specify the annihilation_rate parameter.

    annihilation_rate: <optional> float
        If a float number is provided, the worst individuals will be removed from the population by the
        indicated percentage and the population will be filled with new random individuals or elite individuals
        (The percentage can be modified using the fill_with_elite parameter).

    fill_with_elite: <optional> float
        Percentage of annihilated individuals to be replaced from elite individuals. To specify this parameter
        it is necessary to provide elitism parameter.

    crossover: <optional> pywin.operators.interface.CrossOverStrategy
        Individual cross-over strategy. By default pywin.operators.OnePoint.

    mutation: <optional> pywin.operators.interface.MutationStrategy
        Mutation strategy used to introduce changes in the population. With some strategies it may be necessary
        to indicate the mutation_rate parameter.

    mutation_rate <optional> float
        A parameter that indicates the probability of a random change occurring in a given individual. If this
        argument is provided without a mutation argument by default pywin.operators.RandomMutation will be used.

    imputer: <optional> pywin.imputation.interface.ImputationStrategy
        Strategy to handle missing values. The missing values will be imputed for each individual in the
        population using the individual features.

    generations: int
        Number of generations (the minimum number of generations is 1).

    positive_class: <optional> int
        Class that will be considered as positive class, the rest of the classes will be considered as negative
        classes.
        If a value is provided the class labels will be transformed -> Binary Classification.
        Otherwise they will not be modified -> Multiclass classification.

    random_state : <optional> int
        Random seed.

    id : <optional> str
        Algorithm identifier

    Attributes
    ------------
    best_features: (Getter) Return a list with the best features found by the algorithm.

    get_current_generation: (Getter) Return the current generation of the algorithm.

    population: (Getter) Return the current population of the algorithm.

    population_fitness: (Getter) Return the fitness of the current generation of the algorithm.

    best_performance: (Getter) Return a dict with the score and the best performance achieved.

    fitness: (Getter / Setter) Algorithm fitness strategy.

    generations: (Getter / Setter) Algorithm number of generations.

    selection: (Getter / Setter) Algorithm selection strategy.

    elitism: (Getter / Setter) Algorithm elitism strategy.

    elitism_rate: (Getter / Setter) Algorithm elitism rate.

    annihilation: (Getter / Setter) Algorithm annihilation strategy.

    annihilation_rate: (Getter / Setter) Algorithm annihilation rate.

    fill_with_elite: (Getter / Setter) Percentage of annihilated individuals replaced from the elite.

    mutation: (Getter / Setter) Algorithm mutation strategy.

    mutation_rate: (Getter / Setter) Algorithm mutation rate.

    imputer: (Getter / Setter) Algorithm imputation strategy.

    crossover: (Getter / Setter) Algorithm crossover strategy.

    positive_class: (Getter / Setter) Number of the class selected as positive class.

    random_state: (Getter / Setter) Random state.

    id: (Getter / Setter) Algorithm identifier.


    Methods
    ---------
    set_features(features): Allows you to assign the labels of the columns (the names of the predictor variables).
        If this function is not used before using the fit method, the corresponding numerical value will be assigned
        to the position of the predictor variable. It also can be used after training step.

    fit(X, y): Start algorithm execution.

    continue_training(generations): Continue training the algorithm for an extra number of generations.

    predict(X): Predict the labels of the predictor variables. It is necessary that the model
        has been previously adjusted with the fit method.

    training_evolution(): Returns the parameters monitored during training.It returns a dictionary with the
        following scheme:
            :key total_fitness
                Dictionary that stores as a key the generation and as a value the total fitness of the population.
            :key best_fitness
                Dictionary that stores as a key the generation and as a value the best individual fitness.
            :key mean_num_features
                Dictionary that stores as a key the generation and as a value the average number of features
                in the population.
            :key std_num_features
                Dictionary that stores as a key the generation and as a value the standard deviation in the number
                of features in the population.
            :key best_num_features
                Dictionary that stores as a key the generation and as a value the number of features in best individual.
            :key best_(cv_score)
                Dictionary that stores as a key the generation and as a value the performance measure of the best
                individual.
            :key mean_(cv_score)
                 Dictionary that stores as a key the generation and as a value the average performance in performance
                 measure population.
            :key std_(cv_score)
                 Dictionary that stores as a key the generation and as a value the standard deviation in performance
                 measure in population.
            :key best_features
                Array that in the first position stores the best combination of features found by the algorithm and in
                the second position the fitness of the individual.

    get_dataset(): Function that returns the dataset with the selected features.

    set_population(new_individuals): Method that allows to select a new population of solutions.

    save(file_name, dir_name, overwrite): This function allows to save the model into a file.

    load_model(file_name, dir_name): <class method> This function allows to load a model from a file.
    """

    def __init__(self, **kwargs):
        """
            __init__(**kwargs)

        Notes
        ------
        Use help(BasicGA) to view the required parameters.
        """
        # Call superclass
        super().__init__(**kwargs)

        # Validate parameters
        kwargs = _GA_init(kwargs)

        self.fitness = kwargs['fitness']
        self.selection = kwargs['selection']
        self.elitism = kwargs.get('elitism', None)
        self.elitism_rate = kwargs.get('elitism_rate', None)
        self.annihilation = kwargs.get('annihilation', None)
        self.annihilation_rate = kwargs.get('annihilation_rate', None)
        self.fill_with_elite = kwargs.get('fill_with_elite', 0)
        self.mutation = kwargs.get('mutation', None)
        self.mutation_rate = kwargs.get('mutation_rate', None)
        self.crossover = kwargs.get('crossover', None)

        # Internal attributes
        self._best_model = None
        self.best_individual = None

        # Monitoring parameters
        self._evolution = {
            "total_fitness": {},
            "best_fitness": {},
            "mean_num_features": {},
            "std_num_features": {},
            "best_num_features": {},
            "best_%s" % self.fitness.score: {},
            "mean_%s" % self.fitness.score: {},
            "std_%s" % self.fitness.score: {},
            "best_features": [0, 0]                # best_features[0] = Features; best_features[1] = Fitness
        }

    def __repr__(self):
        return f"GA(population_size={self._population.size} generations={self.generations} " \
               f"elitism={self.elitism} elitism_rate={self.elitism_rate} annihilation={self.annihilation})" \
               f"annihilation_rate={self.annihilation_rate} fill_with_elite={self.fill_with_elite} " \
               f"mutation={self.mutation} mutation_rate={self.mutation_rate} selection={self.selection} " \
               f"fitness={self.fitness} crossover={self.crossover} imputer={self.imputer} " \
               f"positive_class={self.positive_class} random_rate={self.random_state}"

    def __str__(self):
        return self.__repr__()

    @property
    def best_features(self):
        """
        It allows to obtain the best combination of features found by the algorithm.

        Returns
        ----------
        :return: list
            Best combination of features.
        """
        return [self._population.features[idx] for idx in self._evolution['best_features'][0]]

    @property
    def best_performance(self):
        """
        Returns the largest value found for the selected fitness function.

        Returns
        ---------
        :return dict
            Score and maximum value achieved by the algorithm.
        """
        return {self.fitness.score: max(list(self._evolution["best_%s" % self.fitness.score].values()))}

    def training_evolution(self):
        """
        Function that returns a dictionary with the data collected throughout the algorithm search.
        You can use the function plot_evolution() from pywin.visualization.Plotter to display it.

        Returns
        ----------
        :return: dict
            Dictionary with the total fitness of the population and best individual, the number
            of features in the population and best individual, and information on the performance achieved.
        :return str
            Score used to evaluate fitness.
        """
        return self._evolution, self.fitness.score

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Function that begins the execution of the algorithm. First, it processes the class labels to adapt
        them to a mono-class classification problem (only if positive_class has been provided). Then it initializes
        a random population and calls the _core() method where the main logic of the algorithm is defined.

            fit(X: np.ndarray, y: np.ndarray)

        Parameters
        ------------
        :param X: 2d-array
            Values of the target variables.
        :param y: 1d-array
            Labels of each condition.

        Returns
        ----------
        :return: pywin.algorithms.BasicGA
            Model fitted using the best feature combination.
        """

        # Transformation of class labels calling GenAlgBase method (If positive_class has been provided)
        self._y = self._label_processing(y.copy()) if self.positive_class is not None else y.copy()

        # Save data and current generation
        self._X = X
        self._current_generation = self.generations

        # If no column names have been provided, numbers are assigned by default
        if self._population.features is None:
            self._population.init_features([n for n in range(self._X.shape[1])])

        # Initializes population
        self._population.init()

        # Genetic algorithm core
        self._core(generation_start=0, generation_end=self.generations)

        return self

    def continue_training(self, generations: int):
        """
        This function allows to continue training the algorithm since the last generation.

            continue_training(generations: int)

        Parameters
        -------------
        :param generations: int
            Extra number of generations.

        Returns
        -----------
        :return: pywin.algorithms.BasicGA
            Model fitted using the best feature combination.
        """
        if not isinstance(generations, int):
            raise TypeError("Parameter generations must be an integer.")

        # Genetic algorithm core
        self._core(generation_start=self._current_generation, generation_end=(self._current_generation + generations))

        # Save last generation
        self._current_generation = self.generations + generations
        self.generations += generations

        return self

    def predict(self, X: np.ndarray):
        """
        Function that returns the predictions made by the best combination of model and features.

            predict(X: np.ndarray)

        Parameters
        ------------
        :param X: 2d-array
            Data that will be predicted.

        Returns
        ---------
        :return: 1d-array
            Predictions.
        """
        if self._best_model is None:
            raise UnfittedAlgorithm("Before use get_dataset you need to fit the algorithm.")

        # Create dataset calling GenAlgBase method
        X_data = self._create_dataset(self.best_individual, self._X)

        return self._best_model.predict(X_data)

    def get_dataset(self):
        """
        Function that returns a dataset with the selected features and associated class labels.

        Returns
        ---------
        :return: 2d-array
            Predictor variables.
        :return 1d-array
            Class labels.
        """
        if self._best_model is None:
            raise UnfittedAlgorithm("Before use get_dataset you need to fit the algorithm.")

        # Create dataset calling GenAlgBase method
        X_data = self._create_dataset(self.best_individual, self._X)

        return X_data, self._y

    @classmethod
    def load(cls, file_name: str, dir_name: str = './_PyWinModels'):
        """
        Class method that allows retrieving models stored in .algorithms files. By default the files will be recovered
        from _PyWinModels directory.

            load(file_name: str, dir_name: str = './_PyWinModels')

        Parameters
        -------------
        :param file_name: str
            File name.
        :param dir_name: <optional> str
            Directory where the file is located.

        Returns
        ----------
        :return: pywin.algorithms.BasicGA
            Model fitted using the best feature combination.
        """
        return super().load(file_name, dir_name)

    def _evaluate_fitness(self, generation: int, population: Population, annotate: bool = True):
        """
        Function that uses the estimator and the validation strategy provided in the constructor to assess the fitness
        of the individuals. The individuals with greater fitness will be those that contain the features that give a
        better performance based on pywin.fitness.FitnessStrategy.

        Parameters
        ------------
        :param generation: int
            Generation number.
        :param population: pywin.population.Population
            Population of individuals.
        :param annotate: bool
            If the parameter is selected as false the data will not be recorded. (By default true)

        Returns
        ----------
        :return: pywin.population.Population
            Population.
        """
        best_fitness, best_num_feats = 0, 0
        length = population.length
        population_fitness, population_num_feats, performance = [None] * length, [None] * length, [None] * length

        for n, individual in enumerate(population.individuals):

            # Dataset extraction using individual features
            X_data = self._create_dataset(individual, self._X)

            # Get scores using a FitnessStrategy
            score = self.fitness.eval_fitness(X=X_data, y=self._y, num_feats=len(population.features))
            population_fitness[n] = score
            individual.fitness = score

            # Annotate results
            if annotate:
                performance[n] = score
                population_num_feats[n] = len(individual)

                # Save best fitness in the current generation
                if score > best_fitness:
                    best_fitness = score

                # Save the best features subset and fitness value for all generations
                if best_fitness > self._evolution["best_features"][1]:
                    self.best_individual = individual
                    self._evolution['best_features'][0] = individual.features
                    self._evolution['best_features'][1] = best_fitness

        # Save monitoring parameters
        if annotate:
            best_fit_idx = np.argmax(population_fitness)
            self._evolution['total_fitness'][generation + 1] = np.sum(population_fitness)
            self._evolution['best_fitness'][generation + 1] = population_fitness[best_fit_idx]
            self._evolution['best_num_features'][generation + 1] = len(population.get_individual(best_fit_idx))
            self._evolution['mean_num_features'][generation + 1] = np.mean(population_num_feats)
            self._evolution['std_num_features'][generation + 1] = np.std(population_num_feats)
            self._evolution['best_%s' % self.fitness.score][generation + 1] = np.max(performance)
            self._evolution['mean_%s' % self.fitness.score][generation + 1] = np.mean(performance)
            self._evolution['std_%s' % self.fitness.score][generation + 1] = np.std(performance)

        return population

    def _annihilation(self, elite: Population):
        """
        Method that is responsible for remove the least fitted individuals from the population according to the
        pywin.operators.AnnihilationStrategy and filling the eliminated individuals with elite and/or random
        individuals.

        Parameters
        ------------
        :param elite: pywin.population.Population or None
            Elite population or None
        """
        if self.elitism is None and self.fill_with_elite > 0:
            raise InconsistentParameters(
                "The algorithm has not selected an elite selection strategy. Therefore, annihilated individuals "
                "cannot be filled with elite individuals. Change fill_with_elite to 0 (or don't provide the parameter) "
                "or select an elitism rate.")

        # Annihilate worst individuals
        surviving_population = self.annihilation.annihilate(self._population, self.annihilation_rate)

        random_population_length = int(self._population.size * self.annihilation_rate * (1 - self.fill_with_elite))

        if random_population_length > 1:
            # Fill with random population
            random_population = surviving_population.generate_random_population(size=random_population_length)

            # Evaluate new random population
            random_population = self._evaluate_fitness(1, random_population, annotate=False)

            # Merge random population with population
            surviving_population = self._population.merge_population(surviving_population, random_population)

        # Get the number of individuals that will be sampled from elite to fill the population
        elite_sample_length = self._population.length - surviving_population.length

        if elite_sample_length != 0:
            # Get random elite subsample
            elite_individuals = [elite.get_random_individual() for n in range(elite_sample_length)]

            # Create new population
            elite_sample = elite.create_new_population(size=len(elite_individuals), individuals=elite_individuals)

            # Merge elite sample and population
            surviving_population = self._population.merge_population(surviving_population, elite_sample)

        # Assign to population the new population
        self._population = surviving_population

    def _core(self, generation_start: int, generation_end: int):
        """
        Function that performs the main steps of the genetic algorithm. First, it evaluates the fitness of the  initial
        population. Then if an annihilation parameter has been provided it eliminates the worst individuals.  Next, if
        an elitism parameter has been provided it selects the best individuals. Finally this function carry out the
        selection process, the crossover and introduces mutations in the population.
        The process is repeated up to the number of generations indicated by the user.

        Parameters
        -------------
        :param generation_start: int
            Generation from which the algorithm should start
        :param generation_end: int
            Generation until which the algorithm must arrive.
        """

        # Evaluation of the population
        self._population = self._evaluate_fitness(-1, self._population, annotate=True)

        elite_length = 0
        elite = None

        info = " (BasicGA) Generations (form %d to %d)" % (generation_start, generation_end)

        for generation in tqdm(range(generation_start, generation_end), desc=info):

            # Apply elitism
            if self.elitism is not None:
                elite = self.elitism.select_elite(population=self._population, elitism=self.elitism_rate)
                elite_length = elite.length

            # Apply annihilation
            if self.annihilation_rate is not None:
                self._annihilation(elite=elite)

            # Apply selection
            best_individuals = self.selection.select(population=self._population,
                                                     new_pop_length=(self._population.size - elite_length))

            # Apply crossover
            offspring = self.crossover.cross_population(best_individuals)

            # Introduce mutations
            if self.mutation is not None:
                offspring = self.mutation.mutate(population=offspring, mutation_rate=self.mutation_rate)

            # Merge elite and offspring
            if self.elitism is not None and elite.length != 0:
                offspring = self._population.merge_population(offspring, elite)

            # Evaluate fitness
            self._population = self._evaluate_fitness(generation, offspring)

        # Fit the model using the best features subset
        X_data = self._create_dataset(self.best_individual, self._X)

        self._best_model = self.fitness.fit(X_data, self._y)
