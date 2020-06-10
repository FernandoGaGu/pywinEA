# Module that contains all the genetic algorithms implemented in the pywin module.
#
# Author: Fernando García <ga.gu.fernando@gmail.com>
#
# External dependencies
import numpy as np
from math import sqrt
from functools import total_ordering
from tqdm import tqdm
#  Module dependencies
from ..interface.algorithm import MOAbase
from ._algorithm_init import _SPEA2_init
from ..population.population import Population
from ..utils.hypervolume import hypervolume
from ..utils.fast_non_dominated_sort import fast_non_dominated_sort


def fitness_assignment(solutions: list):
    """
    SPEA2 fitness assignment based on:

        "SPEA2: Improving the Strength Pareto Evolutionary Algorithm for Multiobjective Optimization January 2001
         Conference: Evolutionary Methods for Design, Optimization and Control with Applications to Industrial
         Problems. Proceedings of the EUROGEN'2001. Athens. Greece, September 19-21. Eckart Zitzler, Marco Laumanns
         and Lothar Thiele"

    Parameters
    ------------
    :param solutions: list
        List of possible solutions.
    """
    for solution in solutions:
        strength_value = len(solution.dominated_set)

        for dominated_sol in solution.dominated_set:
            dominated_sol.raw_fitness += strength_value


def density_estimation(solutions: list, k: int, distance):
    """
    SPEA2 density estimation based on:

        "SPEA2: Improving the Strength Pareto Evolutionary Algorithm for Multiobjective Optimization January 2001
         Conference: Evolutionary Methods for Design, Optimization and Control with Applications to Industrial
         Problems. Proceedings of the EUROGEN'2001. Athens. Greece, September 19-21. Eckart Zitzler, Marco Laumanns
         and Lothar Thiele"

    Parameters
    ------------
    :param solutions: list
    :param distance: function
        Metric to evaluate the distance between two vectors.
    :param k: int
        Neighbor from which the fitness density of the solution is calculated.
    """
    for solution in solutions:
        distances = [distance(solution.values, other_sol.values) for other_sol in solutions
                     if id(solution) != id(other_sol)]

        solution.distances = sorted(distances)
        solution.select_density_to(k)


def archive_selection(population: Population, archive_length: int):
    """
    SPEA2 archive selection based on:

        "SPEA2: Improving the Strength Pareto Evolutionary Algorithm for Multiobjective Optimization January 2001
         Conference: Evolutionary Methods for Design, Optimization and Control with Applications to Industrial
         Problems. Proceedings of the EUROGEN'2001. Athens. Greece, September 19-21. Eckart Zitzler, Marco Laumanns
         and Lothar Thiele"

    Parameters
    ------------
    :param population: pywin.population.Population
    :param archive_length: int

    Returns
    ---------
    :return: list
        List of solutions that make up the archive.
    """

    def shorter_distance(sol1, sol2, num_neighbors: int, k: int = 0):
        """
        Function that returns true if the first solution has a shorter distance or
        both solutions are the same. Otherwise it will returns false.
        """
        # If they are both the same distance, go to the next neighbor
        if sol1.distances[k] == sol2.distances[k]:
            k += 1

            #  To stop recursion when the two solutions are the same
            if k == num_neighbors:
                return True

            return shorter_distance(sol1, sol2, num_neighbors, k)

        # Solution 1 has a shorter distance
        elif sol1.distances[k] < sol2.distances[k]:
            return True

        else:
            return False

    archive = list()

    not_dominated_front = [individual for individual in population.individuals if individual.fitness.rank == 0]
    front_length = len(not_dominated_front)

    #  Archive length and non-dominated front match
    if front_length == archive_length:
        archive = not_dominated_front

    #  The non-dominated front is not enough to fill the archive
    elif front_length < archive_length:
        # Get dominated solutions
        remaining_sols = [individual for individual in population.individuals if individual.fitness.rank != 0]

        # Sort solutions
        remaining_sols_sorted = sorted(remaining_sols, key=lambda sol: sol.fitness)

        # Fill archive with dominated solutions
        archive = not_dominated_front + remaining_sols_sorted[: (archive_length - front_length)]

    #  The non-dominated front exceeds the length of the archive, apply truncation operator
    #  The individual with the sorthest distance will be removed at each iteration until fill the archive
    else:
        num_neighbors = len(not_dominated_front[0].fitness.distances)

        i = 0
        while len(not_dominated_front) != archive_length:

            minimum_distance = True

            for j in range(len(not_dominated_front)):

                if i == j: continue

                if not shorter_distance(not_dominated_front[i].fitness, not_dominated_front[j].fitness, num_neighbors):
                    minimum_distance = False

            # The shortest will be removed
            if minimum_distance:
                not_dominated_front.pop(i)
            else:
                i += 1

        archive = not_dominated_front

    return archive


def restart_solutions(solutions):
    """
    Method that resets the values of each solution.

    Parameters
    -----------
    :param solutions: list
    """
    for solution in solutions:
        solution.restart()


@total_ordering
class Solution:
    """
    Class that represents a possible solution with the values of each of the objective functions.
    """

    def __init__(self, values: list):
        """

            __init__(values)

        Notes
        -------
        - values: 1d-array -> List of target function values.
        - dominated_set: list(Solution) -> Solutions dominated by the current solution.
        - np: int -> Number of times this solution is dominated.
        - rank: int -> Indicates which front the current solution is on.
        - raw_fitness -> Number of solutions by which the solution is dominated.
        - distances -> Distances from other solutions.
        """
        self.values = values
        self.dominated_set = []
        self.np = 0
        self.fitness = 0
        self.rank = None
        self.raw_fitness = 0
        self.distances = []

    def __str__(self):
        return f"Solution(rank={self.rank} fitness={self.fitness} raw_fitness={self.raw_fitness})"

    def __repr__(self):
        return self.__str__()

    def select_density_to(self, k):
        """
        Return fitness using density estimation to k-th neighbor. Based on

            "SPEA2: Improving the Strength Pareto Evolutionary Algorithm for Multiobjective Optimization January 2001
             Conference: Evolutionary Methods for Design, Optimization and Control with Applications to Industrial
             Problems. Proceedings of the EUROGEN'2001. Athens. Greece, September 19-21. Eckart Zitzler, Marco Laumanns
             and Lothar Thiele"

        """
        self.fitness = self.raw_fitness + (1 / (self.distances[k] + 2))

    def restart(self):
        """
        Reset all values in the solution.
        """
        self.dominated_set = []
        self.np = 0
        self.fitness = 0
        self.rank = None
        self.raw_fitness = 0
        self.distances = []

    def __eq__(self, other):
        """
        Operator ==
        """
        if self.fitness == other.fitness:
            return True

        return False

    def __lt__(self, other):
        """
        Operator <
        """
        if self.fitness < other.fitness:
            return True

        return False


class SPEA2(MOAbase):
    """
    Implementation of the multi-objective SPEA2 algorithm based on:

        "SPEA2: Improving the Strength Pareto Evolutionary Algorithm for Multiobjective Optimization January 2001
         Conference: Evolutionary Methods for Design, Optimization and Control with Applications to Industrial
         Problems. Proceedings of the EUROGEN'2001. Athens. Greece, September 19-21. Eckart Zitzler, Marco Laumanns
         and Lothar Thiele"

    As many target functions as desired can be added from the fitness sub-module. You can also treat the number
    of features as an objective function (to be minimized) by indicating the parameter optimize_features as true.
    To evaluate the evolution of the algorithm, the hypervolume indicator is used. This metric has been implemented
    using the inclusion-exclusion algorithm. It must be taken into account that the larger the population or the
    more objectives have to be maximized, the higher the computational cost will be.


    Parameters
    ------------
    fitness: pywin.fitness.interface.FitnessStrategy / list
        If the parameter optimize_features is True, a unique fitness function of the fitness submodule can be provided
        (also a list with several functions is allowed). If this parameter is selected as False it is necessary to
        provide a list with at least two functions to optimize.

    optimize_features: <optional> bool
        If this parameter is true, the number of targets will be a function to optimize.

    features_function: <optional> function
        User-defined function that should receive a value called "individual" and  another value "tot_feats". This f
        unction should return a single value that will be maximized. By default this function will be:

                f(individual) = 1 - (len(individual.features) / len(all features))

        Important note: Functions cannot be defined as lambda functions as they are not serializable by the pickle
        library and an error will be thrown.

    population: <optional> pywin.population.Population
        If a population is not provided it is necessary to specify the population size using argument
        population_size.

    population_size: <optionally> int
        Only necessary if a population is not provided. By default the algorithm will create a basic
        population (pywin.population.Population).

    archive_length: <optional> int
        Archive size. By default it will be equal to the population size.

    distance: <optional> str
        Distance used to estimate the density of solutions. By default the Euclidean distance will be used.
        Currently available distances: 'euclidean', 'minkowski', 'cosine' and 'canberra'

    selection: <optional> pywin.selection.interface.SelectionStrategy
        Individual selection strategy. By default pywin.selection.TournamentSelection (with two gladiators,
        a single winner and sampling without replacement).

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
    best_features: (Getter) Return a list with the best features in each solution.

    get_current_generation: (Getter) Return the current generation of the algorithm.

    population: (Getter) Return the current population of the algorithm.

    population_fitness: (Getter) Return the fitness of the current generation of the algorithm (This consists of
        instances of Solution).

    best_performance: (Getter) Return a dict with the score and the best performance achieved for each solution.

    distance: (Getter / Setter) Distance used to evaluate the fitness density of the solutions.

    archive_length: (Getter / Setter) Archive size.

    fitness: (Getter / Setter) Algorithm fitness strategy.

    features_function: (Getter / Setter) Function to evaluate the number of features.

    generations: (Getter / Setter) Algorithm number of generations.

    selection: (Getter / Setter) Algorithm selection strategy.

    mutation_rate: (Getter / Setter) Algorithm mutation_rate.

    imputer: (Getter / Setter) Algorithm imputation strategy.

    crossover: (Getter / Setter) Algorithm crossover strategy.

    positive_class: (Getter / Setter) Algorithm selection strategy.

    random_state: (Getter / Setter) Algorithm selection strategy.

    id: (Getter / Setter) Algorithm selection strategy.


    Methods
    ---------
    set_features(features): Allows you to assign the labels of the columns (the names of the predictor variables).
        If this function is not used before using the fit method, the corresponding numerical value will be assigned
        to the position of the predictor variable. It also can be used after training step.

    fit(X, y): Start algorithm execution.

    continue_training(generations): Continue training the algorithm for an extra number of generations.

    predict(X): Method NOT available for NSGA2.

    training_evolution(): Returns the parameters monitored during training.It returns a dictionary with the
        following scheme:
            :key hypervolume:
                Hypervolume indicator in each generation.
            :key num_solutions_front:
                Number of solutions on the Pareto front in each generation.
            :key best_values:
                Best values of each function in each generation.

    get_dataset(): Function that returns all the dataset and class labels.

    set_population(new_individuals): Method that allows to select a new population of solutions.

    save(file_name, dir_name, overwrite): This function allows to save the model into a file.

    load_model(file_name, dir_name): <class method> This function allows to load a model from a file.
    """

    def __init__(self, **kwargs):
        """
            __init__(**kwargs)

        Notes
        ------
        Use help(SPEA2) to view the required parameters.
        """
        # Call superclass
        super().__init__(**kwargs)

        # Check parameters consistency
        kwargs = _SPEA2_init(kwargs)

        self.archive_length = kwargs.get('archive_length', self._population.size)
        self.distance = kwargs['distance']

        #  Archive with best solutions
        self.archive = None

    def __repr__(self):
        return f"SPEA2(population_size={self._population.size} archive_length: {self.archive_length} " \
               f"generations={self.generations} mutation={self.mutation} mutation_rate={self.mutation_rate} " \
               f"selection={self.selection} fitness={self.fitness} optimize_features={self.optimize_features} " \
               f"crossover={self.crossover} imputer={self.imputer} positive_class={self.positive_class} " \
               f"random_rate={self.random_state}"

    def __str__(self):
        return self.__repr__()

    def get_dataset(self):
        """
        Function that returns all the dataset and class labels.

        Returns
        ---------
        :return: 2d-array
            Predictor variables.
        :return 1d-array
            Class labels.
        """
        return self._X, self._y

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
        :return: pywin.algorithms.SPEA2
            Model fitted using the best feature combination.
        """
        # Transformation of class labels
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

        Parameters
        -------------
        :param generations: int
            Extra number of generations.

        Returns
        -----------
        :return: pywin.algorithms.SPEA2
        """
        if not isinstance(generations, int):
            raise TypeError("Parameter generations must be an integer.")

        # Genetic algorithm core
        self._core(generation_start=self._current_generation, generation_end=(self._current_generation + generations))

        # Save last generation
        self._current_generation = self.generations + generations
        self.generations += generations

        return self

    def training_evolution(self):
        """
        Function that returns a dictionary with the data collected throughout the algorithm search.
        You can use the function plot_evolution() from pywin.visualization.Plotter to display it.

        Returns
        ----------
        :return: dict
            Dictionary with the evolution of the hypervolume indicator, number of solutions on the non-dominated
            front and best values for each objective function.
        :return str
            Scores used to evaluate fitness.
        """
        scores = [fitness_func.score for fitness_func in self.fitness]
        if self.optimize_features:
            scores.append("Num. features")

        return self._evolution, scores

    def _evaluate_fitness(self, population: Population):
        """
        Function that evaluates the values of the objective functions for each of the individuals.
        A solution will be assigned to the fitness value of each individual

        Parameters
        ------------
        :param population: pywin.population.Population
            Population of individuals.

        Returns
        ----------
        :return: pywin.population.Population
            Population.
        """
        for n, individual in enumerate(population.individuals):

            # Dataset extraction using individual features
            X_data = self._create_dataset(individual, self._X)

            # Get scores for each fitness strategy (each objective)
            scores = [fitness_func.eval_fitness(X=X_data, y=self._y, num_feats=len(population.features))
                      for fitness_func in self.fitness]

            # If the number of features is an objective
            if self.optimize_features:
                scores.append(self.features_function(individual=individual,
                                                     total_feats=len(self._population.features)))

            # Create a solution
            individual.fitness = Solution(scores)

        return population

    def _annotate(self, generation: int):
        """
        Record the values to be monitored in the algorithm for each generation

        Parameters
        ------------
        :param generation: int
        """
        # Get pareto front
        # Get pareto front
        pareto_front_scores = np.array(
            [individual.fitness.values for individual in self._population.individuals
             if individual.fitness.rank == 0]
        )

        # Calculate hypervolume
        self._evolution['hypervolume'][generation + 1] = hypervolume(pareto_front=pareto_front_scores)

        # Get number of solutions on the Pareto front
        self._evolution['num_solutions_front'][generation + 1] = len(pareto_front_scores)

        # Get best performance achieved for each objective
        self._evolution['best_values'][generation + 1] = np.max(pareto_front_scores, axis=0)

    def _core(self, generation_start: int, generation_end: int):
        """
        Method that carries out the algorithm logic.

            1. Initialization: Generate an initial population (Pt) and create an empty archive (Pt_a).
            2. Fitness assignment: Calculate fitness values of individuals in Pt and Pt_a (see fitness_assignment()).
            3. Environmental selection: Copy all non-dominated individuals in Pt and Pt_a to Pt+1. If size of
                Pt+1 exceeds archive_length then reduce Pt+1 by means of the truncation operator, otherwise
                if size  of Pt+1 is less than archive_length then fill Pt+1 with dominated individuals in Pt
                and Pt_a (see archive_selection()).
            4. Apply selection, cross-over and mutation to Pt+1.

        Adapted from:

            "SPEA2: Improving the Strength Pareto Evolutionary Algorithm for Multiobjective Optimization January 2001
             Conference: Evolutionary Methods for Design, Optimization and Control with Applications to Industrial
             Problems. Proceedings of the EUROGEN'2001. Athens. Greece, September 19-21. Eckart Zitzler, Marco Laumanns
             and Lothar Thiele"

        Parameters
        ------------
        :param generation_start: int
            Generation from which the algorithm should start
        :param generation_end: int
            Generation until which the algorithm must arrive.
        """
        info = "(SPEA2) Generations (form %d to %d)" % (generation_start, generation_end)

        for generation in tqdm(range(generation_start, generation_end), desc=info):

            # Evaluate fitness for each objective
            self._population = self._evaluate_fitness(population=self._population)

            #  if the first interaction has already occurred, put the population and the archive together
            if self.archive is not None:
                restart_solutions(self.archive.fitness)
                pt = self._population.merge_population(self._population, self.archive)
            else:
                pt = self._population

            #  Get PT fitness
            pt_fitness = pt.fitness

            # Sort Pareto front
            fast_non_dominated_sort(pt_fitness)

            # Assign fitness
            fitness_assignment(pt_fitness)

            # Calculate the density of the solutions
            k = int(sqrt(self._population.size + self.archive_length))
            density_estimation(solutions=pt_fitness, k=k, distance=self.distance)

            #  Select solutions to the archive
            archive = archive_selection(archive_length=self.archive_length,
                                        population=pt)

            self.archive = self._population.create_new_population(size=self.archive_length,
                                                                  individuals=archive)

            # Assign archive individuals to population
            self._population.set_new_individuals(archive)

            # Annotate algorithm performance
            self._annotate(generation=generation)

            # Apply selection
            offspring = self.selection.select(population=self._population,
                                              new_pop_length=self._population.size)

            # Apply cross-over
            offspring = self.crossover.cross_population(offspring)

            # Introduces mutations
            if self.mutation is not None:
                offspring = self.mutation.mutate(population=offspring, mutation_rate=self.mutation_rate)

            self._population = offspring

        # Last generation
        population = self._evaluate_fitness(population=self._population)
        pt = self._population.merge_population(population, self.archive)

        #  Get PT fitness
        pt_fitness = pt.fitness

        # Sort Pareto front
        fast_non_dominated_sort(pt_fitness)

        # Assign fitness
        fitness_assignment(pt_fitness)

        # Calculate the density of the solutions
        k = int(sqrt(self._population.size + self.archive_length))
        density_estimation(solutions=pt_fitness, k=k, distance=self.distance)

        #  Select solutions from the file to create the new population
        archive = archive_selection(archive_length=self.archive_length, population=pt)

        self._population = self._population.create_new_population(size=self.archive_length,
                                                                  individuals=archive)
