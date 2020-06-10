# Module that contains all the genetic algorithms implemented in the pywin module.
#
# Author: Fernando García <ga.gu.fernando@gmail.com>
#
# External dependencies
import numpy as np
from math import inf as infinite
from functools import total_ordering
from tqdm import tqdm

#  Module dependencies
from ..interface.algorithm import MOAbase
from ..population.population import Population
from ..utils.hypervolume import hypervolume
from ..utils.fast_non_dominated_sort import fast_non_dominated_sort


def calculate_crowding(solutions: list):
    """
    Method to calculate crowding for all solutions using _crowding_distance() method for each solution.

    Parameters
    -----------
    :param solutions: list
        List with all solutions.
    """

    def crowding_distance(all_solutions: list, idx: int, measure_idx: int):
        """
        Function that calculates the crowding distance (cuboid) for a certain solution.

        Parameters
        ------------
        :param all_solutions: list
            All solutions.
        :param idx: int
            Index indicating the objective solution for which the crowding will be calculated.
        :param measure_idx: int
            Indicates the index at which the scores of a certain objective function are found.
        """
        # Get target function values
        measured_values = [solution.values[measure_idx] for solution in all_solutions]
        f_max = max(measured_values)
        f_min = min(measured_values)

        # If all the solutions are the same crowding is 0
        if f_max == f_min:
            return 0

        # Calculate crowding distance
        distance = (measured_values[idx + 1] - measured_values[idx - 1]) / \
                   (max(measured_values) - min(measured_values))

        return distance

    # Get the number of target functions
    num_objectives = len(solutions[0].values)

    for measure in range(num_objectives):

        # Sort solutions based on measure value (ascending)
        solutions = sorted(solutions, key=lambda solution: solution.values[measure])

        # Select limits to infinite
        solutions[0].crowding_distance, solutions[len(solutions) - 1].crowding_distance = infinite, infinite

        # Calculate crowding distance for target function
        for i in range(1, len(solutions) - 1):
            solutions[i].crowding_distance += crowding_distance(all_solutions=solutions,
                                                                idx=i, measure_idx=measure)


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
        - crowding_distance: float -> Crowding distance.
        """
        self.values = values
        self.dominated_set = []
        self.np = 0
        self.rank = None
        self.crowding_distance = 0

    def __str__(self):
        return f"Solution(values={self.values} rank={self.rank} crowding={self.crowding_distance})"

    def __repr__(self):
        return self.__str__()

    def restart(self):
        """
        Reset all values in the solution.
        """
        self.dominated_set = []
        self.np = 0
        self.rank = None
        self.crowding_distance = 0

    def _crowded_comparision(self, other):
        """
        Comparison operator between two solutions. Based on:

            K. Deb, A. Pratap, S. Agarwal and T. Meyarivan, "A fast and elitist multiobjective genetic algorithm:
            NSGA-II," in IEEE Transactions on Evolutionary Computation, vol. 6, no. 2, pp. 182-197, April 2002.

        Crowded comparision operator:
            If we have two solutions with a different Pareto ranking, we choose the one with the lowest value.
            If they have the same ranking we take the one with the highest crowding (the least covered solution).
        """
        # Current solution dominates
        if (self.rank < other.rank) or ((self.rank == other.rank) and
                                        (self.crowding_distance > other.crowding_distance)):
            return 1
        # Both solutions are equal
        elif (self.rank == other.rank) and (self.crowding_distance == other.crowding_distance):
            return 0

        return -1

    def __eq__(self, other):
        """
        Operator ==
        """
        if self._crowded_comparision(other) == 0:
            return True

        return False

    def __lt__(self, other):
        """
        Operator <
        """
        if self._crowded_comparision(other) == -1:
            return True

        return False


class NSGA2(MOAbase):
    """
    Implementation of the multi-objective NSGAII algorithm based on:

        K. Deb, A. Pratap, S. Agarwal and T. Meyarivan, "A fast and elitist multiobjective genetic algorithm:
        NSGA-II," in IEEE Transactions on Evolutionary Computation, vol. 6, no. 2, pp. 182-197, April 2002.

    As many target functions as desired can be added from the fitness sub-module. You can also treat the number
    of features as an objective function (to be minimized) by indicating the parameter optimize_features as true.
    To evaluate the evolution of the algorithm, the hypervolume indicator is used. This metric has been implemented
    using the inclusion-exclusion algorithm. It must be taken into account that the larger the population or the
    more objectives have to be maximized, the higher the computational cost will be.


    Parameters
    ------------
    fitness: pywin.interface.FitnessStrategy / pywin.fitness / list
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
        Use help(NSGA2) to view the required parameters.
        """
        # Call superclass
        super().__init__(**kwargs)

    def __repr__(self):
        return f"NSGAII(population_size={self._population.size} generations={self.generations} " \
               f"mutation={self.mutation} mutation_rate={self.mutation_rate} selection={self.selection} " \
               f"fitness={self.fitness} optimize_features={self.optimize_features} crossover={self.crossover} " \
               f"imputer={self.imputer} positive_class={self.positive_class} random_rate={self.random_state}"

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
        :return: pywin.algorithms.NSGA2
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
        :return: NSGA2
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

    def _set_new_population(self, parents_offspring: Population):
        """
        Select the new population using the crowding operator until fill the population size. Individuals with a
        lower rank value (better solutions) will be selected. For individuals with the same rank value, those
        with a higher crowding distance will be selected.

        Parameters
        -----------
        :param parents_offspring: pywin.population.Population
        """
        # Get fitness
        parents_offspring_fitness = parents_offspring.fitness

        # Get individual indices sorted by fitness
        indices = np.argsort(parents_offspring_fitness)[:(self._population.size - 1):-1]

        # Get best individuals and their fitness
        best_individuals = [parents_offspring.individuals[idx] for idx in indices]

        # Assign best individuals to population
        self._population.set_new_individuals(best_individuals)

    def _core(self, generation_start: int, generation_end: int):
        """
        Main logic of the algorithm. First, evaluate the individual fitness (creating solutions). Second, applies
        the "fast non dominated sort" algorithm to assign the solutions to different non-dominated fronts and
        calculates the crowding distance. Then apply the basic operations (selection, crossover and mutation)
        to generate offspring. And finally evaluate the fitness of the offspring, merge the offspring with the
        parents and fill the next generation with the best solutions.

        Parameters
        ------------
        :param generation_start: int
            Generation from which the algorithm should start
        :param generation_end: int
            Generation until which the algorithm must arrive.
        """

        # Evaluate fitness for each objective
        self._population = self._evaluate_fitness(population=self._population)

        #  Get population fitness
        population_fitness = self._population.fitness

        # Sort Pareto front
        fast_non_dominated_sort(population_fitness)

        # Calculate crowding
        calculate_crowding(population_fitness)

        info = "(NSGAII) Generations (form %d to %d)" % (generation_start, generation_end)
        for generation in tqdm(range(generation_start, generation_end), desc=info):

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

            # Evaluate offspring
            offspring = self._evaluate_fitness(population=offspring)

            # Restart parent solutions
            restart_solutions(self._population.fitness)

            # Merge parents and offspring
            parents_offspring = self._population.merge_population(self._population, offspring)

            #  Get parents_offspring fitness
            parents_offspring_fitness = parents_offspring.fitness

            # Sort Pareto front
            fast_non_dominated_sort(parents_offspring_fitness)

            # Calculate crowding
            calculate_crowding(parents_offspring_fitness)

            # Set new population
            self._set_new_population(parents_offspring)
