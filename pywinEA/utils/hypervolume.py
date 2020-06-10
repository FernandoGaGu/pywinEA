#  Module that groups the functions used by multi-objective genetic algorithms.
#
# Author: Fernando García <ga.gu.fernando@gmail.com>
#
import numpy as np
from functools import reduce


def hypervolume(pareto_front: np.ndarray):
    """
    Function that calculate hypervolume based on inclusion-exclusion algorithm.

    Parameters
    -----------
    :param pareto_front: list
        List of solutions in Pareto front

    Returns
    ---------
    :return float
        Hypervolume covered by Pareto front
    """

    def vol(solution):
        """
        Function that calculates the volume covered by a solution.

        Parameters
        -----------
        :param solution: Solution

        Returns
        ---------
        :return float
        """
        return reduce(lambda coor1, coor2: coor1 * coor2, solution)

    def vol_intersec(*solutions):
        """
        Function that calculates the volume covered by the intersection between two solutions.

        Parameters
        ------------
        :param solutions: Solution
            One or more solutions.

        Returns
        ---------
        :return float
        """
        return reduce(lambda coor1, coor2: coor1 * coor2, np.min(solutions, axis=0))

    # Sort solution based on one target function
    pareto_front = np.sort(pareto_front, axis=0)

    # Calculate intersections between each pair of adjacent solutions
    intersec = [vol_intersec(pareto_front[n], pareto_front[n + 1]) for n in range(len(pareto_front) - 1)]

    # Calculate the volume of each solution and subtract the volumes of the intersections
    return sum(list(map(vol, pareto_front))) - sum(intersec)
