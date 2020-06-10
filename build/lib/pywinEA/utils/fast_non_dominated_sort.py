#  Module that groups the functions used by multi-objective genetic algorithms.
#
# Author: Fernando García <ga.gu.fernando@gmail.com>
#
from collections import defaultdict


def dominance(solution_1, solution_2):
    """
    Function that analyze solutions dominance.

    Parameters
    -----------
    :param solution_1: Solution
    :param solution_2: Solution

    Returns
    ---------
    :return int
        If solution_1 dominates solution_2 -> return 1
    :return -1
        If solution_2 dominates solution_1 -> return -1
    :return 0
        If neither solution dominates the other -> return 0
    """
    dominance_1, dominance_2 = False, False

    for i, value in enumerate(solution_1.values):

        if value > solution_2.values[i]:
            #  Solution 1 at least greater in one value
            dominance_1 = True

        elif value < solution_2.values[i]:
            # Solution 2 at least greater in one value
            dominance_2 = True

    # Solution 1 dominates solution 2
    if dominance_1 and not dominance_2:
        return 1

    # Solution 2 dominates solution 1
    if not dominance_1 and dominance_2:
        return -1

    return 0


def fast_non_dominated_sort(solutions: list):
    """
    Apply fast non dominated sort.

    Parameters
    -----------
    :param solutions: list
        List of Solution instances.
    """

    # Initialize an empty front
    front = defaultdict(list)

    for solution in solutions:
        for other_solution in solutions:

            # if both solutions are the same pass to next one
            if id(solution) == id(other_solution): continue

            # Analyze dominance between solutions
            dominates = dominance(solution, other_solution)

            if dominates > 0:
                # First solution dominates the other solution
                solution.dominated_set.append(other_solution)

            elif dominates < 0:
                # The other solution dominates the first solution
                solution.np += 1

        # Save only first front
        if solution.np == 0:
            front[solution.np].append(solution)

    # Get other fronts
    i = 0
    while len(front[i]) != 0:
        for solution in front[i]:
            for dominated_solution in solution.dominated_set:

                # Update front
                dominated_solution.np -= 1

                # Check if solution is in the next front
                if dominated_solution.np == 0:
                    # Add to next front
                    front[i + 1].append(dominated_solution)
        i += 1

    # Select rank attribute of each solution
    for f_rank, f_solutions in front.items():
        for solution in f_solutions:
            solution.rank = f_rank
