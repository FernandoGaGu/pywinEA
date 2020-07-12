# Module that provides basic tools to examine the evolution of the algorithms and evaluate the results obtained.
#
# Author: Fernando García <ga.gu.fernando@gmail.com>
#
# External dependencies
import numpy as np
from math import inf as infinite
import warnings
# Module dependencies
from ..interface.wrapper import WrapperBase
from ..wrapper.wrapper import IslandModel
from ..algorithm.basic import GA
from ..interface.algorithm import MOAbase


class Plotter:
    """
    Class that allows to represent the genetic algorithm evolution and performance.

    Methods
    -----------
    plot_evolution(): Represents information of the number of features and performance of the model in
        each generation.
    """

    @classmethod
    def plot_evolution(cls, *algorithms, **kwargs):
        """
        Function that performs the graphic representation of the training process of the algorithm(s).
        Based on the type of algorithm received as a parameter it will make a different representation.
        """
        # Get a list of estimators
        mono_objective_algorithms, multi_objective_algorithms = [], []

        for algorithm in algorithms:
            #  To extract the algorithms in wrapper.
            if isinstance(algorithm, WrapperBase):
                
                for wrapper_alg in algorithm.algorithms:
                    if isinstance(wrapper_alg, GA):
                        mono_objective_algorithms.append(wrapper_alg)
                    if isinstance(wrapper_alg, MOAbase):
                        multi_objective_algorithms.append(wrapper_alg)
            else:
                if isinstance(algorithm, GA):
                    mono_objective_algorithms.append(algorithm)
                if isinstance(algorithm, MOAbase):
                    multi_objective_algorithms.append(algorithm)

        # Call plotting functions
        if len(mono_objective_algorithms) != 0:
            Plotter._plot_mono_objective(mono_objective_algorithms, **kwargs)

        if len(multi_objective_algorithms) != 0:
            Plotter._plot_multi_objective(multi_objective_algorithms, **kwargs)

    @classmethod
    def _plot_mono_objective(cls, algorithms, **kwargs):
        """
        Graphic representation of:

            - The evolution of the total fitness in population.
            - The evolution of the best individual.
            - The number of features in population.
            - The number of features in best individual.
            - The performance measure evolution.
        """
        import matplotlib.pyplot as plt

        scores, generations, y_tot_fit, y_best_fit = [], [], [], []
        y_mean_feat, y_std_feat, y_best_feat = [], [], []
        y_best_performance, y_mean_performance, y_std_performance = [], [], []
        y_best_idx_max, x_best_idx_max, y_tot_idx_max, x_tot_idx_max = [], [], [], []

        # Get parameters
        for n, algorithm in enumerate(algorithms):
            args, score = algorithm.training_evolution()
            generations.append(list(args['total_fitness'].keys()))
            y_tot_fit.append(list(args['total_fitness'].values()))
            y_best_fit.append(list(args['best_fitness'].values()))
            y_mean_feat.append(list(args['mean_num_features'].values()))
            y_std_feat.append(list(args['std_num_features'].values()))
            y_best_feat.append(list(args['best_num_features'].values()))
            y_best_performance.append(list(args['best_%s' % score].values()))
            y_mean_performance.append(list(args['mean_%s' % score].values()))
            y_std_performance.append(list(args['std_%s' % score].values()))
            scores.append(score)

        # Get maximum values
        for n in range(len(generations)):
            y_best_idx_max.append(y_best_fit[n].index(max(y_best_fit[n])))
            x_best_idx_max.append(generations[n][y_best_idx_max[n]])
            y_tot_idx_max.append(y_tot_fit[n].index(max(y_tot_fit[n])))
            x_tot_idx_max.append(generations[n][y_tot_idx_max[n]])

        num_subplots = 6
        fig_length = 40
        if kwargs.get("simplified", False):
            num_subplots = 2
            fig_length = kwargs.get('fig_length', 8) 

        # Plot estimator(s) evolution
        fig, ax = plt.subplots(num_subplots, figsize=(12, fig_length))
        fig.subplots_adjust(wspace=kwargs.get('wspace', 0.2), hspace=0.5)
        cmap = plt.cm.get_cmap(kwargs.get('cmap', 'viridis'), len(generations)+1)

        for n, algorithm in enumerate(algorithms):

            if kwargs.get("simplified", False):
                # Subplot 1 -> Best fitness in population per generation
                ax[0].plot(generations[n], y_best_fit[n], color=cmap(n),
                           label="Best fitness (%s)" % algorithm.id)
                ax[0].annotate('Maximum: %.3f' % max(y_best_fit[n]), xy=(x_best_idx_max[n], max(y_best_fit[n])),
                               xytext=(x_best_idx_max[n], max(y_best_fit[n]) * 1.025),
                               arrowprops=dict(facecolor='black', shrink=0.05))
                # Subplot 3 -> Number of features evolution in best individual per generation
                ax[1].plot(generations[n], y_best_feat[n], color=cmap(n),
                           label="Num. feats best indv. (%s)" % algorithm.id)                
            else:
                # Subplot 0 -> Population evolution
                ax[0].plot(generations[n], y_tot_fit[n], color=cmap(n),
                           label="Population fitness (%s)" % algorithm.id)
                ax[0].annotate('Maximum: %.3f' % max(y_tot_fit[n]), xy=(x_tot_idx_max[n], max(y_tot_fit[n])),
                               xytext=(x_tot_idx_max[n], max(y_tot_fit[n]) * 1.0025),
                               arrowprops=dict(facecolor='black', shrink=0.05))

                # Subplot 1 -> Best fitness in population per generation
                ax[1].plot(generations[n], y_best_fit[n], color=cmap(n),
                           label="Best fitness (%s)" % algorithm.id)
                ax[1].annotate('Maximum: %.3f' % max(y_best_fit[n]), xy=(x_best_idx_max[n], max(y_best_fit[n])),
                               xytext=(x_best_idx_max[n], max(y_best_fit[n]) * 1.025),
                               arrowprops=dict(facecolor='black', shrink=0.05))

                # Subplot 2 -> Number of features evolution in population
                ax[2].plot(generations[n], y_mean_feat[n], color=cmap(n),
                           label="Num. feats in population (%s)" % algorithm.id)
                ax[2].fill_between(generations[n],
                                   np.array(y_mean_feat[n]) - (np.array(y_std_feat[n])),
                                   np.array(y_mean_feat[n]) + (np.array(y_std_feat[n])),
                                   alpha=0.1, color=cmap(n))

                # Subplot 3 -> Number of features evolution in best individual per generation
                ax[3].plot(generations[n], y_best_feat[n], color=cmap(n),
                           label="Num. feats best indv. (%s)" % algorithm.id)

                # Subplot 4 -> Best score in population per generation
                ax[4].plot(generations[n], y_best_performance[n],
                           label="Best %s (%s)" % (scores[n], algorithm.id), color=cmap(n))

                # Subplot 5 -> Score evolution in population
                ax[5].plot(generations[n], y_mean_performance[n],
                           label="Average %s (%s)" % (scores[n], algorithm.id), color=cmap(n))
                ax[5].fill_between(generations[n],
                                   np.array(y_mean_performance[n]) - (np.array(y_std_performance[n])),
                                   np.array(y_mean_performance[n]) + (np.array(y_std_performance[n])),
                                   alpha=0.1, color=cmap(n))

        # Get limits
        y_tot_fit_min, y_tot_fit_max, y_best_fit_min, y_best_fit_max = infinite, 0, infinite, 0
        for tot_fit, best_fit in zip(y_tot_fit, y_best_fit):
            if min(tot_fit) < y_tot_fit_min:
                y_tot_fit_min = min(tot_fit)
            if max(tot_fit) > y_tot_fit_max:
                y_tot_fit_max = max(tot_fit)
            if min(best_fit) < y_best_fit_min:
                y_best_fit_min = min(best_fit)
            if max(best_fit) > y_best_fit_max:
                y_best_fit_max = max(best_fit)

        title_params =  {
            'family': kwargs.get('title_font', 'DejaVu Sans'), 
            'weight': 'normal', 
            'size': kwargs.get('title_size', 20),  
            'verticalalignment': 'baseline'
            }        
        if kwargs.get("simplified", False):
            # Subplot 1 -> Best fitness in population per generation
            ax[0].set_ylim(bottom=y_best_fit_min, top=y_best_fit_max * 1.2)
            ax[0].set_title("Best fitness in population / generation", fontdict=title_params, pad=20)
            # Subplot 2 -> Number of features evolution in population
            ax[1].set_title("Mean number of features in population", fontdict=title_params, pad=20)
            ax[1].legend()            
        else:
            # Subplot 0 -> Population evolution
            ax[0].set_ylim(bottom=y_tot_fit_min, top=y_tot_fit_max * 1.01)
            ax[0].set_title("Total fitness in population", fontdict=title_params, pad=20)
            # Subplot 1 -> Best fitness in population per generation
            ax[1].set_ylim(bottom=y_best_fit_min, top=y_best_fit_max * 1.2)
            ax[1].set_title("Best fitness in population / generation", fontdict=title_params, pad=20)
            # Subplot 2 -> Number of features evolution in population
            ax[2].set_title("Mean number of features in population", fontdict=title_params, pad=20)
            ax[2].legend()
            # Subplot 3 -> Number of features evolution in best individual per generation
            ax[3].set_title("Number of features in best individual", fontdict=title_params, pad=20)
            ax[3].legend()
            # Subplot 4 -> Best score in population per generation
            ax[4].set_title("Best %s per generation" % scores[0], fontdict=title_params, pad=20)
            ax[4].set_ylim(bottom=0.5, top=1.0)
            ax[4].legend()
            # Subplot 5 -> Score evolution in population
            ax[5].set_title("Average %s in population" % scores[0], fontdict=title_params, pad=20)
            ax[5].set_ylim(bottom=0.5, top=1.0)
            ax[5].legend()

        #  Remove borders and add grid to all graphics
        for subplot in ax:
            subplot.grid(True, alpha=0.3)
            subplot.spines['top'].set_visible(False)
            subplot.spines['right'].set_visible(False)
            handles, labels = subplot.get_legend_handles_labels()
            subplot.legend(
                handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), 
                fancybox=True, prop={'size': kwargs.get("legend_size", 12)})

        plt.show()

    @classmethod
    def _plot_multi_objective(cls, algorithms, **kwargs):
        """
        Graphic representation of:

            - Hypervolume indicator.
            - Number of solutions on the non-dominated Pareto front.
            - Best values achieved in each objective function.
        """
        import matplotlib.pyplot as plt

        generations, hypervolumes, num_solutions, best_values = [], [], [], []

        for algorithm in algorithms:
            algorithm_stats, scores = algorithm.training_evolution()
            generations.append(list(algorithm_stats['hypervolume'].keys()))
            hypervolumes.append(list(algorithm_stats['hypervolume'].values()))
            num_solutions.append(list(algorithm_stats['num_solutions_front'].values()))
            best_values.append(list(algorithm_stats['best_values'].values()))

        # Sinmplified version plot only hypervolume convergence and Pareto front
        num_subplots = 4
        fig_length = 38

        if kwargs.get('simplified', False):
            num_subplots = 2
            fig_length = kwargs.get('fig_length', 12) 

        # Plot estimator(s) evolution
        fig, ax = plt.subplots(num_subplots, figsize=(kwargs.get('fig_width', 12), fig_length))
        fig.subplots_adjust(wspace=kwargs.get('wspace', 0.2), hspace=0.5)
        linestyles = ['-', ':', '-.', '--']
        cmap = plt.cm.get_cmap(kwargs.get('cmap', 'viridis'), len(algorithms)+1)

        for n, algorithm in enumerate(algorithms):
            # Subplot 0 -> Hypervolume
            ax[0].plot(generations[n], hypervolumes[n], color=cmap(n),
                       label="Algorithm: (%s)" % algorithm.id)


            #  Get scores for current algorithm
            pareto_front = np.array([individual.fitness.values for individual in algorithm.population.individuals
                                      if individual.fitness.rank == 0])
            #  Sort scores based on one metric
            idx = np.argsort(pareto_front[:, 0])
            pareto_front = pareto_front[idx, :]
        
            ax[int(num_subplots/2)].scatter(x=pareto_front[:, 0], y=pareto_front[:, 1],
                          color=cmap(n), alpha=0.5, label="Algorithm: (%s)" % algorithm.id)

            for score_idx in range(pareto_front.shape[0]):
                ax[int(num_subplots/2)].plot([0, pareto_front[:, 0][score_idx]],
                           [pareto_front[:, 1][score_idx], pareto_front[:, 1][score_idx]],
                           color=cmap(n), alpha=0.1)

                ax[int(num_subplots/2)].plot([pareto_front[:, 0][score_idx], pareto_front[:, 0][score_idx]],
                           [0, pareto_front[:, 1][score_idx]],
                           color=cmap(n), alpha=0.1)

                ax[int(num_subplots/2)].fill_between([0, pareto_front[:, 0][score_idx]],
                                   [pareto_front[:, 1][score_idx], pareto_front[:, 1][score_idx]],
                                   color=cmap(n), alpha=kwargs.get('pf_alpha', 0.01))

            if not kwargs.get('simplified', False):
                # Subplot 1 -> Number of solutions in Pareto front per generation
                ax[1].plot(generations[n], num_solutions[n], color=cmap(n),
                           label="Algorithm: (%s)" % algorithm.id)

                                # Subplot 3 -> Best values for each objective
                for i, score in enumerate(scores):
                    ax[3].plot(generations[n], [value[i] for value in best_values[n]], color=cmap(n),
                               linestyle=linestyles[i], label="Score: (%s) (%s)" % (score, algorithm.id))


        # Get limits
        min_hyper, max_hyper, max_num_sol = 1, 0, 0

        for volumes, num_sols in zip(hypervolumes, num_solutions):
            if min(volumes) < min_hyper: min_hyper = min(volumes)
            if max(volumes) > max_hyper: max_hyper = max(volumes)
            if max(num_sols) > max_num_sol: max_num_sol = max(num_sols)

        title_params =  {
            'family': kwargs.get('title_font', 'DejaVu Sans'), 
            'weight': 'normal', 
            'size': kwargs.get('title_size', 20),  
            'verticalalignment': 'baseline'
            }

        # Subplot 0 ->  Hypervolume
        ax[0].set_title("Hypervolume convergence", fontdict=title_params, pad=20)
        ax[0].set_ylim(bottom=min_hyper, top=max_hyper + 0.001)        
        ax[0].set_xlabel("Generations", size=kwargs.get('x_label_size', 15))
        ax[0].set_ylabel("Hypervolume", size=kwargs.get('y_label_size', 15))

        ax[int(num_subplots/2)].set_title("Pareto front", fontdict=title_params)
        ax[int(num_subplots/2)].set_ylim(bottom=kwargs.get('pf_bottom_y_lim', 0), top=kwargs.get('pf_top_y_lim', 1))
        ax[int(num_subplots/2)].set_xlim(left=kwargs.get('pf_bottom_x_lim', 0), right=kwargs.get('pf_top_x_lim', 1))
        ax[int(num_subplots/2)].set_xlabel(
            "F(num_features)" if scores[0] == 'Num. features' else scores[0].capitalize(), 
            size=kwargs.get('x_label_size', 15))

        ax[int(num_subplots/2)].set_ylabel(
            "F(num_features)" if scores[1] == 'Num. features' else scores[1].capitalize(), 
            size=kwargs.get('y_label_size', 15))
        
        
        if not kwargs.get('simplified', False):
            # Subplot 1 -> Best fitness in population per generation
            ax[1].set_title("Num. solutions in Pareto front", fontdict=title_params, pad=20)
            ax[1].set_ylim(bottom=0, top=max_num_sol + 1)
            ax[1].set_xlabel("Generations", size=kwargs.get('x_label_size', 15))
            ax[1].set_ylabel("Numnber of solutions", size=kwargs.get('y_label_size', 15))

            # Subplot 3 -> Best value for each objective
            ax[3].set_title("Best scores per generation", fontdict=title_params, pad=20)
            ax[3].set_ylim(bottom=0.7, top=1)        
            ax[3].set_xlabel("Generations", size=kwargs.get('x_label_size', 15))
            ax[3].set_ylabel("Score value", size=kwargs.get('y_label_size', 15))
        

        for subplot in ax:
            subplot.grid(True, alpha=0.3)
            subplot.spines['top'].set_visible(False)
            subplot.spines['right'].set_visible(False)
            handles, labels = subplot.get_legend_handles_labels()
            subplot.legend(
                handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), 
                fancybox=True, prop={'size': kwargs.get("legend_size", 12)})

        plt.show()
