import mlrose_hiive
import numpy as np
import pandas as pd
# import logging
# import networkx as nx
# import string
# from ast import literal_eval
# import chess
# from sklearn.datasets import load_iris
# from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from mlrose_hiive import QueensGenerator, MaxKColorGenerator, TSPGenerator, FlipFlopGenerator, KnapsackGenerator,ContinuousPeaksGenerator
from mlrose_hiive import SARunner, GARunner, NNGSRunner, MIMICRunner, RHCRunner
# from mlrose_hiive.runners import NNGSRunner
# # import itertools as it
# import seaborn as sns
import matplotlib.pyplot as plt
#
# from matplotlib.ticker import StrMethodFormatter
# from sklearn import preprocessing
# from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
# # from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, KFold
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neural_network import MLPClassifier
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn import svm
# # from sklearn.neighbors import NearestNeighbors
# from sklearn.neighbors import KNeighborsClassifier
#
# from sklearn.model_selection import cross_validate
# from sklearn.model_selection import learning_curve
# from sklearn.model_selection import validation_curve
#
from sklearn.metrics import accuracy_score
# from sklearn.metrics import recall_score
# from sklearn.metrics import log_loss
# from sklearn.metrics import confusion_matrix
#
# from warnings import simplefilter
# from sklearn.exceptions import ConvergenceWarning
# import time as tm
# # from sklearn import metrics
# from os import path


# https://pypi.org/project/mlrose-hiive/
# pip install mlrose-hiive                        <<<<<<<<<<<<< <<<<<<< <<< <<<<<<
# pip install mlrose-hiive                        <<<<<<<<<<<<< <<<<<<< <<< <<<<<<
# pip install mlrose-hiive                        <<<<<<<<<<<<< <<<<<<< <<< <<<<<<

# https://mlrose.readthedocs.io/en/stable/source/algorithms.html
# https://github.com/gkhayes/mlrose/blob/master/tutorial_examples.ipynb
# https://github.com/hiive/mlrose/blob/master/problem_examples.ipynb


#  #/// hill_climb(problem, max_iters=inf, restarts=0, init_state=None, curve=False, random_state=None)
# random_hill_climb(problem, max_attempts=10, max_iters=inf, restarts=0, init_state=None, curve=False, random_state=None)
## simulated_annealing(problem, schedule=<mlrose.decay.GeomDecay object>, max_attempts=10, max_iters=inf, init_state=None, curve=False, random_state=None)
## genetic_alg(problem, pop_size=200, mutation_prob=0.1, max_attempts=10, max_iters=inf, curve=False, random_state=None)
## mimic(problem, pop_size=200, keep_pct=0.2, max_attempts=10, max_iters=inf, curve=False, random_state=None, fast_mimic=False)


def TSP():
    # Generate a new TSP problem using a fixed seed.
    problem = TSPGenerator().generate(seed=631298, number_of_cities=20)

    ###################################################
    # c. Gen Algo *** Highlight advantages of
    ###################################################
    ga = GARunner(problem=problem,
                  experiment_name='tsp_ga',
                  # output_directory=None,
                  seed=631298, #seed=123456,
                  iteration_list=4 ** np.arange(7),
                  max_attempts=25,
                  population_sizes=[100, 150], #[20, 30],
                  mutation_rates=[0.1, 0.3, 0.5])   #mutation_prob

    # the two data frames will contain the results
    df_run_stats, df_run_curves = ga.run()

    best_fitness = df_run_curves['Fitness'].min()
    best_runs = df_run_curves[df_run_curves['Fitness'] == best_fitness]

    minimum_evaluations = best_runs['FEvals'].min()
    best_curve_run = best_runs[best_runs['FEvals'] == minimum_evaluations]

    best_mr = best_curve_run['Mutation Rate'].iloc()[0]
    best_pop_size = best_curve_run['Population Size'].iloc()[0]
    print('Best Mutation Rate: ',best_mr, 'best Population Size: ',best_pop_size)


    # df_run_curves_mean = df_run_curves.groupby('Iteration').mean().reset_index()
    df_run_stats_mean = df_run_stats.groupby('Iteration').mean().reset_index()
    df_run_stats_all = df_run_stats_mean # Collect results


    df_fitness = df_run_stats_mean[['Iteration', 'Fitness']] # df1 = df.iloc[:, 0:1]
    df_time = df_run_stats_mean[['Iteration', 'Time']]
    df_fevals = df_run_stats_mean[['Iteration', 'FEvals']]

    df_fitness.rename(columns={'Fitness': 'GA'}, inplace=True)
    df_time.rename(columns={'Time': 'GA'}, inplace=True)
    df_fevals.rename(columns={'FEvals': 'GA'}, inplace=True)


    # scoring_metric = 'Fitness'
    # df_run_curves_mean.plot(x=0,y=2, kind='line',logx=False) # Positions of cols
    # # df_run_curves.plot(x=0, y=[2], kind='line',logx=False) # Positions of cols
    # # df_run_stats.plot(x=0, y=[1,2,3], kind='line',logx=False) # Positions of cols
    # # plt.ylabel(scoring_metric.capitalize()) #("Accuracy")
    # plt.ylabel(scoring_metric)
    # # plt.xlabel("Maximum Depth Parameter")
    # # plt.suptitle("Validation Curve"+': '+learner_name)
    # ## plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.3f}'))  # 3 decimal places
    # plt.grid(True)
    # plt.savefig('TSP_fitness_all.png')     # save plot
    # # plt.show()
    # plt.close()
    #
    # scoring_metric = 'Time'
    # df_run_curves_mean.plot(x=0,y=1, kind='line',logx=False) # Positions of cols
    # plt.ylabel(scoring_metric)
    # plt.grid(True)
    # plt.savefig('TSP_Time_all.png')     # save plot
    # # plt.show()
    # plt.close()
    #
    # scoring_metric = 'FEval'
    # df_run_curves_mean.plot(x=0,y=3, kind='line',logx=False) # Positions of cols
    # plt.ylabel(scoring_metric)
    # plt.grid(True)
    # plt.savefig('TSP_FEval_all.png')     # save plot
    # # plt.show()
    # plt.close()



    ###################################################
    # scoring_metric = 'Fitness'
    # df_run_stats_mean.plot(x=0,y=1, kind='line',logx=True) # Positions of cols
    # plt.ylabel(scoring_metric)
    # plt.grid(True)
    # plt.savefig('TSP_fitness_all1.png')     # save plot
    # # plt.show()
    # plt.close()
    #
    # scoring_metric = 'Time(ms)'
    # df_run_stats_mean.plot(x=0,y=3, kind='line',logx=True) # Positions of cols
    # plt.ylabel(scoring_metric)
    # plt.grid(True)
    # plt.savefig('TSP_Time_all1.png')     # save plot
    # # plt.show()
    # plt.close()
    #
    # scoring_metric = 'FEvals'
    # df_run_stats_mean.plot(x=0,y=2, kind='line',logx=True) # Positions of cols
    # plt.ylabel(scoring_metric)
    # plt.grid(True)
    # plt.savefig('TSP_FEval_all1.png')     # save plot
    # # plt.show()
    # plt.close()


    ###################################################
    # a. Random HC
    ###################################################
    rhc = RHCRunner(problem=problem,
                    experiment_name='tsp_rhc',
                    seed=631298,
                    iteration_list=4 ** np.arange(7),
                    max_attempts=25,
                    restart_list=[25, 75, 100])

    # the two data frames will contain the results
    df_run_stats, df_run_curves = rhc.run()


    best_fitness = df_run_curves['Fitness'].min()
    best_runs = df_run_curves[df_run_curves['Fitness'] == best_fitness]

    minimum_evaluations = best_runs['FEvals'].min()
    best_curve_run = best_runs[best_runs['FEvals'] == minimum_evaluations]


    # df_run_curves_mean = df_run_curves.groupby('Iteration').mean().reset_index()
    df_run_stats_mean = df_run_stats.groupby('Iteration').mean().reset_index()


    df_fitness = pd.concat([df_fitness, df_run_stats_mean[['Fitness']]], axis=1)
    df_time = pd.concat([df_time, df_run_stats_mean[['Time']]], axis=1)
    df_fevals = pd.concat([df_fevals, df_run_stats_mean[['FEvals']]], axis=1)

    df_fitness.rename(columns={'Fitness': 'RHC'}, inplace=True)
    df_time.rename(columns={'Time': 'RHC'}, inplace=True)
    df_fevals.rename(columns={'FEvals': 'RHC'}, inplace=True)



    ###################################################
    # b. Sim Ann
    ###################################################
    sa = SARunner(problem=problem,
                  experiment_name='tsp_sa',
                  seed=631298,
                  iteration_list=4 ** np.arange(7),
                  temperature_list=[0.1, 0.5, 0.75, 1.0, 2.0, 5.0,1, 10, 50, 100, 250, 500, 1000, 2500, 5000, 10000],
                  decay_list=[mlrose_hiive.GeomDecay],
                  max_attempts = 25)

    # the two data frames will contain the results
    df_run_stats, df_run_curves = sa.run()


    best_fitness = df_run_curves['Fitness'].min()
    best_runs = df_run_curves[df_run_curves['Fitness'] == best_fitness]

    minimum_evaluations = best_runs['FEvals'].min()
    best_curve_run = best_runs[best_runs['FEvals'] == minimum_evaluations]


    # df_run_curves_mean = df_run_curves.groupby('Iteration').mean().reset_index()
    df_run_stats_mean = df_run_stats.groupby('Iteration').mean().reset_index()


    df_fitness = pd.concat([df_fitness, df_run_stats_mean[['Fitness']]], axis=1)
    df_time = pd.concat([df_time, df_run_stats_mean[['Time']]], axis=1)
    df_fevals = pd.concat([df_fevals, df_run_stats_mean[['FEvals']]], axis=1)

    df_fitness.rename(columns={'Fitness': 'SA'}, inplace=True)
    df_time.rename(columns={'Time': 'SA'}, inplace=True)
    df_fevals.rename(columns={'FEvals': 'SA'}, inplace=True)


    ###################################################
    # d. MIMIC
    ###################################################
    mmc = MIMICRunner(problem=problem,
                      experiment_name='tsp_mmc',
                      seed=631298,  # seed=123456, #random_state
                      iteration_list=4 ** np.arange(7), #max_iters
                      population_sizes=[100, 150],  # [20, 30],
                      keep_percent_list=[0.25, 0.5, 0.75],
                      max_attempts=10)


    # the two data frames will contain the results
    df_run_stats, df_run_curves = mmc.run()


    best_fitness = df_run_curves['Fitness'].min()
    best_runs = df_run_curves[df_run_curves['Fitness'] == best_fitness]

    minimum_evaluations = best_runs['FEvals'].min()
    best_curve_run = best_runs[best_runs['FEvals'] == minimum_evaluations]

    # best_mr = best_curve_run['Mutation Rate'].iloc()[0]
    # best_pop_size = best_curve_run['Population Size'].iloc()[0]
    # print('Best Mutation Rate: ',best_mr, 'best Population Size: ',best_pop_size)


    # df_run_curves_mean = df_run_curves.groupby('Iteration').mean().reset_index()
    df_run_stats_mean = df_run_stats.groupby('Iteration').mean().reset_index()


    df_fitness = pd.concat([df_fitness, df_run_stats_mean[['Fitness']]], axis=1)
    df_time = pd.concat([df_time, df_run_stats_mean[['Time']]], axis=1)
    df_fevals = pd.concat([df_fevals, df_run_stats_mean[['FEvals']]], axis=1)

    df_fitness.rename(columns={'Fitness': 'MIMIC'}, inplace=True)
    df_time.rename(columns={'Time': 'MIMIC'}, inplace=True)
    df_fevals.rename(columns={'FEvals': 'MIMIC'}, inplace=True)



    ###################################################
    ###################################################
    ###################################################
    scoring_metric = 'Fitness'
    df_fitness.plot(x='Iteration',y=['GA','SA','MIMIC','RHC'], kind='line',logx=True) # Positions of cols
    # df_run_stats_all.plot(x='Iteration',y=['Fitness','Fitness_mmc'], kind='line',logx=True) # Positions of cols
    plt.ylabel(scoring_metric)
    plt.grid(True)
    plt.savefig('TSP_fitness_all__.png')     # save plot
    # plt.show()
    plt.close()

    scoring_metric = 'Time(ms)'
    df_time.plot(x='Iteration',y=['GA','SA','MIMIC','RHC'], kind='line',logx=True) # Positions of cols
    plt.ylabel(scoring_metric)
    plt.grid(True)
    plt.savefig('TSP_Time_all__.png')     # save plot
    # plt.show()
    plt.close()

    scoring_metric = 'FEvals'
    df_fevals.plot(x='Iteration',y=['GA','SA','MIMIC','RHC'], kind='line',logx=True) # Positions of cols
    plt.ylabel(scoring_metric)
    plt.grid(True)
    plt.savefig('TSP_FEval_all__.png')     # save plot
    # plt.show()
    plt.close()




def FF():
    # Generate a new FF problem using a fixed seed.
    problem = FlipFlopGenerator().generate(seed=631298, size=15)


    ###################################################
    # c. Gen Algo *** Highlight advantages of
    ###################################################
    ga = GARunner(problem=problem,
                  experiment_name='tsp_ga',
                  # output_directory=None,
                  seed=631298, #seed=123456,
                  iteration_list=4 ** np.arange(9),
                  max_attempts=25,
                  population_sizes=[100, 150], #[20, 30],
                  mutation_rates=[0.1, 0.3, 0.5])   #mutation_prob

    # the two data frames will contain the results
    df_run_stats, df_run_curves = ga.run()

    best_fitness = df_run_curves['Fitness'].min()
    best_runs = df_run_curves[df_run_curves['Fitness'] == best_fitness]

    minimum_evaluations = best_runs['FEvals'].min()
    best_curve_run = best_runs[best_runs['FEvals'] == minimum_evaluations]

    best_mr = best_curve_run['Mutation Rate'].iloc()[0]
    best_pop_size = best_curve_run['Population Size'].iloc()[0]
    print('Best Mutation Rate: ',best_mr, 'best Population Size: ',best_pop_size)


    # df_run_curves_mean = df_run_curves.groupby('Iteration').mean().reset_index()
    df_run_stats_mean = df_run_stats.groupby('Iteration').mean().reset_index()
    df_run_stats_all = df_run_stats_mean # Collect results


    df_fitness = df_run_stats_mean[['Iteration', 'Fitness']] # df1 = df.iloc[:, 0:1]
    df_time = df_run_stats_mean[['Iteration', 'Time']]
    df_fevals = df_run_stats_mean[['Iteration', 'FEvals']]

    df_fitness.rename(columns={'Fitness': 'GA'}, inplace=True)
    df_time.rename(columns={'Time': 'GA'}, inplace=True)
    df_fevals.rename(columns={'FEvals': 'GA'}, inplace=True)


    ###################################################
    # a. Random HC
    ###################################################
    rhc = RHCRunner(problem=problem,
                    experiment_name='tsp_rhc',
                    seed=631298,
                    iteration_list=4 ** np.arange(9),
                    max_attempts=25,
                    restart_list=[25, 75, 100])

    # the two data frames will contain the results
    df_run_stats, df_run_curves = rhc.run()


    best_fitness = df_run_curves['Fitness'].min()
    best_runs = df_run_curves[df_run_curves['Fitness'] == best_fitness]

    minimum_evaluations = best_runs['FEvals'].min()
    best_curve_run = best_runs[best_runs['FEvals'] == minimum_evaluations]


    # df_run_curves_mean = df_run_curves.groupby('Iteration').mean().reset_index()
    df_run_stats_mean = df_run_stats.groupby('Iteration').mean().reset_index()


    df_fitness = pd.concat([df_fitness, df_run_stats_mean[['Fitness']]], axis=1)
    df_time = pd.concat([df_time, df_run_stats_mean[['Time']]], axis=1)
    df_fevals = pd.concat([df_fevals, df_run_stats_mean[['FEvals']]], axis=1)

    df_fitness.rename(columns={'Fitness': 'RHC'}, inplace=True)
    df_time.rename(columns={'Time': 'RHC'}, inplace=True)
    df_fevals.rename(columns={'FEvals': 'RHC'}, inplace=True)



    ###################################################
    # b. Sim Ann
    ###################################################
    sa = SARunner(problem=problem,
                  experiment_name='tsp_sa',
                  seed=631298,
                  iteration_list=4 ** np.arange(9),
                  temperature_list=[50, 100, 1000, 2500, 5000, 10000, 30000],
                  decay_list=[mlrose_hiive.GeomDecay],
                  max_attempts = 25)

    # the two data frames will contain the results
    df_run_stats, df_run_curves = sa.run()


    best_fitness = df_run_curves['Fitness'].min()
    best_runs = df_run_curves[df_run_curves['Fitness'] == best_fitness]

    minimum_evaluations = best_runs['FEvals'].min()
    best_curve_run = best_runs[best_runs['FEvals'] == minimum_evaluations]


    # df_run_curves_mean = df_run_curves.groupby('Iteration').mean().reset_index()
    df_run_stats_mean = df_run_stats.groupby('Iteration').mean().reset_index()


    df_fitness = pd.concat([df_fitness, df_run_stats_mean[['Fitness']]], axis=1)
    df_time = pd.concat([df_time, df_run_stats_mean[['Time']]], axis=1)
    df_fevals = pd.concat([df_fevals, df_run_stats_mean[['FEvals']]], axis=1)

    df_fitness.rename(columns={'Fitness': 'SA'}, inplace=True)
    df_time.rename(columns={'Time': 'SA'}, inplace=True)
    df_fevals.rename(columns={'FEvals': 'SA'}, inplace=True)


    ###################################################
    # d. MIMIC
    ###################################################`
    mmc = MIMICRunner(problem=problem,
                      experiment_name='tsp_mmc',
                      seed=631298,  # seed=123456, #random_state
                      iteration_list=4 ** np.arange(9), #max_iters
                      population_sizes=[100, 150],  # [20, 30],
                      keep_percent_list=[0.25, 0.5, 0.75],
                      max_attempts=10)


    # the two data frames will contain the results
    df_run_stats, df_run_curves = mmc.run()


    best_fitness = df_run_curves['Fitness'].min()
    best_runs = df_run_curves[df_run_curves['Fitness'] == best_fitness]

    minimum_evaluations = best_runs['FEvals'].min()
    best_curve_run = best_runs[best_runs['FEvals'] == minimum_evaluations]

    # best_mr = best_curve_run['Mutation Rate'].iloc()[0]
    # best_pop_size = best_curve_run['Population Size'].iloc()[0]
    # print('Best Mutation Rate: ',best_mr, 'best Population Size: ',best_pop_size)


    # df_run_curves_mean = df_run_curves.groupby('Iteration').mean().reset_index()
    df_run_stats_mean = df_run_stats.groupby('Iteration').mean().reset_index()


    df_fitness = pd.concat([df_fitness, df_run_stats_mean[['Fitness']]], axis=1)
    df_time = pd.concat([df_time, df_run_stats_mean[['Time']]], axis=1)
    df_fevals = pd.concat([df_fevals, df_run_stats_mean[['FEvals']]], axis=1)

    df_fitness.rename(columns={'Fitness': 'MIMIC'}, inplace=True)
    df_time.rename(columns={'Time': 'MIMIC'}, inplace=True)
    df_fevals.rename(columns={'FEvals': 'MIMIC'}, inplace=True)



    ###################################################
    ###################################################
    ###################################################
    scoring_metric = 'Fitness'
    df_fitness.plot(x='Iteration',y=['GA','SA','MIMIC','RHC'], kind='line',logx=True) # Positions of cols
    # df_run_stats_all.plot(x='Iteration',y=['Fitness','Fitness_mmc'], kind='line',logx=True) # Positions of cols
    plt.ylabel(scoring_metric)
    plt.grid(True)
    plt.savefig('FF_fitness_all__.png')     # save plot
    # plt.show()
    plt.close()

    scoring_metric = 'Time(ms)'
    df_time.plot(x='Iteration',y=['GA','SA','MIMIC','RHC'], kind='line',logx=True) # Positions of cols
    plt.ylabel(scoring_metric)
    plt.grid(True)
    plt.savefig('FF_Time_all__.png')     # save plot
    # plt.show()
    plt.close()

    scoring_metric = 'FEvals'
    df_fevals.plot(x='Iteration',y=['GA','SA','MIMIC','RHC'], kind='line',logx=True) # Positions of cols
    plt.ylabel(scoring_metric)
    plt.grid(True)
    plt.savefig('FF_FEval_all__.png')     # save plot
    # plt.show()
    plt.close()



# def problem1_abc(): # Need Harder prob: eg. N-queens/K-coloring ==== TSP
#     print()
#
#     # a. Random HC
#     # b. Sim Ann
#     # c. Gen Algo *** Highlight advantages of  ====
#     # d. MIMIC
#
# def problem2_abc():  # Simpler prob
#     print()
#
#     # a. Random HC
#     # b. Sim Ann *** Highlight advantages of
#     # c. Gen Algo
#     # d. MIMIC
#
# def problem3_abc(): # Need Harder prob: eg. N-queens/K-coloring
#     print()
#
#     # a. Random HC
#     # b. Sim Ann
#     # c. Gen Algo
#     # d. MIMIC *** Highlight advantages of



# continuous_peaks_generator.py
# flip_flop_generator.py
# knapsack_generator.py
# max_k_color_generator.py      <<--------<<<
# queens_generator.py          <<--------<<<
# tsp_generator.py

#
# def nn():
#
#     grid_search_parameters = ({
#         'max_iters': [1, 2, 4, 8, 16, 32, 64, 128],                     # nn params
#         'learning_rate': [0.001, 0.002, 0.003],                         # nn params
#         'schedule': [ArithDecay(1), ArithDecay(100), ArithDecay(1000)]  # sa params
#     })
#     nnr = NNGSRunner(x_train=x_train,
#                      y_train=y_train,
#                      x_test=x_test,
#                      y_test=y_test,
#                      experiment_name='nn_test',
#                      algorithm=mlrose_hiive.algorithms.sa.simulated_annealing,
#                      grid_search_parameters=grid_search_parameters,
#                      iteration_list=[1, 10, 50, 100, 250, 500, 1000, 2500, 5000, 10000],
#                      hidden_layer_sizes=[[44,44]],
#                      bias=True,
#                      early_stopping=False,
#                      clip_max=1e+10,
#                      max_attempts=500,
#                      generate_curves=True,
#                      seed=200972)
#     results = nnr.run()          # GridSearchCV instance returned

def problem_8queens():

    # Initialize fitness function object using pre-defined class
    fitness = mlrose_hiive.Queens()

    # Define optimization problem object
    problem = mlrose_hiive.DiscreteOpt(length = 8, fitness_fn = fitness, maximize=False, max_val=8)

    # Define decay schedule
    schedule = mlrose_hiive.ExpDecay()

    # Solve using simulated annealing - attempt 1
    init_state = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    best_state, best_fitness, _ = mlrose_hiive.simulated_annealing(problem=problem, schedule = schedule, max_attempts = 10,
                                                                max_iters = 1000, init_state = init_state, random_state = 1)

    print('The best state found is: ', best_state)
    print('The fitness at the best state is: ', best_fitness)


    # Solve using simulated annealing - attempt 2
    best_state, best_fitness, _ = mlrose_hiive.simulated_annealing(problem, schedule = schedule, max_attempts = 100,
                                                          max_iters = 1000, init_state = init_state,
                                                          random_state = 1)

    print(best_state)
    print(best_fitness)




def main():
    # problem_8queens()

    # TSP() # Travelling Sales Man

    FF() # Flip Flop
    # nn()




if __name__ == "__main__":
    main()

