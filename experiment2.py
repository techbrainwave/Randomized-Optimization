import mlrose_hiive
import numpy as np
import pandas as pd
from mlrose_hiive import QueensGenerator, MaxKColorGenerator, TSPGenerator, FlipFlopGenerator, KnapsackGenerator,ContinuousPeaksGenerator
from mlrose_hiive import SARunner, GARunner, NNGSRunner, MIMICRunner, RHCRunner
# # import itertools as it
# import seaborn as sns
import matplotlib.pyplot as plt
# from matplotlib.ticker import StrMethodFormatter
from imblearn.over_sampling import RandomOverSampler
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
import time as tm
# # from sklearn import metrics
from os import path


def FFP(size):
    # Generate a new FF problem using a fixed seed.
    problem = FlipFlopGenerator().generate(seed=631298, size=size) #15


    ###################################################
    # c. Gen Algo *** Highlight advantages of
    ###################################################
    t1 = tm.time()
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
    t2 = tm.time()

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
    t3 = tm.time()
    rhc = RHCRunner(problem=problem,
                    experiment_name='tsp_rhc',
                    seed=631298,
                    iteration_list=4 ** np.arange(9),
                    max_attempts=25,
                    restart_list=[25, 75, 100])

    # the two data frames will contain the results
    df_run_stats, df_run_curves = rhc.run()
    t4 = tm.time()

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
    t5 = tm.time()
    sa = SARunner(problem=problem,
                  experiment_name='tsp_sa',
                  seed=631298,
                  iteration_list=4 ** np.arange(9),
                  temperature_list=[50, 100, 1000, 2500, 5000, 10000, 30000],
                  decay_list=[mlrose_hiive.GeomDecay],
                  max_attempts = 25)

    # the two data frames will contain the results
    df_run_stats, df_run_curves = sa.run()
    t6 = tm.time()

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
    t7 = tm.time()
    mmc = MIMICRunner(problem=problem,
                      experiment_name='tsp_mmc',
                      seed=631298,  # seed=123456, #random_state
                      iteration_list=4 ** np.arange(9), #max_iters
                      population_sizes=[100, 150],  # [20, 30],
                      keep_percent_list=[0.25, 0.5, 0.75],
                      max_attempts=10)


    # the two data frames will contain the results
    df_run_stats, df_run_curves = mmc.run()
    t8 = tm.time()

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


    return [t2-t1, t4-t3, t6-t5, t8-t7]




def main():



    f1 = tm.time()
    FFP(10) # Flip Flop
    f2 = tm.time()

    f3 = tm.time()
    ffp_time = FFP(20) # Flip Flop
    f4 = tm.time()

    f5 = tm.time()
    FFP(30) # Flip Flop
    f6 = tm.time()
    ffp_fulltime = [f2-f1, f4-f3, f6-f5]


    df_ffp = pd.DataFrame(ffp_time).T
    df_ffp.rename(columns={0: 'GA', 1: 'RHC', 2:'SA', 3:'MIMIC'}, inplace=True)
    scoring_metric = 'Time'
    df_ffp.plot(y=['GA','RHC','SA','MIMIC'], kind='bar',logx=False) # Positions of cols
    plt.suptitle('FFP')
    plt.xticks([])
    plt.xlabel('Algorithm')
    plt.ylabel(scoring_metric)
    plt.grid(True)
    plt.savefig('ffp_time.png')     # save plot
    plt.show()
    plt.close()


    #########################################
    #########################################
    #########################################
    #########################################

    df_ffp = pd.DataFrame(ffp_fulltime).T
    df_ffp.rename(columns={0: 10, 1: 20, 2: 30}, inplace=True)
    scoring_metric = 'Time'
    df_ffp.plot( kind='bar',logx=False) # Positions of cols
    plt.suptitle('FFP')
    plt.xticks([])
    plt.xlabel('Size')
    plt.ylabel(scoring_metric)
    plt.grid(True)
    plt.savefig('ffp_time.png')     # save plot
    plt.show()
    plt.close()




if __name__ == "__main__":
    main()

