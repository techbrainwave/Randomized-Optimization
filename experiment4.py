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


def nn():
    # Age	year	nodes	Survival
    cancer_df = pd.read_csv(path.join('data','haberman.data'))
    # cancer_df.rename(columns={'Class':'Class_category'}, inplace=True)
    dataset = 2

    cancer_df.columns = ['Age',	'Year',	'Nodes','survival']
    # y = cancer_df.iloc[:, -1]


    # Negative class/Long survival :0 , Positive class/short life :1
    cancer_df.survival.replace((1, 2), (0, 1), inplace=True)


    y = cancer_df['survival']
    X = cancer_df.drop(['survival'], axis=1)


    # define oversampling strategy
    oversample = RandomOverSampler(sampling_strategy='minority')
    # fit and apply the transform
    X_over, y_over = oversample.fit_resample(X, y)
    X = X_over
    y = y_over

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3240)


    ################################################
    # BackProp
    ################################################
    # learner = MLPClassifier(hidden_layer_sizes=(30,5), learning_rate_init=0.000001, max_iter=300)

    # grid_search_parameters = ({
    #     'max_iters': [1, 2, 4, 8, 16, 32, 64, 128],                     # nn params
    #     'learning_rate': [0.001, 0.002, 0.003],                         # nn params
    #     'schedule': [ArithDecay(1), ArithDecay(100), ArithDecay(1000)]  # sa params
    # })

    grid_search_parameters = ({
        'max_iters': [300],  # nn params
        'learning_rate': [0.000001],  # nn params
        'activation': [mlrose_hiive.relu],  # nn params
        # 'restarts': [1],  # rhc params
    })
    nnr = NNGSRunner(x_train=X_train,
                     y_train=y_train,
                     x_test=X_test,
                     y_test=y_test,
                     experiment_name='nn_gd',
                     algorithm=mlrose_hiive.algorithms.gd.gradient_descent,
                     # algorithm=mlrose_hiive.algorithms.sa.simulated_annealing,
                     grid_search_parameters=grid_search_parameters,
                     iteration_list=4 ** np.arange(7),
                     # iteration_list=[1, 10, 50, 100, 250, 500, 1000, 2500, 5000, 10000],
                     hidden_layer_sizes=[[30,5]],
                     # hidden_layer_sizes=[[44,44]],
                     bias=True,
                     early_stopping=False,
                     clip_max=1e+10,
                     max_attempts=10,
                     n_jobs=2, cv=4,
                     generate_curves=True,
                     seed=631298)
    results = nnr.run()          # GridSearchCV instance returned


    # def __init__(self, x_train, y_train, x_test, y_test, experiment_name, seed, iteration_list, algorithm,
    #              grid_search_parameters, grid_search_scorer_method=skmt.balanced_accuracy_score,
    #              bias=True, early_stopping=True, clip_max=1e+10, activation=None,
    #              max_attempts=500, n_jobs=1, cv=5, generate_curves=True, output_directory=None,
    #              **kwargs):



    ################################################
    # 1 RHC
    ################################################

    grid_search_parameters = ({
        'max_iters': [300],  # nn params
        'learning_rate': [0.000001],  # nn params
        'activation': [mlrose_hiive.relu],  # nn params
        # 'restarts': [1],  # rhc params
    })
    nnr = NNGSRunner(x_train=X_train,
                     y_train=y_train,
                     x_test=X_test,
                     y_test=y_test,
                     experiment_name='nn_gd',
                     algorithm=mlrose_hiive.algorithms.rhc.random_hill_climb,
                     # algorithm=mlrose_hiive.algorithms.sa.simulated_annealing,
                     grid_search_parameters=grid_search_parameters,
                     iteration_list=4 ** np.arange(7),
                     # iteration_list=[1, 10, 50, 100, 250, 500, 1000, 2500, 5000, 10000],
                     hidden_layer_sizes=[[30,5]],
                     # hidden_layer_sizes=[[44,44]],
                     bias=True,
                     early_stopping=False,
                     clip_max=1e+10,
                     max_attempts=10,
                     n_jobs=2, cv=4,
                     generate_curves=True,
                     seed=631298)
    results = nnr.run()          # GridSearchCV instance returned


    ################################################
    # 2 SA
    ################################################

    grid_search_parameters = ({
        'max_iters': [300],  # nn params
        'learning_rate': [0.000001],  # nn params
        'activation': [mlrose_hiive.relu],  # nn params
        # 'restarts': [1],  # rhc params
    })
    nnr = NNGSRunner(x_train=X_train,
                     y_train=y_train,
                     x_test=X_test,
                     y_test=y_test,
                     experiment_name='nn_gd',
                     algorithm=mlrose_hiive.algorithms.sa.simulated_annealing,
                     grid_search_parameters=grid_search_parameters,
                     iteration_list=4 ** np.arange(7),
                     # iteration_list=[1, 10, 50, 100, 250, 500, 1000, 2500, 5000, 10000],
                     hidden_layer_sizes=[[30,5]],
                     # hidden_layer_sizes=[[44,44]],
                     bias=True,
                     early_stopping=False,
                     clip_max=1e+10,
                     max_attempts=10,
                     n_jobs=2, cv=4,
                     generate_curves=True,
                     seed=631298)
    results = nnr.run()          # GridSearchCV instance returned
    ################################################
    # 3 GA
    ################################################


    grid_search_parameters = ({
        'max_iters': [300],  # nn params
        'learning_rate': [0.000001],  # nn params
        'activation': [mlrose_hiive.relu],  # nn params
        # 'restarts': [1],  # rhc params
    })
    nnr = NNGSRunner(x_train=X_train,
                     y_train=y_train,
                     x_test=X_test,
                     y_test=y_test,
                     experiment_name='nn_gd',
                     algorithm=mlrose_hiive.algorithms.ga.genetic_alg,
                     grid_search_parameters=grid_search_parameters,
                     iteration_list=4 ** np.arange(7),
                     # iteration_list=[1, 10, 50, 100, 250, 500, 1000, 2500, 5000, 10000],
                     hidden_layer_sizes=[[30, 5]],
                     # hidden_layer_sizes=[[44,44]],
                     bias=True,
                     early_stopping=False,
                     clip_max=1e+10,
                     max_attempts=10,
                     n_jobs=2, cv=4,
                     generate_curves=True,
                     seed=631298)
    results = nnr.run()  # GridSearchCV instance returned


def main():
    nn()




if __name__ == "__main__":
    main()

