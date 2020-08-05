# import libraries
import os
import numpy as np
import pandas as pd
from dask_mpi import initialize
from distributed import Client, wait
import dask.multiprocessing
from pmlb import classification_dataset_names, regression_dataset_names, fetch_data
from sklearn.preprocessing import StandardScaler
import superlearner_tuning as slt
from models import *

def main():

    # initialize parallelization
    initialize(local_directory=None)

    # setup client for parallelization
    client = Client() 

    # generate list of all models and delay to avoid re-hashing
    all_models = slt.getAllModels(model_specs)

    # save model type and parameters prior to delaying all_models object
    all_model_types = [type(model).__name__ for model in all_models]
    all_model_params = hp = [model.get_params() for model in all_models]
    all_model_info = pd.DataFrame(all_model_params)
    all_model_info.insert(0, 'model_type', all_model_types)

    # delay all_models object to avoid rehashing
    all_models = dask.delayed(all_models)

    # initialize empty results list
    all_iteration_output = []

    # looping through datasets
    for dataset in regression_dataset_names:

        # print statement
        print("Working on " + dataset + "...")

        # get data
        X, y = fetch_data(dataset, return_X_y=True)

        # standardize features and outcome
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        y = np.squeeze(scaler.fit_transform(y.reshape(-1, 1)))

        # collected delayed iterations
        iteration_output = dask.delayed(slt.runEvaluationIteration)(X, y, all_models, dataset_name=dataset)
        all_iteration_output.append(iteration_output)

    # compute and collect results
    all_iteration_persisted_output = dask.persist(*all_iteration_output)

    for pout in all_iteration_persisted_output:
        try:
            wait(pout)
        except Exception:
            pass

    all_iteration_computed_output = []
    for pout in all_iteration_persisted_output:
        try:
            all_iteration_computed_output.append(list(dask.compute(pout))[0])
        except Exception:
            pass

    # separate results and meta data into separate lists
    all_results, all_sl_all, all_sl_nonzero, all_cv_mse = map(list, zip(*all_iteration_computed_output)) 

    # flatten meta data
    all_results = [item for sublist in all_results for item in sublist]
    all_sl_all = [item for sublist in all_sl_all for item in sublist]
    all_sl_nonzero = [item for sublist in all_sl_nonzero for item in sublist]
    all_cv_mse = [item for sublist in all_cv_mse for item in sublist]

    # organize results as dataframes

    # actual validation metrics
    all_results = pd.DataFrame(all_results, 
                               columns = ['dataset', 'seed', 'linear_regression',
                                          'sl_full', 'sl_best_subset_1',
                                          'sl_best_subset_3', 'sl_best_subset_5',
                                          'sl_ga_subset', 'sl_random_subset',
                                          'sl_default_subset', 'discrete_sl'])
    
    # indices for base learners selected as input into SuperLearner                                      
    all_sl_all = pd.DataFrame(all_sl_all, 
                              columns = ['dataset', 'seed', 'strategy', 
                                         'index'])

    # indices and weights for baselearners incorporated into SuperLearner                                                                 
    all_sl_nonzero = pd.DataFrame(all_sl_nonzero,  
                                  columns = ['dataset', 'seed', 'strategy', 
                                             'index', 'weight'])

    # CV MSE for all base learners
    all_cv_mse_columns = list(range(len(all_cv_mse[0]) - 2))
    all_cv_mse_columns.insert(0, 'dataset')
    all_cv_mse_columns.insert(1, 'seed')
    all_cv_mse = pd.DataFrame(all_cv_mse, columns = all_cv_mse_columns)

    # export
    os.chdir(os.path.dirname(os.getcwd()))
    all_results.to_csv('results/results.csv', index = False)
    all_sl_all.to_csv('results/sl_inputs.csv', index = False)
    all_sl_nonzero.to_csv('results/sl_weights.csv', index = False)
    all_cv_mse.to_csv('results/base_learner_cv_mse.csv', index = False)
    all_model_info.to_csv('results/base_learner_details.csv', index = False)

if __name__ == '__main__':
    main()
