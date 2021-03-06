# import libraries
import os
import numpy as np
import pandas as pd
from dask_mpi import initialize
from distributed import Client, wait
import dask.multiprocessing
from pmlb import fetch_data
from sklearn.preprocessing import StandardScaler
import superlearner_tuning as slt
from models_test import *
from datasets import *

def main():

    # navigate to parent directory
    os.chdir(os.path.dirname(os.getcwd()))

    # initialize parallelization
    initialize(local_directory=None)

    # setup client for parallelization
    client = Client() 

    # generate list of all models and delay to avoid re-hashing
    all_models = slt.getAllModels(model_specs)

    # save model type and parameters prior to delaying all_models object
    all_model_types = [type(model).__name__ for model in all_models]
    all_model_params = [model.get_params() for model in all_models]
    all_model_info = pd.DataFrame(all_model_params)
    all_model_info.insert(0, 'model_type', all_model_types)
    all_model_info.to_csv('results/parallel/base_learner_details.csv', index = False)

    # delay all_models object to avoid rehashing
    all_models = dask.delayed(all_models)

    # initialize empty results list for current batch
    all_iteration_output = []

    # iterate over datasets for current batch
    for dataset in regression_datasets[0:2]:

        # get data
        dataset_num = regression_datasets.index(dataset)
        X, y = fetch_data(dataset, return_X_y=True)

        # standardize features and outcome
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        y = np.squeeze(scaler.fit_transform(y.reshape(-1, 1)))

        # collected iterations into list of lazy tasks (all_iteration_output)
        iteration_output = slt.runEvaluationIteration(X, y, all_models, 
                                                      dataset_name=dataset,
                                                      dataset_num=dataset_num,
                                                      parallel=True)
        all_iteration_output.append(iteration_output) # list of dataset list of fold lists (of model result lists)

    # execute tasks for current batch (via methods suggested on https://github.com/dask/distributed/issues/2436)
    all_iteration_persisted_output = dask.persist(*all_iteration_output)
    for pout in all_iteration_persisted_output:
        try:
            wait(pout)
        except Exception:
            pass
    
    # retain results without exceptions for current batch
    all_iteration_computed_output = []
    for pout in all_iteration_persisted_output:
        try:
            all_iteration_computed_output.append(dask.compute(pout))
        except Exception:
            pass

    # separate results and meta data into separate lists for current batch
    all_results = []
    all_sl_all = []
    all_sl_nonzero = []
    all_cv_mse = []
    all_num_ga_models = []
    all_y_preds = []
    for i in range(len(all_iteration_computed_output)): # looping over datasets
        for j in range(len(all_iteration_computed_output[i][0])): # looping over folds
            all_results.append(all_iteration_computed_output[i][0][j][0])
            all_sl_all.append(all_iteration_computed_output[i][0][j][1])
            all_sl_nonzero.append(all_iteration_computed_output[i][0][j][2])
            all_cv_mse.append(all_iteration_computed_output[i][0][j][3])
            all_num_ga_models.append(all_iteration_computed_output[i][0][j][4])
            all_y_preds.append(all_iteration_computed_output[i][0][j][5])

    # flatten results as aapprorpiate
    all_sl_all = [item for sublist in all_sl_all for item in sublist]
    all_sl_nonzero = [item for sublist in all_sl_nonzero for item in sublist]

    # organize results as dataframes for current batch
    # actual validation metrics
    results_columns = ['dataset', 'fold', 'linear_mod',
                       'sl_full', 'sl_best_grid_1',
                       'sl_best_grid_3', 'sl_best_grid_5',
                       'sl_best_random10_1', 'sl_best_random10_3', 
                       'sl_best_random10_5', 'sl_best_random25_1', 
                       'sl_best_random25_3', 'sl_best_random25_5', 
                       'sl_best_random50_1', 'sl_best_random50_3', 
                       'sl_best_random50_5', 'sl_default_subset', 
                       'sl_ga_subset', 'discrete_sl']
    all_results = pd.DataFrame(all_results, 
                                columns = results_columns)
    
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

    # num ga models
    all_num_ga_models = pd.DataFrame(all_num_ga_models, columns = ['num_ga_models'])
    all_num_ga_models['dataset'] = [item for sublist in [[x] * 10 for x in regression_datasets for item in sublist]
    all_num_ga_models['fold'] = list(np.arange(1,11)) * len(regression_datasets)

    # observed and predicted outcomes
    all_y_preds_df = pd.DataFrame()
    all_y_preds_val = pd.DataFrame()

    # dataset and folds
    print(all_y_preds)
    print(len(all_y_preds))
    for i in range(len(all_y_preds)):
        all_y_preds_df = pd.concat([all_y_preds_df, 
                                    pd.DataFrame(np.column_stack(all_y_preds[i][0:2]))])

    # values
    for i in range(len(all_y_preds)):
        all_y_preds_val = pd.concat([all_y_preds_val, 
                                    pd.DataFrame(np.column_stack(all_y_preds[i][2:len(all_y_preds[0])]))])

    all_y_preds = pd.concat([all_y_preds_df, all_y_preds_val], axis = 1)
    results_columns.insert(2, 'y')
    all_y_preds.columns = results_columns   

    # export results for current batch
    all_results.to_csv('results/parallel/results.csv', index = False)
    all_sl_all.to_csv('results/parallel/sl_inputs.csv', index = False)
    all_sl_nonzero.to_csv('results/parallel/sl_weights.csv', index = False)
    all_cv_mse.to_csv('results/parallel/base_learner_cv_mse.csv', index = False)
    all_num_ga_models.to_csv('results/parallel/num_ga_models.csv', index = False)
    all_y_preds.to_csv('results/parallel/sl_preds.csv', index = False)

if __name__ == '__main__':
    main()
