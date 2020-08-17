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
    all_model_params = hp = [model.get_params() for model in all_models]
    all_model_info = pd.DataFrame(all_model_params)
    all_model_info.insert(0, 'model_type', all_model_types)
    all_model_info.to_csv('results/base_learner_details.csv', index = False)

    # delay all_models object to avoid rehashing
    all_models = dask.delayed(all_models)

    # looping through datasets

    # initialze batch size and number of batches (for interim saving of results)
    dataset_batch_size = 60
    num_batches = int(np.ceil(len(regression_dataset_names) / dataset_batch_size))
    
    # iterate over batches
    for batch in range(num_batches):

        # define first and last datasets of current batch
        first_dataset = batch*dataset_batch_size
        last_dataset = min((batch+1)*dataset_batch_size, len(regression_dataset_names))

        # initialize empty results list for current batch
        all_iteration_output = []

        # iterate over datasets for current batch
        for dataset in regression_dataset_names[first_dataset:last_dataset]:

            # get data
            dataset_num = regression_dataset_names.index(dataset)
            X, y = fetch_data(dataset, return_X_y=True)

            # standardize features and outcome
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            y = np.squeeze(scaler.fit_transform(y.reshape(-1, 1)))

            # collected iterations into list of lazy tasks (all_iteration_output)
            iteration_output = dask.delayed(slt.runEvaluationIteration)(X, y, 
                                                                        all_models, 
                                                                        dataset_name=dataset,
                                                                        dataset_num=dataset_num)
            all_iteration_output.append(iteration_output)

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
                all_iteration_computed_output.append(list(dask.compute(pout))[0])
            except Exception:
                pass

        # separate results and meta data into separate lists for current batch
        all_results, all_sl_all, all_sl_nonzero, all_cv_mse = map(list, zip(*all_iteration_computed_output)) 

        # flatten results and meta data for current batch
        all_results = [item for sublist in all_results for item in sublist]
        all_sl_all = [item for sublist in all_sl_all for item in sublist]
        all_sl_nonzero = [item for sublist in all_sl_nonzero for item in sublist]
        all_cv_mse = [item for sublist in all_cv_mse for item in sublist]

        # organize results as dataframes for current batch
        # actual validation metrics
        all_results = pd.DataFrame(all_results, 
                                   columns = ['dataset', 'fold', 'linear_mod',
                                              'sl_full', 'sl_best_grid_1',
                                              'sl_best_grid_3', 'sl_best_grid_5',
                                              'sl_best_random10_1', 'sl_best_random10_3', 
                                              'sl_best_random10_5', 'sl_best_random25_1', 
                                              'sl_best_random25_3', 'sl_best_random25_5', 
                                              'sl_best_random50_1', 'sl_best_random50_3', 
                                              'sl_best_random50_5', 'sl_best_random100_1', 
                                              'sl_best_random100_3', 'sl_best_random100_5',                                                                    
                                              'sl_default_subset', 'sl_ga_subset',
                                              'discrete_sl'])
        
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

        # export results for current batch
        all_results.to_csv('results/results_batch_' + str(batch) + '.csv', index = False)
        all_sl_all.to_csv('results/sl_inputs_batch_' + str(batch) + '.csv', index = False)
        all_sl_nonzero.to_csv('results/sl_weights_batch_' + str(batch) + '.csv', index = False)
        all_cv_mse.to_csv('results/base_learner_cv_mse_batch_' + str(batch) + '.csv', index = False)

        # delete objects for current batch
        del all_iteration_output
        del all_iteration_persisted_output
        del all_iteration_computed_output
        del all_results
        del all_sl_all
        del all_sl_nonzero
        del all_cv_mse

if __name__ == '__main__':
    main()
