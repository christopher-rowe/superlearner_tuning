# import libraries
import os
import numpy as np
import pandas as pd
from pmlb import classification_dataset_names, regression_dataset_names, fetch_data
from sklearn.preprocessing import StandardScaler
import superlearner_tuning as slt
from models_test import *

def main():

    # generate list of all models
    all_models = slt.getAllModels(model_specs)

    # initialize empty results dataframes
    all_results = pd.DataFrame()
    all_sl_all = pd.DataFrame()
    all_sl_nonzero = pd.DataFrame()
    all_cv_mse = pd.DataFrame()

    # set seed
    np.random.seed(seed = 6053)

    # looping through datasets
    for dataset in regression_dataset_names[0:1]:

        # print statement
        print("Working on " + dataset + "...")

        # get data
        X, y = fetch_data(dataset, return_X_y=True)

        # standardize features and outcome
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        y = np.squeeze(scaler.fit_transform(y.reshape(-1, 1)))

        # collected iterations
        iteration_output = slt.runEvaluationIteration(X, y, all_models, dataset_name = dataset)
        all_results_single_dataset = iteration_output[0]
        all_sl_all_single_dataset = iteration_output[1]
        all_sl_nonzero_single_dataset = iteration_output[2]
        all_cv_mse_single_dataset = iteration_output[3]

        # actual validation metrics
        all_results_single_dataset = pd.DataFrame(all_results_single_dataset, 
                                                  columns = ['dataset', 'fold', 'linear_regression',
                                                             'sl_full', 'sl_best_grid_1',
                                                             'sl_best_grid_3', 'sl_best_grid_5',
                                                             'sl_best_random10_1', 'sl_best_random10_3', 
                                                             'sl_best_random10_5', 'sl_best_random25_1', 
                                                             'sl_best_random25_3', 'sl_best_random25_5', 
                                                             'sl_best_random50_1', 'sl_best_random50_3', 
                                                             'sl_best_random50_5', 'sl_best_random100_1', 
                                                             'sl_best_random100_3', 'sl_best_random100_5',                                                                    
                                                             'sl_ga_subset', 'sl_default_subset', 
                                                             'discrete_sl'])

        # indices for base learners selected as input into SuperLearner                                      
        all_sl_all_single_dataset = pd.DataFrame(all_sl_all_single_dataset, 
                                                 columns = ['dataset', 'fold', 'strategy', 
                                                            'index'])

        # indices and weights for baselearners incorporated into SuperLearner                                     
        all_sl_nonzero_single_dataset = pd.DataFrame(all_sl_nonzero_single_dataset,  
                                                     columns = ['dataset', 'fold', 'strategy', 
                                                                'index', 'weight'])

        # CV MSE for all base learners
        all_cv_mse_single_dataset_columns = list(range(len(all_cv_mse_single_dataset[0]) - 2))
        all_cv_mse_single_dataset_columns.insert(0, 'dataset')
        all_cv_mse_single_dataset_columns.insert(1, 'fold')
        all_cv_mse_single_dataset = pd.DataFrame(all_cv_mse_single_dataset, columns = all_cv_mse_single_dataset_columns)

        # concatenate datasets
        all_results = pd.concat([all_results, all_results_single_dataset], ignore_index=True)
        all_sl_all = pd.concat([all_sl_all, all_sl_all_single_dataset], ignore_index=True)
        all_sl_nonzero = pd.concat([all_sl_nonzero, all_sl_nonzero_single_dataset], ignore_index=True)
        all_cv_mse = pd.concat([all_cv_mse, all_cv_mse_single_dataset], ignore_index=True)

        # print statement
        print("Finished with " + dataset + "!")

    # model and parameters
    all_model_types = [type(model).__name__ for model in all_models]
    all_model_params = hp = [model.get_params() for model in all_models]
    all_model_info = pd.DataFrame(all_model_params)
    all_model_info.insert(0, 'model_type', all_model_types)

    # export
    os.chdir(os.path.dirname(os.getcwd()))
    all_results.to_csv('results/results.csv', index = False)
    all_sl_all.to_csv('results/sl_inputs.csv', index = False)
    all_sl_nonzero.to_csv('results/sl_weights.csv', index = False)
    all_cv_mse.to_csv('results/base_learner_cv_mse.csv', index = False)
    all_model_info.to_csv('results/base_learner_details.csv', index = False)       

if __name__ == '__main__':
    main()
