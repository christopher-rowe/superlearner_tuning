# import libraries
import os
import numpy as np
import pandas as pd
from pmlb import fetch_data
from sklearn.preprocessing import StandardScaler
import superlearner_tuning as slt
from models_test import *
from datasets import *

def main():
    
    n = []
    p = []
    for dataset in regression_datasets:
        X, y = fetch_data(dataset, return_X_y=True)
        n.append(X.shape[0])
        p.append(X.shape[1])
    dataset_info = pd.DataFrame({'dataset' : regression_datasets,
                                 'n' : n,
                                 'p' : p})


    # generate list of all models
    all_models = slt.getAllModels(model_specs)

    # initialize empty results dataframes
    all_results = pd.DataFrame()
    all_sl_all = pd.DataFrame()
    all_sl_nonzero = pd.DataFrame()
    all_cv_mse = pd.DataFrame()
    all_num_ga_models = pd.DataFrame()
    all_y_preds = pd.DataFrame()

    # set seed
    np.random.seed(seed = 6053)

    # looping through datasets
    for dataset in regression_datasets[0:2]:

        # get data
        dataset_num = regression_datasets.index(dataset)
        X, y = fetch_data(dataset, return_X_y=True)

        # standardize features and outcome
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        y = np.squeeze(scaler.fit_transform(y.reshape(-1, 1)))

        # collected iterations
        iteration_output = slt.runEvaluationIteration(X, y, 
                                                      all_models, 
                                                      dataset_name=dataset,
                                                      dataset_num=dataset_num)
        all_results_single_dataset = iteration_output[0]
        all_sl_all_single_dataset = iteration_output[1]
        all_sl_nonzero_single_dataset = iteration_output[2]
        all_cv_mse_single_dataset = iteration_output[3]
        all_num_ga_models_single_dataset = iteration_output[4]
        all_y_preds_single_dataset = iteration_output[5]

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

        all_results_single_dataset = pd.DataFrame(all_results_single_dataset, 
                                                  columns = results_columns)

        # indices for base learners selected as input into SuperLearner                                      
        all_sl_all_single_dataset = pd.DataFrame(all_sl_all_single_dataset, 
                                                 columns = ['dataset', 'fold', 'strategy', 
                                                            'index'])

        # indices and weights for baselearners incorporated into SuperLearner                                     
        all_sl_nonzero_single_dataset = pd.DataFrame(all_sl_nonzero_single_dataset,  
                                                     columns = ['dataset', 'fold', 'strategy', 
                                                                'index', 'sl_weight'])

        # CV MSE for all base learners
        all_cv_mse_single_dataset_columns = list(range(len(all_cv_mse_single_dataset[0]) - 2))
        all_cv_mse_single_dataset_columns.insert(0, 'dataset')
        all_cv_mse_single_dataset_columns.insert(1, 'fold')
        all_cv_mse_single_dataset = pd.DataFrame(all_cv_mse_single_dataset, columns = all_cv_mse_single_dataset_columns)

        # num ga models
        fold_columns = ['fold_' + z for z in [str(x) for x in list(np.arange(1,11))]]
        all_num_ga_models_single_dataset_columns = ['dataset'] + fold_columns
        all_num_ga_models_single_dataset.insert(0,dataset)
        all_num_ga_models_single_dataset = pd.DataFrame(np.column_stack(all_num_ga_models_single_dataset),
                                                 columns=all_num_ga_models_single_dataset_columns)


        # observed and predicted outcomes
        all_y_preds_single_dataset_df = pd.DataFrame()
        all_y_preds_single_dataset_val = pd.DataFrame()

        # dataset and folds
        for i in range(len(all_y_preds_single_dataset)):
            all_y_preds_single_dataset_df = pd.concat([all_y_preds_single_dataset_df, 
                                                      pd.DataFrame(np.column_stack(all_y_preds_single_dataset[i][0:2]))])

        # values
        for i in range(len(all_y_preds_single_dataset)):
            all_y_preds_single_dataset_val = pd.concat([all_y_preds_single_dataset_val, 
                                                      pd.DataFrame(np.column_stack(all_y_preds_single_dataset[i][2:len(all_y_preds_single_dataset[0])]))])
        
        all_y_preds_single_dataset = pd.concat([all_y_preds_single_dataset_df, all_y_preds_single_dataset_val], axis = 1)
        results_columns.insert(2, 'y')
        all_y_preds_single_dataset.columns = results_columns

        # concatenate datasets
        all_results = pd.concat([all_results, all_results_single_dataset], ignore_index=True)
        all_sl_all = pd.concat([all_sl_all, all_sl_all_single_dataset], ignore_index=True)
        all_sl_nonzero = pd.concat([all_sl_nonzero, all_sl_nonzero_single_dataset], ignore_index=True)
        all_cv_mse = pd.concat([all_cv_mse, all_cv_mse_single_dataset], ignore_index=True)
        all_num_ga_models = pd.concat([all_num_ga_models, all_num_ga_models_single_dataset], ignore_index=True)
        all_y_preds = pd.concat([all_y_preds, all_y_preds_single_dataset], ignore_index=True)


    # model and parameters
    all_model_types = [type(model).__name__ for model in all_models]
    all_model_params = [model.get_params() for model in all_models]
    all_model_info = pd.DataFrame(all_model_params)
    all_model_info.insert(0, 'model_type', all_model_types)

    # export
    os.chdir(os.path.dirname(os.getcwd()))
    dataset_info.to_csv('results/dataset_info.csv', index = False)
    all_results.to_csv('results/results.csv', index = False)
    all_sl_all.to_csv('results/sl_inputs.csv', index = False)
    all_sl_nonzero.to_csv('results/sl_weights.csv', index = False)
    all_cv_mse.to_csv('results/base_learner_cv_mse.csv', index = False)
    all_num_ga_models.to_csv('results/num_ga_models.csv', index = False)
    all_y_preds.to_csv('results/sl_preds.csv', index = False)
    all_model_info.to_csv('results/base_learner_details.csv', index = False)       

if __name__ == '__main__':
    main()
