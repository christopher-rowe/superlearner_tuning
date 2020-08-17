# import libraries
import os
import pandas as pd 
import numpy as np
from pyeasyga import pyeasyga
from scipy.optimize import nnls
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import (GridSearchCV, KFold, train_test_split,
                                     ParameterGrid, RandomizedSearchCV)
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor


def getAllModels(model_specs):
    """Generate list of base models from pre-specified model specifications

    Args:
        model_specs (dictionary): dictionary of model types and hyperparameter 
        settings

    Returns:
        list: list of all base model objects (scikit-learn)
    """

    # initialize empty list for models)
    all_models = []

    # iterate model types and hyperparameter specifications and construct list
    # of model objects
    for model in model_specs:

        for pars in list(ParameterGrid(model_specs[model][1])):

            base_model = clone(model_specs[model][0])
            all_models.append(base_model.set_params(**pars))

    return all_models


def getMSEandPredictions(X, y, models):
    """Obtains 10-fold out-of-sample predictions and mean CV MSE
       for each base model

    Args:
        X (numpy array): feature array
        y (numpy array): outcome vector
        models ([type]): list of all model objects

    Returns:
        numpy arrays: out-of-sample predictions and mean
                      mean risk for each base model, observed outcome 
    """

    # initialize lists for all predictions, true y, and mse
    meta_X, meta_y, cv_mse = list(), list(), list()

    # define folds
    kfold = KFold(n_splits=10, shuffle=True)

    # iterate through folds
    for train_index, test_index in kfold.split(X):

        # initialize lists for single fold predictions and mse
        yhats_fold = list()
        mse_fold = list()

        # get data for single fold
        X_train_cv, X_test_cv = X[train_index], X[test_index]
        y_train_cv, y_test_cv = y[train_index], y[test_index]

        # save true y's
        meta_y.extend(y_test_cv)

        # fit and make predictions with each sub-model
        for model in models:

            # initialize single model
            single_model = clone(model)

            # fit model on training folds
            single_model.fit(X_train_cv, y_train_cv)

            # predict on test fold
            yhat = single_model.predict(X_test_cv)

            # store test predictions for single fold
            yhats_fold.append(yhat.reshape(len(yhat),1))

            # store mse for single fold
            mse_fold.append(mean_squared_error(y_test_cv, yhat))

        # store single fold test predictions as columns
        meta_X.append(np.hstack(yhats_fold))

        # store single fold mse as list
        cv_mse.append(mse_fold)
    
    # calculate cross-validated mse (i.e., mean across folds)
    mean_cv_mse = np.vstack(cv_mse).mean(axis = 0)
    
    return np.vstack(meta_X), np.asarray(meta_y), mean_cv_mse

def getModelIndices(model_list, subset, cv_mse = None, n_random_iter = None, num_models = None):
    """Get model indices to allow subsetting of models and out-of-sample
       predictions

    Args:
        model_list (list): list of all model objects
        subset (str): 'best_grid' to obtain model(s) with best MSE of each
                       model type identified via exhaustive grid search; 
                       'best_random' to obtain model(s) with best MSE of each
                       model type identified via random grid search;
                       'default' for default hyperparameters
        cv_mse (list, optional): Only needed for 'best_grid' and 'best_random'
                                 subset option, list of CV MSE for each model type, 
                                 allows for identification of 'best' models. 
                                 Defaults to None.
        n_random_iter (int, optional): Only needed for 'best_random' subset option, 
                                       number of random iterations to identify
                                       'best' models via random grid search. 
                                       Defaults to None.
        num_models (int, optional): Only needed for 'best' subset option,
                                    number of each model type to be retained. 
                                    Defaults to None.

    Returns:
        list: indices of relevant models
    """

    # get list of model types
    model_type = [type(model).__name__ for model in model_list]

    # get indices for best(re: MSE) models via full grid search
    if subset == 'best_grid':

         # collect model type and cv_mse into  dataframe
        all_cv_mse = pd.DataFrame({'type': model_type,
                                   'mse': cv_mse})

        # identify indices for (num_models) models with minimum mse for each model type
        multi_indices = all_cv_mse.groupby('type')['mse'].nsmallest(num_models).index
        indices = multi_indices.get_level_values(1).values.tolist()

    # get indices for best(re: MSE) models via random grid search
    if subset == 'best_random':

         # collect model type and cv_mse into  dataframe
        all_cv_mse = pd.DataFrame({'type': model_type,
                                   'mse': cv_mse})

        # identify random subset based on number of random iterations
        all_cv_mse = all_cv_mse.sample(frac = 1.0).groupby('type').head(n_random_iter)                           


        # identify indices for (num_models) models with minimum mse for each model type
        multi_indices = all_cv_mse.groupby('type')['mse'].nsmallest(num_models).index
        indices = multi_indices.get_level_values(1).values.tolist()

    # get indices for default models
    if subset == 'default':

        # get hyperparameter specs for each base model
        hp = [model.get_params() for model in model_list]

        # combine into dataframe for identifying indices
        model_hp = pd.DataFrame(list(zip(model_type, hp)), 
                    columns =['model_type', 'hp']) 

        # get default hyperparameter settings for each model type
        default_hp = []
        default_model_types = list(set(model_type))
        for mod in default_model_types:
            default_model = eval(mod + '()')
            default_hp.append(default_model.get_params())

        # get indices for models with default hyperperameters
        indices = []
        for i in range(len(default_model_types)):
            indices.append(model_hp.index[(model_hp['model_type'] == default_model_types[i]) & (model_hp['hp'] == default_hp[i])].values[0])

    # sort indices
    indices.sort()

    return indices

def subsetMetaX(meta_X, indices):
    """Subset meta_X columns using indices provided

    Args:
        meta_X (numpy array): out-of-sample predictions
        indices (list): indices for relevant base models

    Returns:
        numpy array: subset meta_X
    """

    # subset best models for each model type
    meta_X_subset = meta_X[:, indices]

    # return  subset meta_X
    return meta_X_subset

def subsetModels(model_list, indices):
    """Subset model objects using indices provided

    Args:
        model_list (list): list of model objects
        indices (list): indices for relevant base models

    Returns:
        numpy array: subset list of model objects
    """

    # subset models
    models_subset = [model_list[i] for i in indices]

    # return best models
    return models_subset

# fit relevant base models on the training dataset
def fitBaseModels(X, y, models, indices):
    """Fit specified base models to provided X and y data

    Args:
        X (numpy array): feature matrix
        y (numpy array): outcome vector
        models (list): list of model objects
        indices (list): indices for relevant base models
    """
    # iterate through base models
    for i in indices:

        # fit model
        models[i].fit(X, y)

# fit meta model (non-negative least squares)
def fitMetaModel(X, y, nz = True):
    """Fit SuperLearner meta model using non-negative least squares

    Args:
        X (numpy array): feature matrix (out-of-sample predictions)
        y (numpy array): outcome vector
        nz (bool, optional): Also returns compressed non-zero coefficients. 
                             Defaults to True.

    Returns:
        numpy array(s): full vector of NNLS coefficients and, if indicated,
                        compressed vector of non-zero NNLS coefficients.
    """

    # obtain nnls coefficients and scale to sum to one
    nnls_coef = nnls(X, y)[0]
    nnls_coef = nnls_coef/nnls_coef.sum()
    

    # if nz == True, return full vector as well as only non-zero coefficients
    if nz == True:
        nnls_coef_nonzero = nnls_coef[np.nonzero(nnls_coef)]
        return nnls_coef, nnls_coef_nonzero

    # if nz == False, return only full vector of coefficients
    if nz == False:
        return nnls_coef

# evaluate base models
def evaluateBaseModels(X, y, models, indices):
    """Evaluate perfrmance of specified base models using 
       provided X, y

    Args:
        X (numpy array): feature matrix
        y (numpy array): outcome vector
        models (list): list of model objects
        indices (list): indices for model objects to evaluate

    Returns:
        numpy array: array of validation MSE for specified base models
    """

    # initialize list for discrete mse results
    discrete_mse = []

    # iterate over base models
    for idx in indices:

         # predict y 
        yhat = models[idx].predict(X)

        # calculate and append mse 
        discrete_mse.append(mean_squared_error(y, yhat))

    # return discrete mse results
    return np.array(discrete_mse)

def superLearnerPredictions(X, models, meta_coef):
    """Predict yhat using SuperLearner, using provided base models
       and meta-coefficients

    Args:
        X (numpy array): feature matrix
        models (list): list of relevant model objects
        meta_coef (numpy array): SuperLearner meta-coefficients

    Return:
        numpy array: vector of SuperLearner predictions (y-hat)
    """

    # initialize list to hold base model predictions
    meta_X = list()

    # iterate over models
    for model in models:

        # predict y
        yhat = model.predict(X)

        # collect predictions as columns
        meta_X.append(yhat.reshape(len(yhat),1))

    # convert predictions into array
    meta_X = np.hstack(meta_X)

	# return predictions
    return np.dot(meta_coef, meta_X.T)       

def getGeneticIndices(meta_X, meta_y):
    """Obtain indices of base models selected using genetic algorithm
       to minimize SuperLearner MSE

    Args:
        meta_X (numpy array): out-of-sample base model predictions
        meta_y (numpy array): obvserved outcome vector

    Returns:
        list: base model indices selected using genetic algorithm
    """

    # initiate 10-fold cross validation
    kf_ga = KFold(n_splits=10)

    # define fitness function
    def fitness(individual, data):
        # note this uses meta_X and meta_Y which are not passed into this function
        
        # get indices for individual 
        ga_indices = [i for i, x in enumerate(individual) if x == 1]

        # subset meta_X columns (i.e., base models) using ga indices
        meta_X_ga_i = subsetMetaX(meta_X, ga_indices)

        # intialize list for mse for 10 test folds
        ga_cv_mse = []

        # obtain cross-validated mse
        for train_index, test_index in kf_ga.split(meta_X_ga_i):

            # subset train and test data
            meta_X_ga_i_train, meta_X_ga_i_test = meta_X_ga_i[train_index], meta_X_ga_i[test_index]
            meta_y_train, meta_y_test = meta_y[train_index], meta_y[test_index]

            # fit meta model on training data
            meta_coef_ga_i = fitMetaModel(meta_X_ga_i_train, meta_y_train, nz = False)

            # evaluate performance (i.e., obtain MSE) on test data
            fitness_i = mean_squared_error(meta_y_test, 
                                           np.dot(meta_coef_ga_i, 
                                                  meta_X_ga_i_test.T))

            # append test mse for each fold
            ga_cv_mse.append(fitness_i)
        
        # obtain mean mse across folds
        fitness = np.mean(ga_cv_mse)

        return fitness

    # define arbitrary data of appropriate dimensions
    data = list(range(meta_X.shape[1]))

    # initialize genetic algorithm
    ga = pyeasyga.GeneticAlgorithm(data, population_size=meta_X.shape[1],
                                   mutation_probability=0.01,
                                   maximise_fitness=False)

    # specify fitness function
    ga.fitness_function = fitness

    # run genetic algorithm
    ga.run()

    # get final ga indices
    final_ga_indices = [i for i, x in enumerate(ga.best_individual()[1]) if x == 1]

    # return indices
    return final_ga_indices

def identifyNonZeroIndices(coefs, og_indices):
    """Compress indices such that only indices that are used in SuperLearner meta
       algorithm are obtained (i.e., those with non-zero meta coefficients)

    Args:
        coefs (numpy array): meta coefficients from SuperLearner fit
        og_indices ([type]): original list of indices for particular strategy

    Returns:
        list: compressed list of indices that only include indices for models used
              in SuperLearner meta algorithm.
    """

    # identify indices of non-zero coefficients
    coef_indices = [idx for idx, value in enumerate(coefs) if value != 0]

    # identify relevant all_models indices
    model_indices = [og_indices[x] for x in coef_indices]

    # sort
    model_indices.sort()

    # return
    return model_indices

def runSingleFold(X_train, y_train, X_test, y_test, all_models, dataset_name, fold):

    # get out of sample predictions and cv mse for each base model
    print('--Getting CV MSE and predictions for all base learners...')
    meta_X, meta_y, cv_mse = getMSEandPredictions(X_train, y_train, all_models)

    # get indices for each hyperparameter strategy
    print('--Getting model indices for each strategy...')

    # initialize function arguments to get each set of indices
    mod_index_params = [{'model_list': all_models, 'subset': 'best_grid', 
                         'cv_mse': cv_mse, 'num_models': 1},
                        {'model_list': all_models, 'subset': 'best_grid', 
                         'cv_mse': cv_mse, 'num_models': 3},
                        {'model_list': all_models, 'subset': 'best_grid', 
                         'cv_mse': cv_mse, 'num_models': 5},
                        {'model_list': all_models, 'subset': 'best_random', 
                         'cv_mse': cv_mse, 'n_random_iter': 10, 'num_models': 1},    
                        {'model_list': all_models, 'subset': 'best_random', 
                         'cv_mse': cv_mse, 'n_random_iter': 10, 'num_models': 3},  
                        {'model_list': all_models, 'subset': 'best_random', 
                         'cv_mse': cv_mse, 'n_random_iter': 10, 'num_models': 5},  
                        {'model_list': all_models, 'subset': 'best_random', 
                         'cv_mse': cv_mse, 'n_random_iter': 25, 'num_models': 1},  
                        {'model_list': all_models, 'subset': 'best_random', 
                         'cv_mse': cv_mse, 'n_random_iter': 25, 'num_models': 3},  
                        {'model_list': all_models, 'subset': 'best_random', 
                         'cv_mse': cv_mse, 'n_random_iter': 25, 'num_models': 5},                                                                                                                           
                         {'model_list': all_models, 'subset': 'best_random', 
                         'cv_mse': cv_mse, 'n_random_iter': 50, 'num_models': 1},  
                        {'model_list': all_models, 'subset': 'best_random', 
                         'cv_mse': cv_mse, 'n_random_iter': 50, 'num_models': 3},  
                        {'model_list': all_models, 'subset': 'best_random', 
                         'cv_mse': cv_mse, 'n_random_iter': 50, 'num_models': 5},       
                        {'model_list': all_models, 'subset': 'best_random', 
                         'cv_mse': cv_mse, 'n_random_iter': 100, 'num_models': 1},  
                        {'model_list': all_models, 'subset': 'best_random', 
                         'cv_mse': cv_mse, 'n_random_iter': 100, 'num_models': 3},  
                        {'model_list': all_models, 'subset': 'best_random', 
                         'cv_mse': cv_mse, 'n_random_iter': 100, 'num_models': 5},
                        {'model_list': all_models, 'subset': 'default'}]

    # get indices (i.e., models input in SuperLearner) for each tuning strategy
    # List Order: full, grid * 3, random * 12, default, genetic
    indices_list = [getModelIndices(**params) for params in mod_index_params]
    indices_list.insert(0, list(range(len(all_models)))) # all indices for "full" superlearner
    print('--Running genetic algorithm...')
    indices_list.append(getGeneticIndices(meta_X, meta_y)) # genetic indices

    # subset meta_X (i.e. base learner predictions) for each tuning strategy
    # List Order: full, grid * 3, random * 12, default, genetic
    print('--Subsetting base learner predictions for each strategy...')
    meta_X_list = [subsetMetaX(meta_X, inds) for inds in indices_list]

    # fit all SuperLearners and obtain coefficients (both full vector and only non-zero)
    # List Order: full, grid * 3, random * 12, default, genetic
    print('--Fitting SuperLearner for each strategy...')
    meta_coef_list, nz_meta_coef_list = zip(*[fitMetaModel(x, meta_y) for x in meta_X_list])
    meta_coef_list = list(meta_coef_list)
    nz_meta_coef_list = list(nz_meta_coef_list)

    # identify model indices corresponding to non-zero SuperLearner coefficients for each tuning strategy
    # List Order: linear, full, grid * 3, random * 12, detault, ga, discrete
    nonzero_mod_list = [identifyNonZeroIndices(x, y) for x, y in zip(meta_coef_list, indices_list)]
    nonzero_mod_list.insert(0, [0]) # add index for linear model
    nonzero_mod_list.append([np.argmin(cv_mse).astype(int)]) # add index for discrete superlearner

    # identify unique indices for models with non-zero SuperLearner coefficients
    all_indices = list(set([item for sublist in nonzero_mod_list for item in sublist]))
    all_indices.sort()

    # fit minimal base models (those with non-zero SuperLearner weights) on entire training dataset
    print('--Fitting minimal base models on entire training dataset...')
    fitBaseModels(X_train, y_train, all_models, all_indices)

    # evaluate minimal base models (those with non-zero SuperLearner weights) on test set
    print('--Evaluating performance...')
    val_mse_all_models = evaluateBaseModels(X_test, y_test, all_models, all_indices)

    # subset minimal models (those with non-zero SuperLearner weights) for each tuning strategy 
    # note: prediction and evaluation of linear model and discrete SuperLearner are handled
    #       separately from true SuperLearners
    # List Order: full, grid * 3, random * 12, detault, ga
    del nonzero_mod_list[0] # drop linear regression (requires only single model)
    del nonzero_mod_list[-1] # drop discrete superlearner (requires only single model)
    models_list = [subsetModels(all_models, x) for x in nonzero_mod_list]


    # obtain SuperLearner Predictions for each tuning strategy
    # List Order: full, grid * 3, random * 12, detault, ga (linear and discrete don't need this function)
    sl_yhat_list = [superLearnerPredictions(X_test, x, y) for x, y in zip(models_list, nz_meta_coef_list)]

    # evaluate performance of each SuperLearner against test data and save as list
    # note: adding in linear model and discrete SuperLearner performance seaprately
    # List Order: linear, full, grid * 3, random * 12, default, ga, discrete
    val_mse_list = [mean_squared_error(y_test, x) for x in sl_yhat_list]
    val_mse_list.insert(0, val_mse_all_models[0]) # linear regression
    val_mse_list.append(val_mse_all_models[all_indices.index(np.argmin(cv_mse))]) # discrete superlearner

    # add dataset and name to list of results
    val_mse_list.insert(0, dataset_name)
    val_mse_list.insert(1, fold)
    
    # organize meta data

    # indices of models included for each tuning strategy (i.e. models input into SuperLearners)
    # (Order: grid * 3, random * 12, default, genetic, discrete)

    # initialize names for each strategy of appropriate length
    names = ['gs1', 'gs3', 'gs5', 'r10s1', 'r10s3', 'r10s5',
             'r25s1', 'r25s3', 'r25s5', 'r50s1', 'r50s3', 'r50s5',
             'r100s1', 'r100s3', 'r100s5', 'df', 'ga']
    del indices_list[0] # delete full indices
    names_all = [[x] * len(y) for x, y in zip(names, indices_list)]
    names_all = [item for sublist in names_all for item in sublist]
    names_all.append('dsc') # add discrete to end

    # flatten indices of models included for each tuning strategy
    indices_sl_all = [item for sublist in indices_list for item in sublist]
    indices_sl_all.append(np.argmin(cv_mse)) # add discrete to end

    # initialize dataset name and fold
    dataset_name_all = [dataset_name] * len(indices_sl_all)
    fold_all = [fold] * len(indices_sl_all)

    # indices for models with non-zero weights and weight values (i.e. SuperLearners)
    # (Order: full, grid * 3, random * 12, default, genetic, discrete)

    # initialize names for each strategy of appropriate length
    names.insert(0, 'full') # add "full" to names list
    names_nonzero = [[x] * len(y) for x, y in zip(names, nonzero_mod_list)]
    names_nonzero = [item for sublist in names_nonzero for item in sublist]

    # flatten indices of models with non-zero weights for each tuning strategy
    indices_sl_nonzero = [item for sublist in nonzero_mod_list for item in sublist]

    # flatten coefficients of models with non-zero weights for each tuning strategy
    coef_sl_nonzero = [item for sublist in nz_meta_coef_list for item in sublist]
    
    # initialize dataset name and fold
    dataset_name_nonzero = [dataset_name] * len(indices_sl_nonzero)
    fold_nonzero = [fold] * len(indices_sl_nonzero)

    # combine various lists into two sets of meta data:
    # 1. sl_all : indices for models input into SuperLearner for each tuning strategy
    # 2. sl_nonzero : indices and weights for models with non-zero SuperLearner weights 
    #    for each tuning strategy
    sl_all = list(zip(dataset_name_all, fold_all, names_all, indices_sl_all))
    sl_nonzero = list(zip(dataset_name_nonzero, fold_nonzero, names_nonzero, indices_sl_nonzero, coef_sl_nonzero))

    # save within-fold cv_mse for each fold
    cv_mse = cv_mse.tolist()
    cv_mse.insert(0, dataset_name)
    cv_mse.insert(1, fold)

    # return results
    return val_mse_list, sl_all, sl_nonzero, cv_mse

def runEvaluationIteration(X, y, all_models, dataset_name, dataset_num):
    """Run single iteration of performance experiement

    Args:
        X (numpy array): Full feature matrix (incl. train and test)
        y (numpy array): Full outcome vector (incl. train and test)
        all_models (list): Full list of candidate base model objects 
        dataset_name (str): dataset name for saving meta-data

    Returns:
        list: validation MSE for linear model and each hyperparameter tuning
              strategy; specifically: [linear model, full SuperLearner,
              best per model type (via full grid search), 3 best per model type
              (via full grid search), 5 best per model type (via full grid search),
              genetic algorithm selection, default hyperparameters,
              discrete SuperLearner]
    """

    # print statement
    print("Working on dataset # " + str(dataset_num))

    # initialize empty results lists
    all_results_fold = []
    all_sl_all_fold = []
    all_sl_nonzero_fold = []
    all_cv_mse_fold = []

    # initiate 10-fold cross validation
    kf = KFold(n_splits=10)
    fold = 1
    for train_index, test_index in kf.split(X):

        # initiation statement
        print('--Starting Fold! Dataset: ' + dataset_name + '; Fold: ' + str(fold))

        # subset train and test data
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # run single fold
        fold_output = runSingleFold(X_train, y_train, X_test, y_test, all_models, dataset_name, fold)
        all_results_fold.append(fold_output[0])
        all_sl_all_fold.extend(fold_output[1])
        all_sl_nonzero_fold.extend(fold_output[2])
        all_cv_mse_fold.append(fold_output[3])      

        print('--Fold Complete! Dataset: ' + dataset_name + '; Fold: ' + str(fold))  
        fold = fold + 1

    # return results
    return all_results_fold, all_sl_all_fold, all_sl_nonzero_fold, all_cv_mse_fold
