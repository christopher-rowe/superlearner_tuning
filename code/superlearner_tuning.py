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

def getModelIndices(model_list, subset, cv_mse = None, num_models = None):
    """Get model indices to allow subsetting of models and out-of-sample
       predictions

    Args:
        model_list (list): list of all model objects
        subset (str): 'best' to obtain model(s) with best MSE of each
                       model type; 'default' for default hyperparameters
        cv_mse (list, optional): Only needed for 'best' subset option,
                                 list of CV MSE for each model type, allows
                                 for identification of 'best' models. 
                                 Defaults to None.
        num_models (int, optional): Only needed for 'best' subset option,
                                    number of each model type to be retained. 
                                    Defaults to None.

    Returns:
        list: indices of relevant models
    """

    # get list of model types
    model_type = [type(model).__name__ for model in model_list]

    # get indices for best(re: MSE) models
    if subset == 'best':

         # collect model type and cv_mse into  dataframe
        all_cv_mse = pd.DataFrame({'type': model_type,
                                'mse': cv_mse})

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

    # obtain nnls coefficients
    nnls_coef = nnls(X, y)[0]

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

def getBestRandomIndices(meta_X, meta_y, cv_mse, n_iterations, num_models):
    """Obtain indices of combination of base models that provide lowest
       SuperLearner MSE, select base models with probability inversely 
       proportional to CV MSE

    Args:
        meta_X (numpy array): out-of-sample predictions
        meta_y (numpy array): observed outcome vector
        cv_mse (list): CV MSE for each base model
        n_iterations (int): Number of random iterations
        num_models ([type]): Number of base models to include in randomly
                             selected SuperLearner ('random' will randomly
                             vary the number of models included))

    Returns:
        list: indices for base models that provide best SuperLearner MSE
    """

    # initialize random selection probabilites (inversely proportional to CV MSE)
    model_prob = (1/cv_mse)/((1/cv_mse).sum())

    # initialize empty lists for storing random indices and training MSE
    all_random_indices = []
    all_train_mse_sl_random = []

    # iterate
    for _ in range(n_iterations):

        # if appropriate, select random number of models
        if num_models == 'random':
            num_models_i = np.random.choice(a =  np.arange(1, meta_X.shape[1]))
        else:
            num_models_i = num_models

        # draw random indicies with probability inversely proportional to CV MSE
        random_indices = np.random.choice(a = range(meta_X.shape[1]), 
                                          size = num_models_i,
                                          replace = False,
                                          p = model_prob)

        # save random indices
        all_random_indices.append(random_indices)

        # subset meta_X columns (i.e., base models) using random indices
        meta_X_random_subset = subsetMetaX(meta_X, random_indices)

        # fit meta model with only random base-models
        meta_coef_random_subset = fitMetaModel(meta_X_random_subset, 
                                               meta_y,
                                               nz = False)

        # calcualte MSE of meta model using random base learners
        train_mse_sl_random = mean_squared_error(meta_y, 
                                                 np.dot(meta_coef_random_subset, 
                                                        meta_X_random_subset.T))

        # save MSE
        all_train_mse_sl_random.append(train_mse_sl_random)
        
    # identiy index with lowest MSE 
    lowest_mse_index = [idx for idx, element in enumerate(all_train_mse_sl_random) if element == min(all_train_mse_sl_random)][0]

    # conver to to list and sort
    indices = all_random_indices[lowest_mse_index].tolist()
    indices.sort()

    # return best random indices
    return indices

def getGeneticIndices(meta_X, meta_y):
    """Obtain indices of base models selected using genetic algorithm
       to minimize SuperLearner MSE

    Args:
        meta_X (numpy array): out-of-sample base model predictions
        meta_y (numpy array): obvserved outcome vector

    Returns:
        list: base model indices selected using genetic algorithm
    """

    # define fitness function
    def fitness (individual, data):
        # note this uses meta_X and meta_Y which are not passed into this function
        
        # get indices for individual 
        ga_indices = [i for i, x in enumerate(individual) if x == 1]

        # subset meta_X columns (i.e., base models) using random indices
        meta_X_ga_individual = subsetMetaX(meta_X, ga_indices)

        # fit meta model with only random base-models
        meta_coef_ga_individual = fitMetaModel(meta_X_ga_individual, meta_y, nz = False)

        # calcualte MSE of meta model using random base learners
        fitness = mean_squared_error(meta_y, 
                                     np.dot(meta_coef_ga_individual, 
                                            meta_X_ga_individual.T))
                                        
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
    best_cv_mse_indices_1 = getModelIndices(model_list = all_models, 
                                            subset = 'best', cv_mse = cv_mse,
                                            num_models = 1)
    best_cv_mse_indices_3 = getModelIndices(model_list = all_models, 
                                            subset = 'best', cv_mse = cv_mse,
                                            num_models = 3)                                    
    best_cv_mse_indices_5 = getModelIndices(model_list = all_models, 
                                            subset = 'best', cv_mse = cv_mse,
                                            num_models = 5)
    default_indices = getModelIndices(model_list = all_models,
                                    subset = 'default')
    random_indices = getBestRandomIndices(meta_X, meta_y, 
                                        cv_mse, n_iterations = 1000,
                                        num_models = 'random')
    ga_indices = getGeneticIndices(meta_X, meta_y)

    # subset meta_X (i.e. base learner predictions) for each hyperparameter strategy
    print('--Subsetting base learner predictions for each strategy...')
    meta_X_best_subset_1 = subsetMetaX(meta_X, best_cv_mse_indices_1)
    meta_X_best_subset_3 = subsetMetaX(meta_X, best_cv_mse_indices_3)
    meta_X_best_subset_5 = subsetMetaX(meta_X, best_cv_mse_indices_5)    
    meta_X_default_subset = subsetMetaX(meta_X, default_indices)
    meta_X_ga_subset = subsetMetaX(meta_X, ga_indices)
    meta_X_random_subset = subsetMetaX(meta_X, random_indices)

    # Fit all SuperLearners and obtain coefficients (full vector and only non-zero)
    print('--Fitting SuperLearner for each strategy...')
    meta_coef_full, nz_meta_coef_full = fitMetaModel(meta_X, meta_y)
    meta_coef_best_subset_1, nz_meta_coef_best_subset_1  = fitMetaModel(meta_X_best_subset_1, meta_y)
    meta_coef_best_subset_3, nz_meta_coef_best_subset_3 = fitMetaModel(meta_X_best_subset_3, meta_y)
    meta_coef_best_subset_5, nz_meta_coef_best_subset_5 = fitMetaModel(meta_X_best_subset_5, meta_y)
    meta_coef_default_subset, nz_meta_coef_default_subset = fitMetaModel(meta_X_default_subset, meta_y)
    meta_coef_ga_subset, nz_meta_coef_ga_subset = fitMetaModel(meta_X_ga_subset, meta_y)
    meta_coef_random_subset, nz_meta_coef_random_subset = fitMetaModel(meta_X_random_subset, meta_y)

    # identify indices for models with non-zero SuperLearner coefficients
    nonzero_mod_linear = [0]
    nonzero_mod_full = identifyNonZeroIndices(meta_coef_full, list(range(len(all_models))))
    nonzero_mod_discrete = [np.argmin(cv_mse).astype(int)]
    nonzero_mod_best_subset_1 = identifyNonZeroIndices(meta_coef_best_subset_1, best_cv_mse_indices_1)
    nonzero_mod_best_subset_3 = identifyNonZeroIndices(meta_coef_best_subset_3, best_cv_mse_indices_3)
    nonzero_mod_best_subset_5 = identifyNonZeroIndices(meta_coef_best_subset_5, best_cv_mse_indices_5)
    nonzero_mod_default_subset = identifyNonZeroIndices(meta_coef_default_subset, default_indices)
    nonzero_mod_ga_subset = identifyNonZeroIndices(meta_coef_ga_subset, ga_indices)
    nonzero_mod_random_subset = identifyNonZeroIndices(meta_coef_random_subset, random_indices)

    # combine indices for all models with non-zero SuperLearner coefficients
    all_indices = list(set(nonzero_mod_linear + 
                        nonzero_mod_full + 
                        nonzero_mod_discrete +
                        nonzero_mod_best_subset_1 + 
                        nonzero_mod_best_subset_3 + 
                        nonzero_mod_best_subset_5 + 
                        nonzero_mod_default_subset +
                        nonzero_mod_ga_subset +
                        nonzero_mod_random_subset))
    all_indices.sort()

    # fit minimal base models on entire training dataset
    print('--Fitting minimal base models on entire training dataset...')
    fitBaseModels(X_train, y_train, all_models, all_indices)

    # subset minimal models for each strategy
    models_full = subsetModels(all_models, nonzero_mod_full)
    models_best_subset_1 = subsetModels(all_models, nonzero_mod_best_subset_1)
    models_best_subset_3 = subsetModels(all_models, nonzero_mod_best_subset_3)
    models_best_subset_5 = subsetModels(all_models, nonzero_mod_best_subset_5)
    models_default_subset = subsetModels(all_models, nonzero_mod_default_subset)
    models_ga_subset = subsetModels(all_models, nonzero_mod_ga_subset)
    models_random_subset = subsetModels(all_models, nonzero_mod_random_subset)

    # evalute miminal base models 
    print('--Evaluating performance...')
    val_mse_all_models = evaluateBaseModels(X_test, y_test, all_models, all_indices)

    # Obtain SuperLearner Predictions for each strategy
    sl_yhat_full = superLearnerPredictions(X_test, 
                                        models_full, 
                                        nz_meta_coef_full)
    sl_yhat_best_subset_1 = superLearnerPredictions(X_test, 
                                                    models_best_subset_1, 
                                                    nz_meta_coef_best_subset_1)
    sl_yhat_best_subset_3 = superLearnerPredictions(X_test, 
                                                    models_best_subset_3, 
                                                    nz_meta_coef_best_subset_3)
    sl_yhat_best_subset_5 = superLearnerPredictions(X_test, 
                                                    models_best_subset_5, 
                                                    nz_meta_coef_best_subset_5)
    sl_yhat_default_subset = superLearnerPredictions(X_test, 
                                                    models_default_subset,
                                                    nz_meta_coef_default_subset)
    sl_yhat_ga_subset = superLearnerPredictions(X_test, 
                                                models_ga_subset,
                                                nz_meta_coef_ga_subset)
    sl_yhat_random_subset = superLearnerPredictions(X_test, 
                                                    models_random_subset,
                                                    nz_meta_coef_random_subset)

    # evaluate performance of each SuperLearner against test data
    val_mse_linear_regression = val_mse_all_models[0]
    val_mse_sl_full = mean_squared_error(y_test, sl_yhat_full)
    val_mse_sl_best_subset_1 = mean_squared_error(y_test, sl_yhat_best_subset_1)
    val_mse_sl_best_subset_3 = mean_squared_error(y_test, sl_yhat_best_subset_3)
    val_mse_sl_best_subset_5 = mean_squared_error(y_test, sl_yhat_best_subset_5)       
    val_mse_sl_ga_subset = mean_squared_error(y_test, sl_yhat_ga_subset)    
    val_mse_sl_random_subset = mean_squared_error(y_test, sl_yhat_random_subset)
    val_mse_sl_default_subset = mean_squared_error(y_test, sl_yhat_default_subset)    
    val_mse_discrete_sl = val_mse_all_models[all_indices.index(np.argmin(cv_mse))]

    # concatenate performance results 
    iteration_results = [dataset_name, 
                         fold,
                         val_mse_linear_regression,
                         val_mse_sl_full,
                         val_mse_sl_best_subset_1,
                         val_mse_sl_best_subset_3,
                         val_mse_sl_best_subset_5,
                         val_mse_sl_ga_subset,
                         val_mse_sl_random_subset,
                         val_mse_sl_default_subset,
                         val_mse_discrete_sl]

    # organize meta data (i.e. indices and SuperLearner coefficients)
    indices_sl_all = []
    indices_sl_all.extend(best_cv_mse_indices_1)
    indices_sl_all.extend(best_cv_mse_indices_3)
    indices_sl_all.extend(best_cv_mse_indices_5)
    indices_sl_all.extend(ga_indices)
    indices_sl_all.extend(random_indices)
    indices_sl_all.extend(default_indices) 
    indices_sl_all.extend(nonzero_mod_discrete)

    names_all = []
    names_all.extend(["bs1"] * len(best_cv_mse_indices_1))
    names_all.extend(["bs2"] * len(best_cv_mse_indices_3))
    names_all.extend(["bs3"] * len(best_cv_mse_indices_5))
    names_all.extend(["ga"] * len(ga_indices))
    names_all.extend(["rnd"] * len(random_indices))
    names_all.extend(["df"] * len(default_indices))
    names_all.extend(["dsc"] * len(nonzero_mod_discrete))

    dataset_name_all = [dataset_name] * len(indices_sl_all)
    seed_all = [fold] * len(indices_sl_all)

    indices_sl_nonzero = []
    indices_sl_nonzero.extend(nonzero_mod_full)
    indices_sl_nonzero.extend(nonzero_mod_best_subset_1)
    indices_sl_nonzero.extend(nonzero_mod_best_subset_3)
    indices_sl_nonzero.extend(nonzero_mod_best_subset_5)
    indices_sl_nonzero.extend(nonzero_mod_ga_subset)
    indices_sl_nonzero.extend(nonzero_mod_random_subset)
    indices_sl_nonzero.extend(nonzero_mod_default_subset)

    coef_sl_nonzero = []
    coef_sl_nonzero.extend(nz_meta_coef_full)
    coef_sl_nonzero.extend(nz_meta_coef_best_subset_1)
    coef_sl_nonzero.extend(nz_meta_coef_best_subset_3)
    coef_sl_nonzero.extend(nz_meta_coef_best_subset_5)
    coef_sl_nonzero.extend(nz_meta_coef_ga_subset)
    coef_sl_nonzero.extend(nz_meta_coef_random_subset)
    coef_sl_nonzero.extend(nz_meta_coef_default_subset)

    names_nonzero = []
    names_nonzero.extend(["full"] * len(nonzero_mod_full))
    names_nonzero.extend(["bs1"] * len(nonzero_mod_best_subset_1))
    names_nonzero.extend(["bs3"] * len(nonzero_mod_best_subset_3))
    names_nonzero.extend(["bs5"] * len(nonzero_mod_best_subset_5))
    names_nonzero.extend(["ga"] * len(nonzero_mod_ga_subset))
    names_nonzero.extend(["rnd"] * len(nonzero_mod_random_subset))
    names_nonzero.extend(["df"] * len(nonzero_mod_default_subset))

    dataset_name_nonzero = [dataset_name] * len(indices_sl_nonzero)
    seed_nonzero = [fold] * len(indices_sl_nonzero)

    sl_all = list(zip(dataset_name_all, seed_all, names_all, indices_sl_all))
    sl_nonzero = list(zip(dataset_name_nonzero, seed_nonzero, names_nonzero, indices_sl_nonzero, coef_sl_nonzero))

    cv_mse = cv_mse.tolist()
    cv_mse.insert(0, dataset_name)
    cv_mse.insert(1, fold)

    return iteration_results, sl_all, sl_nonzero, cv_mse

def runEvaluationIteration(X, y, all_models, dataset_name):
    """Run single iteration of performance experiement

    Args:
        X (numpy array): Full feature matrix (incl. train and test)
        y (numpy array): Full outcome vector (incl. train and test)
        all_models (list): Full list of candidate base model objects 
        dataset_name (str): dataset name for saving meta-data

    Returns:
        list: validation MSE for linear model and each hyperparameter tuning
              strategy; specifically: [linear model, full SuperLearner,
              best per model type, 3 best per model type, 5 best per model type,
              genetic algorithm selection, random subset, default hyperparameters,
              discrete SuperLearner]
    """

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
