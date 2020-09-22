# import libraries
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor

# initialize models & hyperparameter grids (ensure that default parameters are included)
model_specs = {'ols': [LinearRegression(), {}],
                'ols_lasso': [Lasso(), 
                                {'alpha': [0.0001, 0.00025, 0.0005, 0.00075, 
                                           0.001, 0.0025, 0.005, 0.0075, 0.01, 
                                           0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 
                                           0.75, 1.0],
                                 'max_iter': [100000]}],                     
                'ols_ridge': [Ridge(), 
                                {'alpha': [0.0001, 0.00025, 0.0005, 0.00075, 
                                           0.001, 0.0025, 0.005, 0.0075, 0.01, 
                                           0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 
                                           0.75, 1.0, 2.5, 5, 7.5, 10, 25,
                                           50],   
                                 'max_iter': [100000]}], 
                'ols_elasticnet': [ElasticNet(), 
                                    {'alpha': [0.0001, 0.00025, 0.0005, 0.00075, 
                                           0.001, 0.0025, 0.005, 0.0075, 0.01, 
                                           0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 
                                           0.75, 1.0],    
                                     'l1_ratio': [0.1, 0.2, 0.3, 0.4, 
                                                 0.5, 0.6, 0.7, 0.8, 
                                                 0.9],
                                      'max_iter': [100000]}], 
                'random_forest': [RandomForestRegressor(), 
                                    {'max_depth': [None, 3, 5, 7],
                                     'max_features': ['auto', 'sqrt'],
                                     'min_samples_split': [2, 5, 10],
                                     'n_estimators': [100, 500, 1000, 2000]}],    
                'xgboost': [GradientBoostingRegressor(), 
                            {'learning_rate': [0.01, 0.1, 0.5],
                             'n_estimators': [100, 500, 1000],
                             'subsample': [1.0, 0.75, 0.5],
                             'min_samples_split': [2, 5, 10],
                             'max_depth': [3, 5, 7],
                             'max_features': [None, 'sqrt']}],  
                'knn': [KNeighborsRegressor(), 
                        {'n_neighbors': [3,5,7,10,15,20],
                        'weights': ['uniform', 'distance']}]}   

