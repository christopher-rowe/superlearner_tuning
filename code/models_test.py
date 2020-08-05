# import libraries
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor

# initialize models & hyperparameter grids (ensure that default parameters are included)
model_specs = {'ols': [LinearRegression(), {}],
                'ols_lasso': [Lasso(), 
                                {'alpha': [0.1, 0.5, 1.0, 5, 
                                           10, 50, 100, 500, 1000]}],                     
                'ols_ridge': [Ridge(), 
                                {'alpha': [0.1, 0.5, 1.0, 5, 
                                           10, 50, 100, 500, 1000]}], 
                'ols_elasticnet': [ElasticNet(), 
                                    {'alpha': [0.1, 1.0, 10, 100, 1000],
                                    'l1_ratio': [0.1, 0.15, 0.2, 0.25,
                                                 0.3, 0.35, 0.4, 0.45,
                                                 0.5, 0.55, 0.6, 0.65,
                                                 0.7, 0.75, 0.8, 0.85,
                                                 0.9, 0.95]}], 
                'random_forest': [RandomForestRegressor(), 
                                    {'max_depth': [None, 100],
                                    'max_features': ['auto', 'log2'],
                                    'n_estimators': [100, 500]}],    
                'xgboost': [GradientBoostingRegressor(), 
                            {'learning_rate': [0.1, 0.3],
                             'n_estimators': [100],
                            'subsample': [1.0],
                            'min_samples_split': [2, 4],
                            'min_samples_leaf': [1],
                            'max_depth': [3],
                            'max_features': [None]}],  
                'knn': [KNeighborsRegressor(), 
                        {'n_neighbors': [3,5,7,11],
                        'weights': ['uniform', 'distance']}]}   


                   