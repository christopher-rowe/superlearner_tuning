# SuperLearner Tuning
Examining the performance of different strategies for hyperparameter tuning using SuperLearner.

**NOTE: This repository includes code and an organizational structure that is under active development**

## Overview
This repository contains code (results pending) for a study of the finite sample predictive performance of various strategies for tuning hyperparameters and determining inclusion of base models in an ensemble learning algorithm called [SuperLearner](https://www.degruyter.com/view/journals/sagmb/6/1/article-sagmb.2007.6.1.1309.xml.xml).  Currently, only regression problems are considered, but binary classification problems might be added.

Briefly, the strategies being compared include:
- Include all models with all considered hyperpameter specifications as candidates (i.e. base learners) for inclusion in the SuperLearner algorithm.
- Identify subsets of the best performing hyperparameter specifications for each base learner algorithm (e.g., Lasso regression, Random Forest, etc.) using either a comprehensive grid search or random grid searches of varying size, and include only models with best performing specifications as candidates for inclusion in the SuperLearner algorithm.
- Include only models with scikit-learn default hyperparameter specifications as candidates for inclusion in the SuperLearner algorithm (as might be done in practice by analysts with little experience in hyperpamater tuning).
- Include a subset of models and hyperparameter specifications identified via a genetic algorithm with fitness corresponding to minimum cross-validated mean squared error of the SuperLearner algorithm

The above strategies are implemented on set of 98 benchmark datasets available via the Penn Machine Learning Benchmark ([PMLB](https://github.com/EpistasisLab/penn-ml-benchmarks)). Predictive performance is evaluated by mean squared error estimated via 10-fold cross-validation.

## Organization
- **\code**
  - superlearner_tuning.py : script with main study functions
  - models_test.py : hyperparameter specifications to be considered (testing subset)
  - models.py : hyperparameter specifications to be considered 
  - main.py : main study script for running on local machine
  - main_parallel.py : main study script for running in parallel on high performance computing cluster (HPCC)
- **\results**
  - **\raw**
  - **\processed**
- **\notebooks**
- **\reports**
