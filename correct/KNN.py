import os                                              # for creating folders, file, etc.
import time                                            # to track time
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split   # splitting data
from sklearn.preprocessing import StandardScaler                        # for scaling
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             precision_recall_curve, average_precision_score)  # the metrics
from imblearn.pipeline import Pipeline as ImbPipeline     # to create a pipeline of steps
from imblearn.over_sampling import SMOTE                  # smote
from sklearn.neighbors import KNeighborsClassifier        # knn
from optuna.integration import OptunaSearchCV             # optuna
from optuna.distributions import IntDistribution          # for int distributions in optuna
import random                                             # python tool for randomness

SEED = 12345                                                # for consistent results
random.seed(SEED)                                           # set the seed
np.random.seed(SEED)                                        # NumPys random number generator


### Load data
df = pd.read_csv(r"C:\Users\sroes\PycharmProjects\PythonProject\model_input_thesis.csv")  # load the data
df.dropna(inplace=True)                                         # Remove rows that contain missing values
df = pd.get_dummies(df, columns=['gender'], drop_first=True)    # one-hot encoding
target_col = 'Conversion'                                       # target column
X = df.drop(columns=[target_col])                               # Set X to all the input features (everything except the column we're trying to predict)
y = df[target_col]                                              # Set y to just the column we are trying to predict (conversion outcome)

### Functions
def build_pipeline(scenario_steps, base_model):         # builds the steps of the pipeline for consistency, optuna integration, and avoids leakage because preprocessing only happens on training folds during CV
    return ImbPipeline(scenario_steps + [('clf', base_model)])  # output example is [('scaler', StandardScaler()), ('smote', SMOTE()), ('clf', KNeighborsClassifier())]

def get_optimal_threshold(y_val, y_probs):              # to find the best treshold for highest f1
    precision, recall, thresholds = precision_recall_curve(y_val, y_probs)  # precision and recall for thresholds
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)  # F1 score, small number is added to avoid issues.
    best_thresh = thresholds[np.argmax(f1_scores)]      # threshold where the F1 score is highest
    return best_thresh

def evaluate_model(name, model, X_test, y_test, X_val, y_val):  # function to evaluate the model
    y_val_probs = model.predict_proba(X_val)[:, 1]      # predicted probabilities for validation set
    best_thresh = get_optimal_threshold(y_val, y_val_probs)  # use the optimal treshold function to get the best threshold based on validation set

    y_probs = model.predict_proba(X_test)[:, 1]         # predict probabilities on test set (final performance set) index on 1 which is the positive class
    y_pred = (y_probs >= best_thresh).astype(int)       # convert probabilities to 0/1 using the best threshold (.astype(int) converts boolean to 0/1)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()  # calculate the confusion matrix (it makes a 2x2 matrix ravels it into 1 with 4 values

    return {                                            # Return all evaluation scores on the final test set
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
        'AUC': roc_auc_score(y_test, y_probs),
        'PR AUC': average_precision_score(y_test, y_probs),
        'Threshold': best_thresh,
        'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn
    }

### Main KNN function
def run_knn(X, y, output_dir="results/KNN"):            # the main function that runs everything for KNN
    model_name = 'KNN'
    base_model = KNeighborsClassifier()                 # define base model

    include_cols = ['left1000', 'left2000', 'left4000', 'right1000', 'right2000', 'right4000',
                    'AgeAtTest', 'pta_left_avg', 'pta_right_avg', 'pta_avg']  # columns we scale

    param_dist = {                                      # define the parameters we want to optimize with optuna
        'clf__n_neighbors': IntDistribution(1, 30)
    }

    scenarios = {                                       # define the different preprocessing scenarios and its steps
        'Plain': [],
        'Scaled': [('scaler', 'manual')],
        'Smote': [('smote', SMOTE(random_state=SEED))],
        'Scaled + Smote': [('smote', SMOTE(random_state=SEED)), ('scaler', 'manual')],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)  # split data into 5 folds while keeping class balance similar
    all_results = []                                    # intialize a list to store results from each run
    os.makedirs(output_dir, exist_ok=True)              # Make a folder to save the results if it doesn't already exist

    for scen_name, steps in scenarios.items():          # loop through each scenario setup

        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):   # Loop through each train/test split
            X_train_full, X_test = X.iloc[train_idx], X.iloc[test_idx]      # Get the actual training and test data by indexing
            y_train_full, y_test = y.iloc[train_idx], y.iloc[test_idx]
            X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=SEED)  # Split part of training into validation set

            if 'scaler' in dict(steps):                # when scaler is in the specific scenario
                scaler = StandardScaler()              # scaler that will normalize values
                X_train_scaled = X_train.copy()        # copies to store scaled versions
                X_val_scaled = X_val.copy()
                X_test_scaled = X_test.copy()

                # scale only on the features used for training and not include questionnaire which is already ordinally encoded (include_cols)
                X_train_scaled[include_cols] = scaler.fit_transform(X_train[include_cols])  # X_train is training data
                X_val_scaled[include_cols] = scaler.transform(X_val[include_cols])          # X_val is validation data for hyperparameter tuning
                X_test_scaled[include_cols] = scaler.transform(X_test[include_cols])        # x_test is the final test data to evaluate the model after tuning
            else:
                X_train_scaled = X_train                 # If no scaling, just use data as-is
                X_val_scaled = X_val
                X_test_scaled = X_test

            pipeline = build_pipeline([step for step in steps if step[0] != 'scaler'], base_model)  # Build the pipeline using the function defined earlier
            # 'steps' is the list ('smote', SMOTE()).
            # Each 'step' is a where step[0] is the name (e.g., 'scaler') and step[1] is the instruction of what we want to do
            # scaler is filtered out from the steps because we handle scaling manually above this is because only certain features need to be scaled, in this way it was easier
            # The remaining steps (SMOTE if applicable) are combined with the base model using build_pipeline() defined function earlier
            # The final pipeline is a sequence of preprocessing steps followed by the KNN classifier.


            start = time.time()                          # Start the clock to see how long training takes
            search = OptunaSearchCV(                     # define the Optuna search for hyperparameter tuning
                pipeline, param_distributions=param_dist, n_trials=100,
                cv=StratifiedKFold(5), scoring='f1', random_state=SEED, n_jobs=2  # 5-fold stratified cross-validation, f1 chooses best model
            )
            search.fit(X_train_scaled, y_train)          # optuna searches for the best hyperparameters using the training split, theres internal cross validation
            best_pipe = search.best_estimator_           # best model found by optuna within a scenario
            best_params = search.best_params_            # store the best parameters, in this case its the best K

            elapsed = round(time.time() - start, 2)      # the time it took (its still in the fold loop so it will be the time for each fold)
            result = evaluate_model(                     # Get all the evaluation scores for the model with the defined evaluate_model function
                f"{model_name}_{scen_name}_fold{fold_idx}", best_pipe, X_test_scaled, y_test, # here we name the model with the model name, scenario name, and fold index, and give the necessary data. use the best parameters within the scenario (and fold) to evaluate
                X_val_scaled, y_val
            )
            y_train_probs = best_pipe.predict_proba(X_train_scaled)[:, 1]       # Get predictions on training set
            train_preds = (y_train_probs >= result['Threshold']).astype(int)    # final predictions in training set with best treshold
            train_f1 = f1_score(y_train, train_preds)                           # F1 score for training set
            result['Train F1'] = train_f1                   # save the training F1 score in the result for this fold in scenario
            result['Val F1'] = result['F1 Score']           # save the validation F1 score in the result for this fold in scenario

            result['Model Type'] = model_name               # Add info about model
            result['Scenario'] = scen_name                  # scenario
            result['Fold'] = fold_idx                       # fold
            result['Time (s)'] = elapsed                    # time of fold
            result['Best Params'] = best_params             # best parameters

            all_results.append(result)                      # Saving this run’s results by adding it to the full results list

    results_df = pd.DataFrame(all_results)                  # makes a table out of the dictionary of fold results
    results_df.to_csv(os.path.join(output_dir, f"{model_name}_results.csv"), index=False)  # Save the final table to a CSV file

run_knn(X, y)                                       # Start running everything, X = input features, y = target column defined in the beginning
