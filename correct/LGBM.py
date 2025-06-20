import os
import time
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             precision_recall_curve, average_precision_score)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from optuna.integration import OptunaSearchCV
from optuna.distributions import FloatDistribution, IntDistribution
import random

SEED = 12345
random.seed(SEED)
np.random.seed(SEED)

# load data
df = pd.read_csv(r"C:\Users\sroes\PycharmProjects\PythonProject\model_input_thesis.csv")
df.dropna(inplace=True)
df = pd.get_dummies(df, columns=['gender'], drop_first=True)
target_col = 'Conversion'
X = df.drop(columns=[target_col])
y = df[target_col]

# help functions
def build_pipeline(scenario_steps, base_model):
    return ImbPipeline(scenario_steps + [('clf', base_model)])

def get_optimal_threshold(y_val, y_probs):
    precision, recall, thresholds = precision_recall_curve(y_val, y_probs)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_thresh = thresholds[np.argmax(f1_scores)]
    return best_thresh

def evaluate_model(name, model, X_test, y_test, X_val, y_val):
    y_val_probs = model.predict_proba(X_val)[:, 1]
    best_thresh = get_optimal_threshold(y_val, y_val_probs)

    y_probs = model.predict_proba(X_test)[:, 1]
    y_pred = (y_probs >= best_thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    return {
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

# main functions
def run_lightgbm(X, y, output_dir="results/LGBM"):
    model_name = 'LightGBM'
    base_model = LGBMClassifier(random_state=SEED)

    include_cols = ['left1000', 'left2000', 'left4000', 'right1000', 'right2000', 'right4000',
                    'AgeAtTest', 'pta_left_avg', 'pta_right_avg', 'pta_avg']

    param_dist = {
        'clf__n_estimators': IntDistribution(10, 1000),
        'clf__learning_rate': FloatDistribution(0.01, 0.2),
        'clf__bagging_fraction': FloatDistribution(0.5, 0.95)
    }

    scenarios = {
        'Plain': [],
        'Scaled': [('scaler', 'manual')],
        'Smote': [('smote', SMOTE(random_state=SEED))],
        'Scaled + Smote': [('smote', SMOTE(random_state=SEED)), ('scaler', 'manual')],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    all_results = []
    os.makedirs(output_dir, exist_ok=True)

    for scen_name, steps in scenarios.items():

        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            X_train_full, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train_full, y_test = y.iloc[train_idx], y.iloc[test_idx]
            X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=SEED)

            if 'scaler' in dict(steps):
                scaler = StandardScaler()
                X_train_scaled = X_train.copy()
                X_val_scaled = X_val.copy()
                X_test_scaled = X_test.copy()

                X_train_scaled[include_cols] = scaler.fit_transform(X_train[include_cols])
                X_val_scaled[include_cols] = scaler.transform(X_val[include_cols])
                X_test_scaled[include_cols] = scaler.transform(X_test[include_cols])
            else:
                X_train_scaled = X_train
                X_val_scaled = X_val
                X_test_scaled = X_test

            pipeline = build_pipeline([step for step in steps if step[0] != 'scaler'], base_model)

            start = time.time()
            search = OptunaSearchCV(
                pipeline, param_distributions=param_dist, n_trials=100,
                cv=StratifiedKFold(5), scoring='f1', random_state=SEED, n_jobs=-1
            )
            search.fit(X_train_scaled, y_train)
            best_pipe = search.best_estimator_
            best_params = search.best_params_

            elapsed = round(time.time() - start, 2)
            result = evaluate_model(
                f"{model_name}_{scen_name}_fold{fold_idx}", best_pipe, X_test_scaled, y_test,
                X_val_scaled, y_val
            )

            y_train_probs = best_pipe.predict_proba(X_train_scaled)[:, 1]
            train_preds = (y_train_probs >= result['Threshold']).astype(int)
            train_f1 = f1_score(y_train, train_preds)
            result['Train F1'] = train_f1
            result['Val F1'] = result['F1 Score']

            result['Model Type'] = model_name
            result['Scenario'] = scen_name
            result['Fold'] = fold_idx
            result['Time (s)'] = elapsed
            result['Best Params'] = best_params

            # Save feature importances
            importances = best_pipe.named_steps['clf'].feature_importances_
            feature_df = pd.DataFrame({
                'Feature': X.columns,
                'Importance': importances
            }).sort_values(by='Importance', ascending=False)
            feature_df['Normalized'] = feature_df['Importance'] / feature_df['Importance'].sum()
            feature_df.to_csv(
                os.path.join(output_dir, f"{model_name}_{scen_name}_fold{fold_idx}_importance.csv"),
                index=False
            )
            all_results.append(result)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(output_dir, f"{model_name}_results.csv"), index=False)

# Run the function directly
run_lightgbm(X, y)