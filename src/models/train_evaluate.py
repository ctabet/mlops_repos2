import warnings

warnings.filterwarnings(action="ignore")

import os
import yaml
import mlflow
import joblib
import pandas as pd
import numpy as np
import mlflow.sklearn
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             classification_report, confusion_matrix, roc_curve, precision_recall_curve, auc)
class ConfigObject:
    def __init__(self, d):
        self.processed = None
        for key, value in d.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigObject(value))
            else:
                setattr(self, key, value)

with open("mlops_training_repo/config/main.yaml", "r") as stream:
    try:
        config_dict = yaml.safe_load(stream)
        config = ConfigObject(config_dict)
    except yaml.YAMLError as exc:
        print(exc)
def load_data(config: ConfigObject):
    # Accessing directory and file names from the config
    processed_dir = config.processed.dir
    x_train_name = config.processed.x_train.name
    y_train_name = config.processed.y_train.name
    x_test_name = config.processed.x_test.name
    y_test_name = config.processed.y_test.name

    # Constructing the file path
    x_train_path = os.path.join(processed_dir, x_train_name)
    y_train_path = os.path.join(processed_dir, y_train_name)
    x_test_path = os.path.join(processed_dir, x_test_name)
    y_test_path = os.path.join(processed_dir, y_test_name)

    x_train = pd.read_csv(x_train_path)
    y_train = pd.read_csv(y_train_path)
    x_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path)
    return x_train, y_train, x_test, y_test

def load_model_name_and_dir(config_file):
    with open(config_file, "r") as stream:
        config_dict = yaml.safe_load(stream)
        model_name = config_dict["model"]["name"]
        model_dir = config_dict["model"]["dir"]
        return model_name, model_dir

# Load model-specific parameters from model_params.yaml
def load_model_params(params_file, model_name):
    with open(params_file, "r") as stream:
        config_dict = yaml.safe_load(stream)
        model_params = next((item["params"] for item in config_dict["models"] if item["name"] == model_name), {})
        return model_params

def load_model():
    model_name, model_dir = load_model_name_and_dir("mlops_training_repo/config/main.yaml")
    model_params = load_model_params("mlops_training_repo/config/model/model.yaml", model_name)
    return model_name, model_params

def create_param_grid(param_specs):
    param_grid = {}
    for param, spec in param_specs.items():
        if isinstance(spec, dict) and all(key in spec for key in ['start', 'stop', 'num', 'dtype']):
            param_grid[param] = np.linspace(spec['start'], spec['stop'], spec['num'], dtype=spec['dtype']).tolist()
        elif isinstance(spec, list):
            param_grid[param] = spec
    return param_grid

def tune_train(model_name, model_params, x_train, y_train):
    # Fetch the model class based on the model name string using globals()
    model_class = globals().get(model_name)

    if model_class is None or not callable(model_class):
        raise ValueError(f"Model class '{model_name}' not found or not callable.")

    grid_search = GridSearchCV(model_class(**model_params), model_params, cv=5, scoring="accuracy")
    grid_model = grid_search.fit(x_train, y_train)
    return grid_model

def evaluate_model(config, grid_search, x_test, y_test):
    mlflow.set_tracking_uri(config.mlflow_tracking_ui)
    # mlflow.set_experiment("employee_churn")
    os.environ['MLFLOW_TRACKING_USERNAME'] = config.mlflow_USERNAME
    os.environ['MLFLOW_TRACKING_PASSWORD'] = config.mlflow_PASSWORD

    with mlflow.start_run():
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        # Predict on the test set
        y_pred = best_model.predict(x_test)
        y_proba = (best_model.predict_proba(x_test)[:, 1] if hasattr(grid_search, "predict_proba") else None)

        # Accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")

        # Precision
        precision = precision_score(y_test, y_pred)
        print(f"Precision: {precision:.4f}")

        # Recall
        recall = recall_score(y_test, y_pred)
        print(f"Recall: {recall:.4f}")

        # F1 Score
        f1 = f1_score(y_test, y_pred)
        print(f"F1 Score: {f1:.4f}")

        # ROC AUC Score
        if y_proba is not None:
            roc_auc = roc_auc_score(y_test, y_proba)
            print(f"ROC AUC Score: {roc_auc:.4f}")

        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # Confusion matrix
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        # ROC Curve (if applicable)
        if y_proba is not None:
            fpr, tpr, thresholds = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label="ROC curve (area = %0.2f)" % roc_auc)
            plt.plot([0, 1], [0, 1], "k--")  # Random guess line
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("Receiver Operating Characteristic (ROC) Curve")
            plt.legend(loc="lower right")
            plt.savefig(os.path.join(config.figures.dir, "roc_curve_chart.png"))
            plt.show()
        # Precision-Recall Curve (if applicable)
        if y_proba is not None:
            precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, label="Precision-Recall curve")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("Precision-Recall Curve")
            plt.legend()
            plt.savefig(os.path.join(config.figures.dir,"precision_recall_curve_chart.png"))
            plt.show()

        # Save model
        processed_dir = config.model.dir
        model_name = config.model.name
        model_path = os.path.join(processed_dir, model_name)
        joblib.dump(best_model, model_path)

        # Log metrics
        mlflow.log_metrics({'f1': f1, 'accuracy': accuracy})
        mlflow.log_params(best_params)
        # Log the model
        mlflow.sklearn.log_model(best_model, str(model_name))

def train_evaluate():
    x_train, y_train, x_test, y_test = load_data(config)
    model, model_params = load_model()
    model_params = create_param_grid(model_params)
    grid_search = tune_train(model, model_params, x_train, y_train)
    evaluate_model(config, grid_search, x_test, y_test)

if __name__ == '__main__':
    train_evaluate()