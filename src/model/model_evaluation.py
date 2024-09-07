import numpy as np
import pandas as pd

import pickle
import json

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score

# Load Model
def load_model(model_path: str):

    return pickle.load(open(model_path,'rb'))

# Load Data
def load_data(url: str) -> pd.DataFrame:
    test_data = pd.read_csv(url)
    return test_data

# Calculate evaluation metrics
def calculate_metrics(y_test, y_pred, y_pred_proba):
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    return accuracy, precision, recall, auc


def main():

    clf = load_model("C:/Data Analyst/CampusX Study Material/MLOPS/Practice Modules/versioning-demo/models/model.pkl")

    test_data = load_data('./data/processed/test_bow.csv')

    X_test = test_data.iloc[:,0:-1].values
    y_test = test_data.iloc[:,-1].values

    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]

    accuracy, precision, recall, auc = calculate_metrics(y_test=y_test, y_pred=y_pred, y_pred_proba=y_pred_proba)

    metrics_dict={
    'accuracy':accuracy,
    'precision':precision,
    'recall':recall,
    'auc':auc
}
    
    with open('metrics.json', 'w') as file:
        json.dump(metrics_dict, file, indent=4)

if __name__ == '__main__':
    main()