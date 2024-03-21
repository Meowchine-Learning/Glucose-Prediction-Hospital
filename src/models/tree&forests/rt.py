import csv
import json

import pandas as pd
from ast import literal_eval
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

FEATURES = [
    "UniqueSampleID",
    "LabTests",
    "#Day",
    "#Time",
    "Med",
    "Activity",
    "Nutrition",
    "Weight",
    "Height",
    "Age",
    "Sex",
    "Operations",
    "MedActs",
    "Diseases",
    "PriorMed"
]


def _dataInput_csv(inputPath="../../features/output/FormalizedDATA.csv"):
    x_col_idx = list(range(2, 15))
    y_col_idx = 1
    X = []
    y = []

    with open(inputPath, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)

        for row in csvreader:
            x_values = []
            for i in x_col_idx:
                if row[i].startswith('[') and row[i].endswith(']'):
                    x_values.append(literal_eval(row[i]))
                else:
                    x_values.append(row[i])
            X.append(x_values)

            y_value = row[y_col_idx]
            if y_value.startswith('[') and y_value.endswith(']'):
                y.append(literal_eval(y_value))

    return X, y


def _dataInput_json(inputPath="../../features/output/FormalizedDATA.json"):
    print(f"\n>> Got Preprocessed Data from {inputPath}.")
    with open(inputPath, 'r') as f:
        INPUT = json.load(f)
        X, y = [], []
        for uniqueSampleID in INPUT.keys():
            X.append([
                INPUT[uniqueSampleID][FEATURES[2]],
                INPUT[uniqueSampleID][FEATURES[3]],
                INPUT[uniqueSampleID][FEATURES[4]],
                INPUT[uniqueSampleID][FEATURES[5]],
                INPUT[uniqueSampleID][FEATURES[6]],
                INPUT[uniqueSampleID][FEATURES[7]],
                INPUT[uniqueSampleID][FEATURES[8]],
                INPUT[uniqueSampleID][FEATURES[9]],
                INPUT[uniqueSampleID][FEATURES[10]],
                INPUT[uniqueSampleID][FEATURES[11]],
                INPUT[uniqueSampleID][FEATURES[12]],
                INPUT[uniqueSampleID][FEATURES[13]],
                INPUT[uniqueSampleID][FEATURES[14]]
            ])
            y.append(map(float, INPUT[uniqueSampleID][FEATURES[1]].split("_")))
    return X, y


if __name__ == '__main__':
    X, y = _dataInput_csv()
    #X, y = _dataInput_json()

    training_window_size = 250
    evaluation_window_size = 1
    rolling_step = 1

    performance_metrics = []
    for start_point in tqdm(range(0, len(X) - training_window_size, rolling_step)):
        end_point = start_point + training_window_size
        evaluation_end_point = end_point + evaluation_window_size

        X_train = X[start_point:end_point]
        y_train = y[start_point:end_point]
        X_eval = X[end_point:evaluation_end_point]
        y_eval = y[end_point:evaluation_end_point]

        model = DecisionTreeRegressor()
        model.fit(X_train, y_train)
        predictions = model.predict(X_eval)

        mse = mean_squared_error(y_eval, predictions)
        performance_metrics.append(mse)

    average_performance = np.mean(performance_metrics)
    print(f"Average MSE over all evaluation windows: {average_performance}")