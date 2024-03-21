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


def _dataInput_npy(inputPath="../../features/output/FormalizedDATA.npy"):
    data = np.load(inputPath, allow_pickle=True)
    if data.shape[1] < 2:
        raise ValueError("The data must contain at least two columns for sampleID and y value.")

    uniqueSampleIDs = data[:, 0]
    y = np.array(data[:, 1:17], dtype=np.float32)
    X = np.array(data[:, 17:], dtype=np.float32)

    return uniqueSampleIDs, X, y


if __name__ == '__main__':
    _, X, y = _dataInput_npy()

    training_window_size = 250
    evaluation_window_size = 50
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