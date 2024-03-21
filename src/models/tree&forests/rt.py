import csv
from ast import literal_eval
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from tqdm import tqdm


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
            else:
                y.append(y_value)

    return X, y


if __name__ == '__main__':
    X, y = _dataInput_csv()

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