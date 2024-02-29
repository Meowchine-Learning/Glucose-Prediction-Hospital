import pandas as pd
import csv


# Todo - Sample time window creator
def createTimeWindows(windowSize: int = 10, dataFileName=None):
    data = pd.read_csv(dataFileName)

    for i in range(1, windowSize + 1):
        data[f'window{i}'] = data['value'].shift(i)

    X = data[[f'window{i}' for i in range(1, windowSize + 1)]]
    y = data['value']
