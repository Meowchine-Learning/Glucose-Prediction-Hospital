#----------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Some libraries:

'''Math libraries: '''
import pandas as pd
import numpy as np

'''Plotting libraries: '''
import matplotlib.pyplot as plt

'''Tensorflow libraries for GRU model: '''
import tensorflow as tf
from keras.models import Sequential
from keras.models import load_model
from keras.layers import *
from keras.callbacks import ModelCheckpoint
from keras.losses import MeanSquaredError
from keras.metrics import RootMeanSquaredError
from keras.optimizers import Adam
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Creating data for model:

csv_path = None #path to main csv file
df = pd.read_csv(csv_path)

def df_to_X_y(df, window_size=12):
    df_as_np = df.to_numpy()
    X = []
    y = []

    for i in range(len(df_as_np)-window_size):
        row = [[a] for a in df_as_np[i:i+window_size]] #all data from window_size number of previous meals
        X.append(row)
        label = df_as_np[i+window_size("glucose_level")] #glucose level in the next meal
        y.append(label)
    return np.array(X), np.array(y)

X, y = df_to_X_y(df)
X.shape, y.shape

val1 = None
val2 = None
X_train, y_train = X[:val1],y[:val1]
X_val, y_val = X[val1:val2],y[val1:val2]
X_test, y_test = X[val2],y[val2:]
X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Creating LSTM model:

lstm_model = Sequential()
lstm_model.add(LSTM(64, activation='relu', input_shape=(X_train.shape(1), X_train.shape(2)), return_sequences = True))
lstm_model.add(LSTM(32, activation='relu', return_sequences = False))
lstm_model.add(Dropout(rate=0.2))
lstm_model.add(Dense(1, "linear"))

lstm_model.summary()

cp = ModelCheckpoint('models/lstm_model/', save_best_only = True)
lstm_model.compile(loss=MeanSquaredError(), optimizer = Adam(learning_rate = 0.0001), metrics = [RootMeanSquaredError()])

lstm_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, callbakcs=[cp])
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Testing:

'''Loading Model: '''
lstm_model = load_model("models/lstm_model/")

'''Performance on training data: '''
train_predictions = lstm_model.predict(X_train).flatten()
train_results = pd.DataFrame(data={"Train Predictions":train_predictions, 'Actuals': y_train})

plt.plot(train_results['Train Predictions'][:100])
plt.plot(train_results['Actuals'][:100])

'''Performance on validation data: '''
val_predictions = lstm_model.predict(X_val).flatten()
val_results = pd.DataFrame(data={"Train Predictions":val_predictions, 'Actuals': y_val})

plt.plot(val_results['Train Predictions'][:100])
plt.plot(val_results['Actuals'][:100])

'''Performance on test data: '''
test_predictions = lstm_model.predict(X_test).flatten()
test_results = pd.DataFrame(data={"Train Predictions":test_predictions, 'Actuals': y_test})

plt.plot(test_results['Train Predictions'][:100])
plt.plot(test_results['Actuals'][:100])