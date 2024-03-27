#----------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Some libraries:

'''Math libraries: '''
# import pandas as pd
import numpy as np
import os
import dask.dataframe as dd
import pandas as pd

'''Plotting libraries: '''
import matplotlib.pyplot as plt

'''Tensorflow libraries for GRU model: '''
import tensorflow as tf
from keras.models import Sequential
from keras.models import load_model
from keras.layers import *
from keras.callbacks import ModelCheckpoint
from keras.losses import MeanAbsoluteError, MeanSquaredError
from keras.metrics import MeanAbsoluteError, MeanSquaredError
from keras.optimizers import Adam

'''Local file'''
from utils import *          #uncomment for main data-------------------------------------------------------------------------------------------------------------
#Creating data for model:

#Load data:
print()
print("LOADING DATA ...")
df_path = "./data/data_processed.csv"                       #uncomment for main data
df = pd.read_csv(df_path)                       #uncomment for main data
print("LOADING DONE")
print()

#Preprocess data if needed
print()
print("CONVERTING TOD ...")
df = convert_time_to_seconds(df,"RESULT_TOD")
print("CONVERTING DONE")
print()

#Creating Sequential data
print("CREATING SEQUENTIAL DATA...")
X,y = seq_data(df, window_size=4)  
print("SEQUENTIAL DATA DONE")                     #uncomment for main data
test_shape(X,y)

val1 = 8000                #uncomment for main data
val2 = 10000               #uncomment for main data
X_train, y_train = X[:val1],y[:val1]                #uncomment for main data
X_val, y_val = X[val1:val2],y[val1:val2]                #uncomment for main data
X_test, y_test = X[val2:],y[val2:]                #uncomment for main data


print("PREPROCESSING NUMERICAL DATA...")
numerical_list = [2,3,4,15,16,17]
X_train = preprocess_numerical(X_train,X_train, numerical_list)
X_val = preprocess_numerical(X_val,X_train, numerical_list)
X_test = preprocess_numerical(X_test, X_train, numerical_list)
print("PREPROCESSING NUMERICAL DONE")
print()

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Creating GRU model:

gru_model = Sequential()
gru_model.add(InputLayer((X.shape[1],X.shape[2])))
gru_model.add(GRU(3,activation="tanh",recurrent_activation="sigmoid", return_sequences= False))
gru_model.add(Dropout(rate=0.2))
gru_model.add(Dense(1))

gru_model.summary()

cp = ModelCheckpoint('models/gru_model/', save_best_only = True)
gru_model.compile(loss=MeanAbsoluteError(), optimizer = Adam(learning_rate = 0.0001), metrics = [MeanAbsoluteError()])

gru_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, callbacks=[cp])
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Testing:

'''Loading Model: '''
gru_model = load_model("models/gru_model/")

'''Performance on training data: '''
train_predictions = gru_model.predict(X_train).flatten()
train_results = pd.DataFrame(data={"Train Predictions":train_predictions, 'Actuals': y_train})

plt.plot(train_results['Train Predictions'][:100])
plt.plot(train_results['Actuals'][:100])

'''Performance on validation data: '''
val_predictions = gru_model.predict(X_val).flatten()
val_results = pd.DataFrame(data={"Train Predictions":val_predictions, 'Actuals': y_val})

plt.plot(val_results['Train Predictions'][:100])
plt.plot(val_results['Actuals'][:100])

'''Performance on test data: '''
test_predictions = gru_model.predict(X_test).flatten()
test_results = pd.DataFrame(data={"Train Predictions":test_predictions, 'Actuals': y_test})

plt.plot(test_results['Train Predictions'][:100])
plt.plot(test_results['Actuals'][:100])