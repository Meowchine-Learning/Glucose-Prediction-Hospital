#----------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Some libraries:

'''Math libraries: '''
# import pandas as pd
import numpy as np
import os
import dask.dataframe as dd

'''Plotting libraries: '''
import matplotlib.pyplot as plt

'''Tensorflow libraries for GRU model: '''
import tensorflow as tf
from keras.models import Sequential
from keras.models import load_model
from keras.layers import *
from keras.callbacks import ModelCheckpoint
from keras.losses import MeanAbsoluteError
from keras.metrics import MeanAbsoluteError
from keras.optimizers import Adam

'''Local file'''
from utils import *          #uncomment for main data
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# #Create data if necessary:
# create_data()

#Load data:
print()
print("LOADING DATA ...")
df_path = "./models/rnn/combined_data.csv"                       #uncomment for main data
df = dd.read_csv(df_path)                       #uncomment for main data
print("LOADING DONE")
print()

#Preprocess data if necessary:
print("PREPROCESSING CATEGORICAL DATA ...")
df = df.sort_values(by=['STUDY_ID','ENCOUNTER_NUM','HOSP_DISCHRG_HRS_FROM_ADMIT'], ascending=True)
print("SORTING DONE")

numerical_list = [0,4,5,6,7,14,19,20]
one_hot_list = ["CURRENT_ICD10_LIST","MEDICATION_ATC","MAR_ACTION","DOSE_UNIT"]
binary_list = ["SEX", "ROUTE"]
chunk = categorical_to_hotcolumns(df, one_hot_list)
print("HOTCOLUMNS DONE")
chunk = categorical_to_binary(df, binary_list)
print("BINARY DONE")
print("PREPROCESSING CATEGORICAL DONE")
print()


#Creating Sequential data
print("CREATING SEQUENTIAL DATA...")
X,y = seq_data(df, window_size=15)  
print("SEQUENTIAL DATA DONE")                     #uncomment for main data
test_shape(X,y)
print("CREATING SEQUENTIAL DATA DONE")
print()

val1 = 12000                #uncomment for main data
val2 = 17000                #uncomment for main data
X_train, y_train = X[:val1],y[:val1]                #uncomment for main data
X_val, y_val = X[val1:val2],y[val1:val2]                #uncomment for main data
X_test, y_test = X[val2:],y[val2:]                #uncomment for main data


print("PREPROCESSING NUMERICAL DATA...")
X_train = preprocess_numerical(X_train,X_train, numerical_list)
X_val = preprocess_numerical(X_val,X_train, numerical_list)
X_test = preprocess_numerical(X_test, X_train, numerical_list)
print("PREPROCESSING NUMERICAL DONE")
print()


# #test data
# zip_path = tf.keras.utils.get_file(
#     origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
#     fname='jena_climate_2009_2016.csv.zip',
#     extract=True)
# csv_path, _ = os.path.splitext(zip_path)
# df = pd.read_csv(csv_path)

# df.index = pd.to_datetime(df['Date Time'], format='%d.%m.%Y %H:%M:%S')
# temp = df['T (degC)']

# def df_to_X_y(df, window_size=5):
#   df_as_np = df.to_numpy()
#   X = []
#   y = []
#   for i in range(len(df_as_np)-window_size):
#     row = [[a] for a in df_as_np[i:i+window_size]]
#     X.append(row)
#     label = df_as_np[i+window_size]
#     y.append(label)
#   return np.array(X), np.array(y)

# WINDOW_SIZE = 5
# X1, y1 = df_to_X_y(temp, WINDOW_SIZE)
# X_train, y_train = X1[:60000], y1[:60000]
# X_val, y_val = X1[60000:65000], y1[60000:65000]
# X_test, y_test = X1[65000:], y1[65000:]
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Creating LSTM model:
print("CREATING MODEL...")
print()
lstm_model = Sequential()
lstm_model.add(InputLayer((X.shape[1],X.shape[2])))
lstm_model.add(LSTM(64))
lstm_model.add(Dense(8, "relu"))
lstm_model.add(Dense(1, "linear"))

lstm_model.summary()

cp = ModelCheckpoint('./models/rnn/models/lstm_model/', save_best_only = True)
lstm_model.compile(loss=MeanAbsoluteError(), optimizer = Adam(learning_rate = 0.0001), metrics = [MeanAbsoluteError()])
lstm_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, callbacks=[cp])
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Testing:
print("TESTING MODEL ...")
print()

'''Loading Model: '''
lstm_model = load_model("./models/rnn/models/lstm_model/")

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