import pandas as pd
import numpy as np
import ast



def create_data():
    lab_path = "data/processed_labs.csv"
    encounters_path = "data/processed_encounters.csv"

    labs = pd.read_csv(lab_path)
    encounters = pd.read_csv(encounters_path)

    #Keeping glucose values only
    labs['KeepRow'] = labs['COMPONENT_ID'].apply(lambda x: int(x.strip("[]").split(", ")[4]) == 1)
    filtered_labs = labs[labs['KeepRow']]
    filtered_labs = filtered_labs.drop(columns=['KeepRow'])

    #Left join for final data for modules
    rnn_data = pd.merge(filtered_labs,encounters, on=["STUDY_ID","ENCOUNTER_NUM"],how="left")
    rnn_data = rnn_data.sort_values(by=['STUDY_ID','ENCOUNTER_NUM'], ascending=True)

    rnn_data = rnn_data.drop("COMPONENT_ID",axis=1)

    #Converting One-Hot string vectors to columns
    rnn_data['SEX'] = rnn_data['SEX'].apply(lambda x: ast.literal_eval(x)[0] if pd.notnull(x) else None)

    #Save file
    rnn_data.to_csv("models/rnn/rnn_data.csv",index=False)
#--------------------------------------------------------------------------------------------------

def seq_data(data_csv):
    counts = data_csv.groupby(['STUDY_ID','ENCOUNTER_NUM']).size()

    lstm_data = data_csv.to_numpy()
    frequencies = counts.to_numpy()
    X = []
    y = []

    current_index = 0
    window_size = 20
    for freq in frequencies:
        if freq > window_size:
            for i in range(freq-window_size):
                row = [a for a in lstm_data[i+current_index:i+window_size+current_index]] #all data from window_size number of previous meals
                X.append(row)
                label = lstm_data[i+window_size+current_index][4] #glucose level in the next meal
                y.append(label)
        current_index+=freq
    return np.array(X), np.array(y)

def preprocess_numerical(X,X_train):
    for i in range(X.shape[2]):
        mean = np.mean(X_train[:, :, i])
        std = np.std(X_train[:, :, i])
        X[:, :, i] = (X[:, :, i] - mean) / std
        return X
    
def test_shape(np_matrix_1, np_matrix_2 = []):
    print()
    print("-----------------------------")
    print("Checking shapes of data")
    print(f"X: {np_matrix_1.shape}")
    if np_matrix_2 != []:
        print(f"y: {np_matrix_2.shape}")
    print("-----------------------------")
    print()

        