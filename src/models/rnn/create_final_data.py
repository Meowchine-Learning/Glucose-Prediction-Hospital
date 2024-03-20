import pandas as pd
import numpy as np
import ast

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


counts = rnn_data.groupby(['STUDY_ID','ENCOUNTER_NUM']).size()
rnn_data.to_csv("models/rnn/rnn_data.csv",index=False)
#--------------------------------------------------------------------------------------------------

def lstm_data():
    window_size = 4
    lstm_data = rnn_data.to_numpy()
    frequencies = counts.to_numpy()
    X = []
    y = []

    current_index = 0
    for freq in frequencies:
        for i in range(freq-window_size):
            row = [[a] for a in lstm_data[i+current_index:i+window_size+current_index]] #all data from window_size number of previous meals
            X.append(row)
            label = lstm_data[i+window_size+current_index][4] #glucose level in the next meal
            y.append(label)
        current_index+=freq
    return np.array(X), np.array(y)
