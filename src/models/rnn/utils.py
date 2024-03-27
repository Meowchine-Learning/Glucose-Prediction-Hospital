import pandas as pd
import numpy as np
import ast


def create_data():
    lab_path = "./data/processed_labs.csv"
    encounters_path = "./data/processed_encounters.csv"

    labs = pd.read_csv(lab_path)
    encounters = pd.read_csv(encounters_path)

    # Keeping glucose values only
    labs['KeepRow'] = labs['COMPONENT_ID'].apply(
        lambda x: int(x.strip("[]").split(", ")[4]) == 1)
    filtered_labs = labs[labs['KeepRow']]
    filtered_labs = filtered_labs.drop(columns=['KeepRow'])

    # Left join for final data for modules
    df = pd.merge(filtered_labs, encounters, on=[
                  "STUDY_ID", "ENCOUNTER_NUM"], how="left")
    df = df.sort_values(by=['STUDY_ID', 'ENCOUNTER_NUM'], ascending=True)

    df = df.drop("COMPONENT_ID", axis=1)

    # Converting One-Hot string vectors to columns
    df['SEX'] = df['SEX'].apply(lambda x: ast.literal_eval(x)[
                                0] if pd.notnull(x) else None)

    # Save file
    df.to_csv("./models/rnn/data.csv", index=False)
# --------------------------------------------------------------------------------------------------


def seq_data(data_csv, window_size=15):
    counts = data_csv.groupby(['STUDY_ID', 'ENCOUNTER_NUM']).size()

    lstm_data = data_csv.to_numpy()
    frequencies = counts.to_numpy()
    X = []
    y = []

    current_index = 0
    for freq in frequencies:
        if freq > window_size:
            for i in range(freq-window_size):
                # all data from window_size number of previous meals
                row = [a for a in lstm_data[i +
                                            current_index:i+window_size+current_index]]
                X.append(row)
                # glucose level in the next meal
                label = lstm_data[i+window_size+current_index][4]
                y.append(label)
        current_index += freq
    return np.array(X), np.array(y)


def preprocess_numerical(X, X_train, np_column_list):
    for column in np_column_list:
        mean = np.mean(X_train[:, :, column])
        std = np.std(X_train[:, :, column])
        X[:, :, column] = (X[:, :, column] - mean) / std
    return X


def convert_time_to_seconds(df, column_name):
    df[column_name] = pd.to_timedelta(df[column_name])
    df[column_name] = df[column_name].dt.total_seconds()
    return df


def test_shape(np_matrix_1, np_matrix_2=[]):
    print()
    print("-----------------------------")
    print("Checking shapes of data")
    print(f"X: {np_matrix_1.shape}")
    if np_matrix_2 != []:
        print(f"y: {np_matrix_2.shape}")
    print("-----------------------------")
    print()


def categorical_to_binary(df, csv_column_list):
    # Pandas:
    # for column in csv_column_list:
    #     df = pd.get_dummies(df, columns = [column], drop_first=True)
    # return df

    df = df.categorize(columns=csv_column_list)

    for column in csv_column_list:
        # Assume the first category is what we want to encode as 1 (binary)
        # This step assigns 1 to the first category and 0 otherwise
        df[column + '_binary'] = df[column].map_partitions(
            lambda x: x.cat.codes, meta=('x', 'int64')
        )
        # Optionally, drop the original column if you no longer need it
        df = df.drop(columns=[column])
    return df
