import pandas as pd

lab_path = "data/processed_labs.csv"
admit_path = "data/processed_admit.csv"
encounters_path = "data/processed_encounters.csv"

labs = pd.read_csv(lab_path)
encounters = pd.read_csv(encounters_path)

labs['KeepRow'] = labs['COMPONENT_ID'].apply(lambda x: int(x.strip("[]").split(", ")[4]) == 1)
# Filter the DataFrame to contain Glucose Values only
filtered_labs = labs[labs['KeepRow']]
filtered_labs = filtered_labs.drop(columns=['KeepRow'])

rnn_data = pd.merge(filtered_labs,encounters, on=["STUDY_ID","ENCOUNTER_NUM"],how="left")
rnn_data = rnn_data.sort_values(by=['STUDY_ID','ENCOUNTER_NUM'], ascending=True)
rnn_data.to_csv("models/rnn/rnn_data.csv",index=False)