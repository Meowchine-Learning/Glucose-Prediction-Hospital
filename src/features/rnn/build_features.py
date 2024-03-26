import pandas as pd
import numpy as np
from functools import partial


def aggregate_meds(med_admin, row):
    # Filter the meds dataframe for this ID and meds administered before this glucose measurement time
    filtered_meds = med_admin[(med_admin['STUDY_ID'] == row['STUDY_ID']) & (
        med_admin["ENCOUNTER_NUM"] == row["ENCOUNTER_NUM"]) & (med_admin['TAKEN_HRS_FROM_ADMIT'] <= row['RESULT_HRS_FROM_ADMIT'])]

    # Drop the meds that have been accounted for to prevent them from being included in future rows
    med_admin.drop(filtered_meds.index, inplace=True)

    # Return the list of medications, or an empty list if none match
    return list(filtered_meds['MEDICATION_ATC']) if not filtered_meds.empty else []


def write_to_csv(df_file, name):
    df_file.to_csv("data/"+name+".csv", index=None, header=True)


if __name__ == '__main__':

    encounters = pd.read_csv("data/processed_ENCOUNTERS.csv")
    admit = pd.read_csv("data/processed_ADMIT_DX.csv")
    labs = pd.read_csv("data/processed_LABS.csv", index_col=False)

    med_admin = pd.read_csv("data/processed_MEDICATION_ADMINISTRATIONS.csv")

    write_to_csv(med_admin, "MEDICATION_ADMINISTRATIONS.csv")

    # Apply the function to each row in the glucose dataframe
    labs['INSULIN_MEDS'] = labs.apply(
        lambda row: aggregate_meds(med_admin, row), axis=1)

    # Merge the dynamic and static features based on ID

    df_combined = pd.merge(labs, encounters, on=[
                           'STUDY_ID', 'ENCOUNTER_NUM'], how='left')
    df_combined = pd.merge(df_combined, admit, on=[
                           'STUDY_ID', 'ENCOUNTER_NUM'], how='left')

    write_to_csv(df_combined, "data_processed")
