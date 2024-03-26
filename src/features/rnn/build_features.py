import pandas as pd
import numpy as np


def aggregate_meds(med_admin, row, atc_columns):
    # Filter the meds dataframe for this ID and meds administered before this glucose measurement time
    filtered_meds = med_admin[(med_admin['STUDY_ID'] == row['STUDY_ID']) & (
        med_admin["ENCOUNTER_NUM"] == row["ENCOUNTER_NUM"]) & (med_admin['TAKEN_HRS_FROM_ADMIT'] <= row['RESULT_HRS_FROM_ADMIT'])]

    # Sum the one-hot encodings for filtered meds
    summed_meds = filtered_meds[atc_columns].sum().clip(upper=1)

    # Drop the meds that have been accounted for to prevent them from being included in future rows
    med_admin.drop(filtered_meds.index, inplace=True)

    # Return the summed one-hot encodings as a Series (or empty Series if no meds match)
    return summed_meds if not summed_meds.empty else pd.Series(0, index=atc_columns)


def write_to_csv(df_file, name):
    df_file.to_csv("data/"+name+".csv", index=None, header=True)


if __name__ == '__main__':

    encounters = pd.read_csv("data/processed_ENCOUNTERS.csv")
    admit = pd.read_csv("data/processed_ADMIT_DX.csv")
    labs = pd.read_csv("data/processed_LABS.csv", index_col=False)

    med_admin = pd.read_csv("data/processed_MEDICATION_ADMINISTRATIONS.csv")

    write_to_csv(med_admin, "MEDICATION_ADMINISTRATIONS.csv")

    atc_columns = [col for col in med_admin.columns if 'MEDICATION_ATC' in col]
    for col in atc_columns:
        labs[col] = 0

    for index, row in labs.iterrows():
        # Use a copy to avoid changing med_admin directly
        summed_meds = aggregate_meds(med_admin.copy(), row, atc_columns)
        labs.loc[index, atc_columns] = summed_meds.values

    # Merge the dynamic and static features based on ID

    df_combined = pd.merge(labs, encounters, on=[
                           'STUDY_ID', 'ENCOUNTER_NUM'], how='left')
    df_combined = pd.merge(df_combined, admit, on=[
                           'STUDY_ID', 'ENCOUNTER_NUM'], how='left')

    write_to_csv(df_combined, "data_processed")
