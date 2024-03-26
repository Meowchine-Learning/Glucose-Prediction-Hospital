import pandas as pd
import numpy as np

TIME_INTERVAL = 24
ATC_CUT_FLAG = False
ATC_CUT_LEN = 3


def encoding(name, df, column_list):
    for column in column_list:
        if column == "CURRENT_ICD10_LIST":
            # Step 1: Split comma-separated codes and explode into separate rows
            df_exploded = df.assign(CURRENT_ICD10_LIST=df[column].str.split(
                ', ')).explode(column)

            # Step 2: One-hot encode the exploded DataFrame
            df_one_hot = pd.get_dummies(
                df_exploded, columns=[column], prefix='', prefix_sep='')

            # Step 3: Aggregate the one-hot encoded rows back to the original row structure
            # Summing up the one-hot encoded columns to ensure rows with multiple codes have two 1's
            df = df_one_hot.groupby(
                ['STUDY_ID', 'ENCOUNTER_NUM'], as_index=False).sum().apply(list)

        elif column == "SEX":
            df = pd.get_dummies(df, columns=[column], drop_first=True)
        else:
            df = pd.get_dummies(df, columns=[column], prefix=column)
        # Write DataFrame to CSV
        write_to_csv(df, name)


def preprocess_ENCOUNTERS(filePath_ADMIT):
    df = pd.read_csv(filePath_ADMIT)
    df.drop(["HOSP_ADMSN_TOD", "HOSP_DISCHRG_TOD", "HOSP_DISCHRG_HRS_FROM_ADMIT"],
            axis=1, inplace=True)
    return df


def preprocess_ADMIT(filePath_ADMIT):
    df = pd.read_csv(filePath_ADMIT)
    df.drop(['DIAGNOSIS_LINE'], axis=1, inplace=True)

    def process_icd10_codes(cell):
        # Split codes by comma
        codes = cell.split(', ')
        # Keep only the first 3 characters of each code
        truncated_codes = [code[:3] for code in codes]
        # Rejoin the truncated codes with commas
        return ', '.join(truncated_codes)

    # Apply the function to the ICD10 column
    df['CURRENT_ICD10_LIST'] = df['CURRENT_ICD10_LIST'].apply(
        process_icd10_codes)
    df = df.groupby(['STUDY_ID', 'ENCOUNTER_NUM'], as_index=False)[
        ['STUDY_ID', 'ENCOUNTER_NUM', 'CURRENT_ICD10_LIST']].agg({'CURRENT_ICD10_LIST': lambda x: ', '.join(x.unique())})

    return df


def preprocess_MEDICATION_ADMIN(filePath_MEDICATION_ADMIN):
    df = pd.read_csv(filePath_MEDICATION_ADMIN)

    insulin_names = ["A10AB01", "A10AB04", "A10AB04",
                     "A10AC01", "A10AD04", "A10AE04", "A10AE05", "A10AE06"]
    # dataframe with only rows with those meds
    df = df[df['MEDICATION_ATC'].isin(insulin_names)]

    if ATC_CUT_FLAG:
        for index, row in df.iterrows():
            atc = row['MEDICATION_ATC'][:ATC_CUT_LEN]
            df.iloc[index, 2] = atc

    return df


def preprocess_LABS(filePath_LABS):
    df = pd.read_csv(filePath_LABS)

    # convert ord value to numeric
    # df.at[20188, "ORD_VALUE"] = '33.3'
    # df.at[30111, "ORD_VALUE"] = '0.6'
    # df.at[66064, "ORD_VALUE"] = '0.6'
    # df.at[74868, "ORD_VALUE"] = '33.3'
    # df.at[82221, "ORD_VALUE"] = '33.3'
    # df.at[85492, "ORD_VALUE"] = '33.3'
    # ord_value = pd.to_numeric(df.loc[:, "ORD_VALUE"]).to_frame()
    # df["ORD_VALUE"] = ord_value["ORD_VALUE"].values

    return df


def write_to_csv(df_file, name):
    df_file.to_csv("data/"+name+".csv", index=None, header=True)


if __name__ == '__main__':
    """ Data Structure 
      Group by STUDY_ID, then group by ENCOUNTER_NUM
      Do need to indicate which meal it is for the model input?
      Consider having hourly time intervals to account for many measurements throughout the day (probably not)
      Consider keeping lab values as constant to account from the last time they were measured
      NOTES: there are about 3.3 glucose measurements/24 hours across all patients, hourly might not be necessary? Might be more efficient to strcuture by mealtime
      772 patients out of the 803 are get glucose labs checked (exclude the rest?)
      TODO: Figure out how to handle LABS after asking Anna
      dataPreprocessed =
       Hour |  BG  |   INSULIN_ADMINISTRATION [ATC,SIG] |   OR_PROC_ID  |   CURRENT_ICD10_LIST  |   AGE  |  HEIGHT_CM  |  WEIGHT_KG  |  SEX  |
      ----- |------|------------------------------------|---------------|-----------------------|--------|-------------|-------------|--------
        1   |  54  |         [[0, 1, 0,..], 2]          |  [0, 0, 1...] |       [0,1,0...]      |   46   |     175     |     98      |   1   |
          ...
  """

    encounters = preprocess_ENCOUNTERS("data/ENCOUNTERS.csv")
    encoding("processed_ENCOUNTERS", encounters, ["SEX"])

    med_admin = preprocess_MEDICATION_ADMIN(
        "data/MEDICATION_ADMINISTRATIONS.csv")
    encoding("processed_MEDICATION_ADMINISTRATIONS", med_admin, [
             "MEDICATION_ATC"])

    admit = preprocess_ADMIT("data/ADMIT_DX.csv")
    encoding("processed_ADMIT_DX", admit, ["CURRENT_ICD10_LIST"])

    labs = preprocess_LABS("data/LABS.csv")
    encoding("processed_LABS", labs, ["MEAL"])
