import pandas as pd
import numpy as np

TIME_INTERVAL = 24
ATC_CUT_FLAG = False
ATC_CUT_LEN = 3


def split_time(time_column):
    split = np.ceil(time_column.div(TIME_INTERVAL)).astype(np.int32).to_frame()
    return split


def preprocess_ENCOUNTERS(filePath_ADMIT):
    df = pd.read_csv(filePath_ADMIT)
    df.drop(["HOSP_ADMSN_TOD", "HOSP_DISCHRG_TOD"],
            axis=1, inplace=True)
    return df


def preprocess_ADMIT(filePath_ADMIT):
    df = pd.read_csv(filePath_ADMIT)
    df.drop(['DIAGNOSIS_LINE'], axis=1, inplace=True)

    for index, row in df.iterrows():
        icds = row['CURRENT_ICD10_LIST'].split(',')
        if len(icds) > 1:
            for i in range(len(icds)):
                icds[i] = icds[i].strip()
                icd = icds[i].split('.')[0]

                if i == 0:
                    df.iloc[index, 2] = icd
                else:
                    row.loc[['CURRENT_ICD10_LIST']] = icd
                    new_row = row.to_frame().transpose()
                    df = pd.concat([df, new_row])
        else:
            icds = row['CURRENT_ICD10_LIST'].split('.')
            df.iloc[index, 2] = icds[0]

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
    write_to_csv(encounters, "processed_ENCOUNTERS")

    med_admin = preprocess_MEDICATION_ADMIN(
        "data/MEDICATION_ADMINISTRATIONS.csv")
    write_to_csv(med_admin, "processed_MEDICATION_ADMINISTRATIONS")

    admit = preprocess_ADMIT("data/ADMIT_DX.csv")
    write_to_csv(admit, "processed_ADMIT_DX")

    labs = preprocess_LABS("data/LABS.csv")
    write_to_csv(labs, "processed_LABS")
