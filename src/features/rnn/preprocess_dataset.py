import pandas as pd
import numpy as np
import csv

TIME_INTERVAL = 24

'''
def calc_avg_mealtime(filePath_LABS):
    LABS_df = pd.read_csv(filePath_LABS, index_col=0)
    col = LABS_df.loc[:, "RESULT_TOD"]

    out = list(col[col.str.contains("^([5-9]|10|11):.*$")])
'''


def split_time(time):
    pass


def preprocess_ENCOUNTERS(filePath_ENCOUNTERS):
    data_ENCOUNTERS = pd.read_csv(filePath_ENCOUNTERS, index_col=0)
    df = data_ENCOUNTERS.sort_values(by=['STUDY_ID', 'ENCOUNTER_NUM'],
                                     ascending=[True, True])
    hospital_stay = df.loc[:, "HOSP_DISCHRG_HRS_FROM_ADMIT"]
    hospital_intvs = np.ceil(hospital_stay.div(
        TIME_INTERVAL)).astype(np.int32)
    rows = hospital_intvs.to_numpy().sum()
    dataset = [[] for i in range(1, rows)]
    return dataset


def preprocess_ADMIT():
    pass


def preprocess_MEDICATION_ADMIN(dataset, filePath_MEDICATION_ADMIN):
    data_MEDICATION_ADMIN = pd.read_csv(filePath_MEDICATION_ADMIN, index_col=0)
    df = data_MEDICATION_ADMIN.sort_values(by=['STUDY_ID', 'ENCOUNTER_NUM'],
                                           ascending=[True, True])


def preprocess_LABS(dataset, filePath_LABS):
    data_LABS = pd.read_csv(filePath_LABS, index_col=0)
    df = data_LABS.sort_values(by=['STUDY_ID', 'ENCOUNTER_NUM'],
                               ascending=[True, True])


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

    dataset = preprocess_ENCOUNTERS("data/ENCOUNTERS.csv")
    preprocess_LABS(dataset, "data/LABS.csv")
    # calc_avg_mealtime("data/LABS.csv")
