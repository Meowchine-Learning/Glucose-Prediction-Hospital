import pandas as pd
import numpy as np
import csv


def calc_avg_mealtime(filePath_LABS):
    LABS_df = pd.read_csv(filePath_LABS, index_col=0)
    col = LABS_df.loc[:, "RESULT_TOD"]

    out = list(col[col.str.contains("^([5-9]|10|11):.*$")])
    print(out[10:20])


def determine_meal(time):
    pass


def preprocess_ENCOUNTERS(filePath_ENCOUNTERS):
    data_ENCOUNTERS = pd.read_csv(filePath_ENCOUNTERS, index_col=0)
    df = data_ENCOUNTERS.sort_values(by=['STUDY_ID', 'ENCOUNTER_NUM'],
                                     ascending=[True, True])


if __name__ == '__main__':
    """ Data Structure 
      Group by STUDY_ID, then group by ENCOUNTER_NUM
      Do need to indicate which meal it is for the model input?
      Consider having hourly time intervals to account for many measurements throughout the day (probably not)
      Consider keeping lab values as constant to account from the last time they were measured
      TODO: Figure out how to handle LABS after asking Anna
      dataPreprocessed =
       Meal time |  BG  |   INSULIN_ADMINISTRATION [ATC,SIG] |   OR_PROC_ID  |   CURRENT_ICD10_LIST  |   AGE  |  HEIGHT_CM  |  WEIGHT_KG  |  SEX  |
      ---------- |------|------------------------------------|---------------|-----------------------|--------|-------------|-------------|--------
       Breakfast |  54  |         [[0, 1, 0,..], 2]          |  [0, 0, 1...] |       [0,1,0...]      |   46   |     175     |     98      |   1   |
          ...
  """

    preprocess_ENCOUNTERS("data/ENCOUNTERS.csv")
    calc_avg_mealtime("data/LABS.csv")
