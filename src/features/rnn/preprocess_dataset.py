import pandas as pd
import numpy as np
import csv

TIME_INTERVAL = 24
ATC_CUT_FLAG = True
ATC_CUT_LEN = 3

'''
def calc_avg_mealtime(filePath_LABS):
    LABS_df = pd.read_csv(filePath_LABS, index_col=0)
    col = LABS_df.loc[:, "RESULT_TOD"]

    out = list(col[col.str.contains("^([5-9]|10|11):.*$")])
'''


def split_time(time_column):
    split = np.ceil(time_column.div(TIME_INTERVAL)).astype(np.int32).to_frame()
    return split


def preprocess_ENCOUNTERS(filePath_ENCOUNTERS):
    df = pd.read_csv(filePath_ENCOUNTERS)
    # df = data_ENCOUNTERS.sort_values(by=['STUDY_ID', 'ENCOUNTER_NUM'], ascending=[True, True])

    # convert time of day to fraction of day
    hosp_adm = df.loc[:, "HOSP_ADMSN_TOD"]
    hosp_adm_hrs = pd.to_numeric(hosp_adm.str.slice(stop=-6)).mul(60)
    hosp_adm_mins = hosp_adm.str.slice(stop=-3)
    hosp_adm_mins = pd.to_numeric(hosp_adm_mins.str.slice(start=-2)).add(hosp_adm_hrs).div(1440).to_frame()
    df["HOSP_ADMSN_TOD"] = hosp_adm_mins["HOSP_ADMSN_TOD"].values

    hosp_dis = df.loc[:, "HOSP_DISCHRG_TOD"]
    hosp_dis_hrs = pd.to_numeric(hosp_dis.str.slice(stop=-6)).mul(60)
    hosp_dis_mins = hosp_dis.str.slice(stop=-3)
    hosp_dis_mins = pd.to_numeric(hosp_dis_mins.str.slice(start=-2)).add(hosp_dis_hrs).div(1440).to_frame()
    df["HOSP_DISCHRG_TOD"] = hosp_dis_mins["HOSP_DISCHRG_TOD"].values

    # divide total hours by time interval
    hospital_hrs = df.loc[:, "HOSP_DISCHRG_HRS_FROM_ADMIT"]
    hospital_intvs = split_time(hospital_hrs)
    df["HOSP_DISCHRG_HRS_FROM_ADMIT"] = hospital_intvs["HOSP_DISCHRG_HRS_FROM_ADMIT"].values

    # rows = hospital_intvs.to_numpy().sum()
    # dataset = [[] for i in range(1, rows)]

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
    # df = data_MEDICATION_ADMIN.sort_values(by=['STUDY_ID', 'ENCOUNTER_NUM'],ascending=[True, True])
    
    # convert time of day to fraction of day
    taken_tod = df.loc[:, "TAKEN_TOD"]
    taken_tod_hrs = pd.to_numeric(taken_tod.str.slice(stop=-6)).mul(60)
    taken_tod_mins = taken_tod.str.slice(stop=-3)
    taken_tod_mins = pd.to_numeric(taken_tod_mins.str.slice(start=-2)).add(taken_tod_hrs).div(1440).to_frame()
    df["TAKEN_TOD"] = taken_tod_mins["TAKEN_TOD"].values

    # divide total hours by time interval
    taken_hrs = df.loc[:, "TAKEN_HRS_FROM_ADMIT"]
    taken_intvs = split_time(taken_hrs)
    df["TAKEN_HRS_FROM_ADMIT"] = taken_intvs["TAKEN_HRS_FROM_ADMIT"].values

    if ATC_CUT_FLAG:
        for index, row in df.iterrows():
            atc = row['MEDICATION_ATC'][:ATC_CUT_LEN]
            df.iloc[index, 2] = atc

    return df


def preprocess_LABS(filePath_LABS):
    df = pd.read_csv(filePath_LABS)
    # df = data_LABS.sort_values(by=['STUDY_ID', 'ENCOUNTER_NUM'],ascending=[True, True])

    # convert time of day to fraction of day
    result_tod = df.loc[:, "RESULT_TOD"]
    result_tod_hrs = pd.to_numeric(result_tod.str.slice(stop=-6)).mul(60)
    result_tod_mins = result_tod.str.slice(stop=-3)
    result_tod_mins = pd.to_numeric(result_tod_mins.str.slice(start=-2)).add(result_tod_hrs).div(1440).to_frame()
    df["RESULT_TOD"] = result_tod_mins["RESULT_TOD"].values

    # divide total hours by time interval
    result_hrs = df.loc[:, "RESULT_HRS_FROM_ADMIT"]
    result_intvs = split_time(result_hrs)
    df["RESULT_HRS_FROM_ADMIT"] = result_intvs["RESULT_HRS_FROM_ADMIT"].values

    # convert ord value to numeric
    df.at[20188, "ORD_VALUE"] = '33.3'
    df.at[30111, "ORD_VALUE"] = '0.6'
    df.at[66064, "ORD_VALUE"] = '0.6'
    df.at[74868, "ORD_VALUE"] = '33.3'
    df.at[82221, "ORD_VALUE"] = '33.3'
    df.at[85492, "ORD_VALUE"] = '33.3'
    ord_value = pd.to_numeric(df.loc[:, "ORD_VALUE"]).to_frame()
    df["ORD_VALUE"] = ord_value["ORD_VALUE"].values

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
    write_to_csv(encounters, "processed_encounters")

    med_admin = preprocess_MEDICATION_ADMIN("data/MEDICATION_ADMINISTRATIONS.csv")
    write_to_csv(med_admin, "processed_med_admin")

    admit = preprocess_ADMIT("data/ADMIT_DX.csv")
    write_to_csv(admit, "processed_admit")

    labs = preprocess_LABS("data/LABS.csv")
    write_to_csv(labs, "processed_labs")
    

    # calc_avg_mealtime("data/LABS.csv")
