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

def devide_labs(encounters,labs):
    unique_ids = labs['STUDY_ID'].nunique()  #730 
    # print("Number of unique IDs in the 'study_id' column of the 'labs' sheet:", unique_ids)

    unique_ids = encounters['STUDY_ID'].nunique()  # 770
    # print("Number of unique IDs in the 'study_id' column of the 'encounters' sheet:", unique_ids)

    merged_df = pd.merge(encounters, labs, on='STUDY_ID')  # 728 
    unique_ids = merged_df['STUDY_ID'].nunique()
    # print("Number of unique IDs in the 'study_id' column of the 'merged_df' sheet:", unique_ids)

    missing_ids_labs = set(encounters['STUDY_ID']) - set(labs['STUDY_ID'])
    print("Missing IDs in labs DataFrame:", missing_ids_labs)

    id_counts = merged_df['STUDY_ID'].value_counts()
    merged_df = merged_df[merged_df["STUDY_ID"].isin(id_counts[id_counts>3].index)]
    # merged_df = merged_df.drop(columns=["ENCOUNTER_NUM_x",'HOSP_ADMSN_TOD','HOSP_DISCHRG_TOD','ENCOUNTER_NUM_y','MEAL'])

    tod1,tod2,tod3 = [],[],[]
    glucose1, glucose2, glucose3=[],[],[]
    td1, td2, td3 = [],[],[]
    
    df = 0
    new_data = [] 
    merged_df.reset_index(drop=True, inplace=True)
    

    for group_key, group_indices in  merged_df.groupby("STUDY_ID").groups.items():
        first_index = group_indices[0]

        
        for i in range(len(group_indices)-3): # (0,2)  0,1 [3,4,5,6]
            tod1 = merged_df.iloc[first_index+i]["RESULT_TOD"]
            tod2 = merged_df.iloc[first_index+i+1]["RESULT_TOD"]
            tod3 = merged_df.iloc[first_index+i+2]["RESULT_TOD"]
            tod4 = merged_df.iloc[first_index+i+3]["RESULT_TOD"]
            glucose1 = merged_df.iloc[first_index+i]["GLUCOSE (mmol/L)"]
            glucose2 = merged_df.iloc[first_index+i+1]["GLUCOSE (mmol/L)"]
            glucose3 = merged_df.iloc[first_index+i+2]["GLUCOSE (mmol/L)"]
            td1=0
            td2 = merged_df.iloc[first_index+i+1]["RESULT_HRS_FROM_ADMIT"] -  merged_df.iloc[first_index+i]["RESULT_HRS_FROM_ADMIT"]
            td3 = merged_df.iloc[first_index+i+2]["RESULT_HRS_FROM_ADMIT"] -  merged_df.iloc[first_index+i+1]["RESULT_HRS_FROM_ADMIT"]
            td4 = merged_df.iloc[first_index+i+3]["RESULT_HRS_FROM_ADMIT"] -  merged_df.iloc[first_index+i+2]["RESULT_HRS_FROM_ADMIT"]

            
            # prediction : y 
            glucose4 = merged_df.loc[first_index+i+3]["GLUCOSE (mmol/L)"]

             # Extract static feature from -- encounter 
            # encounters = merged_df.loc[first_index + i]
            # weight = encounters['WEIGHT_KG']
            # height = encounters['HEIGHT_CM']
            # age = encounters['AGE']
            # sex = encounters['SEX']
        
            # new_data.append({"STUDY_ID": group_key,
            #                     'WEIGHT_KG': weight, 'HEIGHT_CM': height,
            #                     'AGE': age, 'SEX': sex,
            #                  'RESULT_TOD1': tod1, 'GLUCOSE1': glucose1, 'TD1': td1,
            #                 'RESULT_TOD2': tod2, 'GLUCOSE2': glucose2, 'TD2': td2,
            #                  'RESULT_TOD3': tod3, 'GLUCOSE3': glucose3, 'TD3': td3,
            #                  'RESULT_TOD4': tod4,  'TD4': td4,
            #                  "GLUCOSE4": glucose4 })
            
            new_data.append({
                "STUDY_ID": group_key,
                    'RESULT_TOD1': tod1, 'GLUCOSE1': glucose1, 'TD1': td1,
                'RESULT_TOD2': tod2, 'GLUCOSE2': glucose2, 'TD2': td2,
                    'RESULT_TOD3': tod3, 'GLUCOSE3': glucose3, 'TD3': td3,
                    'RESULT_TOD4': tod4,  'TD4': td4,
                    "GLUCOSE4": glucose4 })
            

    
    new_labs = pd.DataFrame(new_data)
    print(new_labs)
    print(new_labs.duplicated())
     # Check for duplicate rows
    duplicate_rows = new_labs[new_labs.duplicated()]

    # If duplicate_rows is empty, there are no duplicate rows
    if duplicate_rows.empty:
        print("No duplicate rows found.")
    else:
        print("Duplicate rows found:")
        print(duplicate_rows)
    print("----------------------")
    new_labs = new_labs.drop_duplicates()

    # TODO: Change the tod to hours (float) 

    return new_labs


def write_to_csv(df_file, name):
    df_file.to_csv("data/"+name+".csv", index=None, header=True)


if __name__ == '__main__':

    encounters = pd.read_csv("data/processed_ENCOUNTERS.csv")
    admit = pd.read_csv("data/processed_ADMIT_DX.csv")
    labs = pd.read_csv("data/processed_LABS.csv", index_col=False)
    med_admin = pd.read_csv("data/processed_MEDICATION_ADMINISTRATIONS.csv")

    write_to_csv(med_admin, "MEDICATION_ADMINISTRATIONS.csv")
    
    # dealing with medication_ATC
    atc_columns = [col for col in med_admin.columns if 'MEDICATION_ATC' in col]
    for col in atc_columns:
        labs[col] = 0

    for index, row in labs.iterrows():
        # Use a copy to avoid changing med_admin directly
        summed_meds = aggregate_meds(med_admin.copy(), row, atc_columns)
        labs.loc[index, atc_columns] = summed_meds.values

    # devide labs for each patient based on the meal ( use first three meal to predict the fourth meal)
    new_labs = devide_labs(encounters, labs)
    print(len(new_labs))

    
    df_combined = pd.merge(encounters, new_labs, on=
                           'STUDY_ID', how='inner')
    print(len(df_combined))

    # admit = admit[['STUDY_ID', 'CURRENT_ICD10_LIST']].drop_duplicates(subset = "STUDY_ID").reset_index(drop = True)
    df_combined = pd.merge( admit, df_combined, on=[
                           'STUDY_ID', 'ENCOUNTER_NUM'], how='inner')
    
    print(len(df_combined))

    # Filter out rows where encounter_name equals 1
    df_combined = df_combined[df_combined['ENCOUNTER_NUM'] != 2]
    print(len(df_combined))

    # Check for duplicate rows
    duplicate_rows = df_combined[df_combined.duplicated()]

    # If duplicate_rows is empty, there are no duplicate rows
    if duplicate_rows.empty:
        print("No duplicate rows found.")
    else:
        print("Duplicate rows found:")
        print(duplicate_rows)
    
    df_combined = df_combined.drop_duplicates()
    print(len(df_combined))

    write_to_csv(df_combined,"new_dt")   
    
    


    # # Merge the dynamic and static features based on ID

    # df_combined = pd.merge(labs, encounters, on=[
    #                        'STUDY_ID', 'ENCOUNTER_NUM'], how='left')
    # df_combined = pd.merge(df_combined, admit, on=[
    #                        'STUDY_ID', 'ENCOUNTER_NUM'], how='left')

    # df_combined = df_combined.dropna()

    # write_to_csv(df_combined, "data_processed")