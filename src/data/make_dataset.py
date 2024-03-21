import pandas as pd
import numpy as np
import datetime
import csv



def main():

    df_map = pd.read_excel(
        "data/ACCESS 1853 Dataset update 20240228.xlsx", sheet_name=None)

    encounters = df_map["ENCOUNTERS"]
    admit_dx = df_map["ADMIT_DX"]
    or_proc_orders = df_map["OR_PROC_ORDERS"]
    orders_activity = df_map["ORDERS_ACTIVITY"]
    orders_nutrition = df_map["ORDERS_NUTRITION"]
    labs = df_map["LABS"]
    med_admin = df_map["MEDICATION_ADMINISTRATIONS"]
    pin = df_map["PIN"]

    clean_encounters(encounters)
    clean_admit(admit_dx)
    clean_or_proc_orders(or_proc_orders)
    clean_orders_activiy(orders_activity)
    clean_orders_nutrition(orders_nutrition)
    clean_labs(labs)
    clean_med_admin(med_admin)
    clean_pin(pin)

    # encoding("ENCOUNTERS",encounters, ["SEX"])
    # encoding("OR_PROC_ORDERS",or_proc_orders, ["OR_PROC_ID"])
    # encoding("ADMIT_DX", admit_dx, ["CURRENT_ICD10_LIST"] )
    # encoding("ORDERS_NUTRITION",orders_nutrition, ["PROC_ID"])
    # encoding("LABS", labs, ["COMPONENT_ID"])
    # encoding("MEDICATION_ADMINISTRATIONS", med_admin, ["MEDICATION_ATC","MAR_ACTION","DOSE_UNIT","ROUTE"])

    process_meal_time(labs)
    dataset_tcn(encounters,labs)


    for key in df_map.keys():
        write_to_csv(df_map[key], key)


def encoding(name,df,column_list):
    def preprocess_column(column):
        # Convert numerical values to strings
        column = column.astype(str)
        return column
    if column_list[0]== "CURRENT_ICD10_LIST":
        # Example usage
        df["CURRENT_ICD10_LIST"] = preprocess_column(df["CURRENT_ICD10_LIST"])

    for column in column_list:
        # Perform one-hot encoding
        categories = df[column].unique()
        np_column = df[column].values.flatten()  # numpy array 
        categories = np.unique(np_column)  # categories 
        category_index = {category: index for index, category in enumerate(categories)}
        one_hot_encoded = []

        for i in range(len(np_column)):
            data = np_column[i]
            one_hot = [0] * len(categories) # [0,0]
            index = np.where(data==categories)[0][0]
            one_hot[index] = 1
            one_hot_encoded.append(one_hot)
        
        # Replace the values in the "SEX" column with one-hot encoded values
        df[column] = one_hot_encoded

        # Write DataFrame to CSV
        write_to_csv(df, name)

def process_meal_time(df):
    # filter for glucose measurements
    # 885 is the COMPONENT_ID for glucose
    df.query('COMPONENT_ID == 885', inplace=True)
    df.drop('COMPONENT_ID', axis=1, inplace=True)
    # rename ORD_VALUE to GLUCOSE (mmol/L)
    df.rename(columns={'ORD_VALUE': 'GLUCOSE (mmol/L)'}, inplace=True)

    # Breakfast: 8:00 AM - 9:30 AM. Lunch: 12:30 AM - 1:30 PM. Supper: 5:00 PM - 6:30 PM
    # breakfast ∈ [7:00, 10:00], lunch ∈ [11:00, 14:00], supper ∈ [16:00, 19:00]

    def classify_time(time):
        breakfast_start = datetime.time(7, 0)
        breakfast_end = datetime.time(10, 0)
        lunch_start = datetime.time(11, 0)
        lunch_end = datetime.time(14, 0)
        supper_start = datetime.time(16, 0)
        supper_end = datetime.time(19, 0)
        
        if breakfast_start <= time <= breakfast_end:
            return "breakfast"
        elif lunch_start <= time <= lunch_end:
            return "lunch"
        elif supper_start <= time <= supper_end:
            return "supper"
        else:
            return "other"
    
    # group by STUDY_ID, sort measurements by HRS_FROM_ADMIT
    df.sort_values(by=['STUDY_ID', 'RESULT_HRS_FROM_ADMIT'], inplace=True)
    
    # for row in filtered_labs:
    df['MEAL'] = df['RESULT_TOD'].apply(classify_time)

    # only keep the last row in any close consecutive breakfast, lunch, or supper measurements
    df.reset_index(drop=True, inplace=True)
    for i in range(1, len(df)):
        if df.at[i, 'STUDY_ID'] == df.at[i-1, 'STUDY_ID'] and \
            df.at[i, 'MEAL'] == df.at[i-1, 'MEAL'] and \
            df.at[i, 'RESULT_HRS_FROM_ADMIT'] - df.at[i-1, 'RESULT_HRS_FROM_ADMIT'] < 12: # also check that the hours from admit are not more than 12h apart to avoid classifying different days as the same meal
                df.at[i-1, 'MEAL'] = "other"
    # drop rows with MEAL == "other"
    len_before = len(df)
    df.query('MEAL != "other"', inplace=True)
    print(f"*\t[process_meal_time] Dropped {len_before - len(df)} rows with meal_time == other, out of {len_before} rows.")


def dataset_tcn(encounters, labs):
    merged_df = pd.merge(encounters, labs, on='STUDY_ID')
    id_counts = merged_df['STUDY_ID'].value_counts()
    merged_df = merged_df[merged_df["STUDY_ID"].isin(id_counts[id_counts>3].index)]
    merged_df.drop(columns=['HOSP_ADMSN_TOD','HOSP_DISCHRG_TOD','ENCOUNTER_NUM_y','MEAL'])

    tod1,tod2,tod3 = [],[],[]
    glucose1, glucose2, glucose3=[],[],[]
    td1, td2, td3 = [],[],[]
    

    # id = merged_df['STUDY_ID'][0]
    df = 0
    combined_df = merged_df.groupby("STUDY_ID")
    with open('output_file.csv', 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)

        # Write headers to the CSV file
        csv_writer.writerow(['TOD1', 'Glucose1', 'TD1', 'TOD2', 'Glucose2', 'TD2', 'TOD3', 'Glucose3', 'TD3'])
        for group_key, group_indices in  merged_df.groupby("STUDY_ID").groups.items():
            first_index = group_indices[0]
            for i in range(len(group_indices)-2): # (0,2)  0,1 [3,4,5,6]
                # group = merged_df.iloc[i:i+3]
                tod1 = merged_df.iloc[i]["RESULT_TOD"]
                tod2 = merged_df.iloc[i+1]["RESULT_TOD"]
                tod3 = merged_df.iloc[i+2]["RESULT_TOD"]
                glucose1 = merged_df.iloc[i]["GLUCOSE (mmol/L)"]
                glucose2 = merged_df.iloc[i+1]["GLUCOSE (mmol/L)"]
                glucose3 = merged_df.iloc[i+2]["GLUCOSE (mmol/L)"]
                # print("i: ", i, first_index)
                # if i==0:
                #     td1 = 0
                # else:
                # td1 = merged_df.iloc[i]["RESULT_HRS_FROM_ADMIT"] -  merged_df.iloc[i-1]["RESULT_HRS_FROM_ADMIT"]
                td1=0
                td2 = merged_df.iloc[i+1]["RESULT_HRS_FROM_ADMIT"] -  merged_df.iloc[i]["RESULT_HRS_FROM_ADMIT"]
                td3 = merged_df.iloc[i+2]["RESULT_HRS_FROM_ADMIT"] -  merged_df.iloc[i+1]["RESULT_HRS_FROM_ADMIT"]
                
                csv_writer.writerow([tod1, glucose1, td1, tod2, glucose2, td2, tod3, glucose3, td3])

                
                # print(tod1,glucose1, td1, tod2,glucose2, td2, tod3, glucose3 , td3)
            




    # for index, row in merged_df.iterrows():
    #     print(index)
    #     print(row)
    #     print(row["GLUCOSE (mmol/L)"])
    #     break










def clean_encounters(df):
    # drop A1C columns
    df.drop(['PRE_ADMISSION_A1C', 'A1C_DAYS'], axis=1, inplace=True)

    # find/remove rows with empty heights
    empty_heights = df[(df['HEIGHT_CM'].notnull()) == False].index

    df.drop(empty_heights, axis=0, inplace=True)


def clean_admit(df):
    '''
        I found rows with missing ICD10 codes, I filled those missing codes using the rows's Admit_diag_text cells (or Dx_name)
        I match the codes from Admit_Dx
        Detaisl are in the dict below
    '''

    # missing codes
    codes = {"Heart Failure": "I50.9", "Heart Failure, Pericardial Effusion": "I31.3, I50.9", "Critical Aortic Stenosis with Heart Failure": "I35.0, I50.9", "CHF": "I50.0",
             "AORTIC STENOSIS": "I35.0",
             "STEMI": "I21.3, R94.30", "NSTEMI exacerbation": "I21.4, R94.31", "NSTEMI": "I21.4, R94.31", "Acute MI": "I21.9",
             "Chest pain and SOB": "R07.4, R06.0", "Chest Pain": "R07.4", "SOB": "R06.0",
             "Infected Endocarditis": "I33.0", "Endocarditis": "I38",

             "Triple Vessel Disease": "I25.19", "3-vessel CAD": "I25.19", "Left Main CAD": "I25.19", "Coronary Artery Disease": "I25.19", "multivessel disease": "I25.19",
             "Lung Transplant": "Z94.2", "Lung Transplant.": "Z94.2", "Lung Tx.": "Z94.2", "Transplantation": "Z94.9",
             "mitral regurgitation": "I34.0",
             "Postop Sternal Pain": "R07.3", "Sternal wound infection": "S21.11", "Sternum infection": "S21.11", "Sternal infection": "S21.11",
             "Wound Infection": "T14.1, T79.3", "post op infection": "T81.4", "Driveline Infection": "T82.79, Y83.1",

             "Cardioverter defibrillator subcutaneous insertion (SICD)": "Z45.01",
             "Fluid overload": "E87.7",
             # "Abscess to left thigh": "L02.4",
             "Unstable angine": "I20.0",
             "Symptomatic Bradycardia": "R00.1",
             "Cardiogenic shock": "R57.0",
             "SAH": "I60.9",
             "ACS": "I24.9",
             "Afib, new onset": "I48.90",

             "stroke": "I64",
             "CABG (CORONARY ARTERY BYPASS GRAFT) [1070528]": "Z95.1", "CABG, WITH AORTIC VALVE REPLACEMENT": "Z95.1, Z95.4", "CABG": "Z95.1",
             "NSTEMI/CABG": "Z95.1, I21.4, R94.31", "NSTEMI/wtg CABG": "Z95.1, I21.4, R94.31",
             "Aortic Dissection": "I71.0", "MINI-STERNOTOMY AORTIC VALVE REPLACEMENT": "Z95.4", "REOPERATION, WITH AORTIC VALVE REPAIR OR REPLACEMENT [1072465]": "Z95.4", "Valve repair post op complication": "Y83.8",
             "90% LM stenosis": "I35.0",
             "Fluid overload VAD": "Z95",
             "STERNAL WOUND": "S21.11", "Sternal wire infected": "T82.79",
             "Pulmonary Fibrosis": "J84.9", "Pulmonary": "J44.9", "REPAIR, PARTIAL ANOMALOUS PULMONARY VENOUS RETURN": "Q24.8",
             "Gram Positive Sepsis": "A41.9",
             "HEART Transplant": "Z76.805", "Heart tx": "Z94.9",
             "Acute MI and decompensated heart failure": "I50.0, I21.9",
             "Double Lung Transplant": "Z94.2", "Preop lung tx": "Z76.803",
             "MVR": "Z95.4",
             "Fungal infection": "Z76.803",
             "TAVR work up": "Z95.4",
             "low hgb": "Z13.6",
             "Generator change": "Z45.00",
             "Pacemaker Problem": "Z45.00",
             "Valvular heart failure": "I50.9",
             "Coronary angiography +/- PCI": "Z13.6", "Coronary angiography W&R Cath from LAC LA BICHE; - non isolated, heparin gtt, 15/15, independent, RA": "Z13.6",
             "Coronary angiography/possible PCI; Patient in Westlock hospital W&R Cath; - non isolated, RA, SL, able to lay flat, 15/15, independent": "Z13.6",
             "Coronary angiography W&R Cath from Northern Lights Hospital 7807916296; - 48 Iso due to SOB and no Cough covid negative, independent, able to lay flat with pain management, 15/15": "Z13.6",
             "LEFT HEART CATHETERIZATION +/- PCI": "4A023N", "Right heart catheterization": "4A023N6",
             "VAD Workup": "Z13.6", "AVR May 11 - +/- VAD. VAD workup": "I35.9", "VAD / bradycardia": "Z95",
             "DEBRIDEMENT, STERNUM, WITH REPAIR USING PLATE": "T81.4",
             "Lung disease": "J98.4",
             "Pre op MVR and lead extraction": "I05.9",
             "EDPD": "Z76.803",
             "Abd pain": "R10.4",
             "Enlarging hematoma to left chest wall": "T82.9",
             "Pump thrombosis": "T82.9",
             "Bacteremia/Mitral Vegetation": "I05.9",
             "Penetrating Ulcer - Distal Aortic Arch": "I70.0",
             "EXTRACTION, ELECTRODE LEAD, CARDIAC, USING LASER; \Reimplant of CRT-D with new RV and LV leads": "I05.9",
             "liver biopsy VAD patient": "Z95", "VAD patient for generator change Monday": "Z45.00,Z95", "REMOVAL, ELECTRODE LEAD, ICD [1072379]": "I05.9", "VAD work- up": "Z13.6", "REMOVAL, ELECTRODE LEAD, ICD [1072379]": "I05.9"
             }

    # turn all to lowercase
    codes = {k.lower(): v for k, v in codes.items()}

    # drop dx_id column
    df.drop('DX_ID', axis=1, inplace=True)

    # get a list of the indices where the code cell is missing
    list = df[(df['CURRENT_ICD10_LIST'].notnull()) == False].index

    # fill in missing codes
    for i in range(len(list)):
        # check if free text cell is non-empty
        if isinstance(df.at[list[i], "ADMIT_DIAG_TEXT"], str):
            dx = df.at[list[i], "ADMIT_DIAG_TEXT"].lower()
        else:
            dx = df.at[list[i], "DX_NAME"].lower()  # if missing, use DX_Name
        if dx in codes:
            df.loc[list[i], "CURRENT_ICD10_LIST"] = codes[dx]
        else:
            # drop rows where we don't know know the code
            df.drop(list[i], axis=0, inplace=True)

    df.drop('ADMIT_DIAG_TEXT', axis=1, inplace=True)
    df.drop('DX_NAME', axis=1, inplace=True)


def clean_or_proc_orders(df):
    # drop all columns except STUDY_ID, ENCOUNTER_NUM, and OR_PROC_ID
    df.drop(['PROC_DISPLAY_NAME', 'ANESTHESIA_START_TOD', 'ANESTHESIA_START_HRS_FROM_ADMIT', 'PROCEDURE_START_TOD', 'PROCEDURE_START_HRS_FROM_ADMIT',
            'PROCEDURE_COMP_TOD', 'PROCEDURE_COMP_HRS_FROM_ADMIT', 'ANESTHESIA_STOP_TOD', 'ANESTHESIA_STOP_HRS_FROM_ADMIT'], axis=1, inplace=True)


def clean_orders_activiy(df):
    # drop PROC_NAME, ORDER_PROC_ID
    df.drop(['PROC_NAME', 'ORDER_PROC_ID'], axis=1, inplace=True)


def clean_orders_nutrition(df):
    # drop PROC_NAME, ORDER_PROC_ID
    df.drop(['PROC_NAME', 'ORDER_PROC_ID'], axis=1, inplace=True)

    # find/remove rows with empty DISCON_HRS
    empty_discon = df[(df['ORDER_DISCON_TOD'].notnull()) == False].index

    df.drop(empty_discon, axis=0, inplace=True)


def clean_labs(df):
    pd.DataFrame.dropna(df)
    # drop COMPONENT_NAME, EXTERNAL_NAME, REFERENCE_UNIT
    df.drop(['COMPONENT_NAME', 'EXTERNAL_NAME',
            'REFERENCE_UNIT'], axis=1, inplace=True)

    # find/remove rows with empty ORD_VALUE
    empty_ord = df[(df['ORD_VALUE'].notnull()) == False].index

    df.drop(empty_ord, axis=0, inplace=True)


def clean_med_admin(df):
    '''
    H02AB02
    DEXAMETHASONE IN NACL 0.9% 50 ML 6000637

    H02AB09
    HYDROCORTISONE SODIUM SUCCINATE IN NACL 0.9% 100 ML (100 MG VIAL)(FLOOR) 6003152
    HYDROCORTISONE SODIUM SUCCINATE IN NACL 0.9% 50 ML (100 MG VIAL)(FLOOR) 6003151
    HYDROCORTISONE SODIUM SUCCINATE IN NACL 0.9% 50 ML BAG 6001118

    A10AB05
    INSULIN ASPART (NOVORAPID FLEXTOUCH) 100 UNIT/ML INJECTION PEN 4002908

    A10AB04
    INSULIN LISPRO (HUMALOG KWIKPEN) 100 UNIT/ML INJECTION PEN 4002884
    INSULIN LISPRO (HUMALOG) IN NACL 0.9% 50 ML BAG (FLOOR) 6001625

    A10AB01
    ZZZ_INSULIN REGULAR (HUMULIN R) 1 UNIT/ML IN NACL 0.9% 100 ML (RN) 6004606
    INSULIN REGULAR (HUMULIN R) 1 UNIT/ML (100 UNIT) IN NACL 0.9% 100 ML INFUSION BAG 6002910

    H02AB04
    ZZZMETHYLPREDNISOLONE SODIUM SUCCINATE IN D5W 50 ML BAG (125 MG/2 ML VIAL)(RN) 6003166
    '''

    # ATC code fixes
    dexamethasone = df[df['MEDICATION_ID'] == 6000637].index
    for i in range(len(dexamethasone)):
        df.loc[dexamethasone[i], "MEDICATION_ATC"] = 'H02AB02'

    hydrocortisone = df[df['MEDICATION_ID'] == 6003152].index
    for i in range(len(hydrocortisone)):
        df.loc[hydrocortisone[i], "MEDICATION_ATC"] = 'H02AB09'
    hydrocortisone = df[df['MEDICATION_ID'] == 6003151].index
    for i in range(len(hydrocortisone)):
        df.loc[hydrocortisone[i], "MEDICATION_ATC"] = 'H02AB09'
    hydrocortisone = df[df['MEDICATION_ID'] == 6001118].index
    for i in range(len(hydrocortisone)):
        df.loc[hydrocortisone[i], "MEDICATION_ATC"] = 'H02AB09'

    insulin_aspart = df[df['MEDICATION_ID'] == 4002908].index
    for i in range(len(insulin_aspart)):
        df.loc[insulin_aspart[i], "MEDICATION_ATC"] = 'A10AB05'

    insulin_lispro = df[df['MEDICATION_ID'] == 4002884].index
    for i in range(len(insulin_lispro)):
        df.loc[insulin_lispro[i], "MEDICATION_ATC"] = 'A10AB04'
    insulin_lispro = df[df['MEDICATION_ID'] == 6001625].index
    for i in range(len(insulin_lispro)):
        df.loc[insulin_lispro[i], "MEDICATION_ATC"] = 'A10AB04'

    insulin_regular = df[df['MEDICATION_ID'] == 6004606].index
    for i in range(len(insulin_regular)):
        df.loc[insulin_regular[i], "MEDICATION_ATC"] = 'A10AB01'
    insulin_regular = df[df['MEDICATION_ID'] == 6002910].index
    for i in range(len(insulin_regular)):
        df.loc[insulin_regular[i], "MEDICATION_ATC"] = 'A10AB01'

    methylprednisolone = df[df['MEDICATION_ID'] == 6003166].index
    for i in range(len(methylprednisolone)):
        df.loc[methylprednisolone[i], "MEDICATION_ATC"] = 'H02AB04'

    # infusion unit fixes
    inf_unit_fixes = df[df['INFUSION_RATE_UNIT'] == 'mL/hr'].index
    for i in range(len(inf_unit_fixes)):
        # fix infusion rate == 0
        if df.loc[inf_unit_fixes[i], "INFUSION_RATE"] == 0:
            df.loc[inf_unit_fixes[i], "SIG"] = 0
            df.loc[inf_unit_fixes[i], "DOSE_UNIT"] = 'mL/hr'
        # fix empty values in SIG
        elif pd.isnull(df.loc[inf_unit_fixes[i], "SIG"]):
            df.loc[inf_unit_fixes[i],
                   "SIG"] = df.loc[inf_unit_fixes[i], "INFUSION_RATE"]
            df.loc[inf_unit_fixes[i], "DOSE_UNIT"] = 'mL/hr'
        # normalize units
        elif df.loc[inf_unit_fixes[i], "DOSE_UNIT"] not in ['g', 'mL', 'mg']:
            df.loc[inf_unit_fixes[i], "DOSE_UNIT"] = 'mL/hr'

    # missing routes for meds
    med_routes = {4000287: "oral", 124838: "subcutaneous", 2365: "intravenous", 4002245: "intravenous",
                  6000183: "intravenous", 174845: "oral", 2365: "intravenous", 33009: "oral"}

    # insulin_list = ["17405", "28534", "30080", "124838", "124845", "124847", "124854", "124857", "125482", "130342", "134056", "162674", "166114", "169138", "199429", "4002243",
    #                "4002245", "4002455", "4002541", "4002722", "4002723", "4002884", "4002908", "4002909", "6000598", "6001625", "6002910", "6004503", "6004606"]

    # get rows where route is empty
    list = df[(df['ROUTE'].notnull()) == False].index

    # fill in empty values
    for i in range(len(list)):
        df.loc[list[i], "ROUTE"] = med_routes[df.loc[list[i], "MEDICATION_ID"]]

    # find/remove rows with empty values
    empty_vals = df[(df['SIG'].notnull()) == False].index
    df.drop(empty_vals, axis=0, inplace=True)

    # drop columns
    df.drop(['MEDICATION_ID', 'MEDICATION_NAME', 'INFUSION_RATE',
            'INFUSION_RATE_UNIT', 'STRENGTH'], axis=1, inplace=True)


def clean_pin(df):
    # drop everything but STUDY_ID, DISP_DAYS_PRIOR, SUPP_DRUG_ATC_CODE
    df.drop(['ENCOUNTER_NUM', 'DRUG_DIN', 'BRAND_NAME', 'DSPN_AMT_QTY', 'DSPN_AMT_UNT_MSR_CD',
            'DSPN_DAY_SUPPLY_QTY', 'DSPN_DAY_SUPPLY_UNT_MSR_CD'], axis=1, inplace=True)


def write_to_csv(df_file, name):
    df_file.to_csv("data/"+name+".csv", index=None, header=True)


def preprocess_data(df_file, name):
    pass




if __name__ == "__main__":
    main()
