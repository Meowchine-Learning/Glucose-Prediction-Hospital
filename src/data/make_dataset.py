import pandas as pd
import numpy as np
import datetime
import datetime
import csv


def main():

    df_map = pd.read_excel(
        "data/ACCESS 1853 Dataset update 20240228.xlsx", sheet_name=None)

    encounters = df_map["ENCOUNTERS"]
    encounters = encounters.sort_values(
        by=['STUDY_ID', 'ENCOUNTER_NUM'], ascending=[True, True])
    admit_dx = df_map["ADMIT_DX"]
    admit_dx = admit_dx.sort_values(
        by=['STUDY_ID', 'ENCOUNTER_NUM'], ascending=[True, True])
    or_proc_orders = df_map["OR_PROC_ORDERS"]
    or_proc_orders = or_proc_orders.sort_values(
        by=['STUDY_ID', 'ENCOUNTER_NUM', 'PROCEDURE_START_HRS_FROM_ADMIT'], ascending=[True, True, True])
    orders_activity = df_map["ORDERS_ACTIVITY"]
    orders_activity = orders_activity.sort_values(
        by=['STUDY_ID', 'ENCOUNTER_NUM', 'ORDER_HRS_FROM_ADMIT'], ascending=[True, True, True])
    orders_nutrition = df_map["ORDERS_NUTRITION"]
    orders_nutrition = orders_nutrition.sort_values(
        by=['STUDY_ID', 'ENCOUNTER_NUM', 'ORDER_HRS_FROM_ADMIT'], ascending=[True, True, True])
    med_orders = df_map["MEDICATION_ORDERS"]
    med_orders = med_orders.sort_values(
        by=['STUDY_ID', 'ENCOUNTER_NUM', 'ORDER_HRS_FROM_ADMIT'], ascending=[True, True, True])
    labs = df_map["LABS"]
    labs = labs.sort_values(
        by=['STUDY_ID', 'ENCOUNTER_NUM', 'RESULT_HRS_FROM_ADMIT'], ascending=[True, True, True])
    med_admin = df_map["MEDICATION_ADMINISTRATIONS"]
    med_admin = med_admin.sort_values(
        by=['STUDY_ID', 'ENCOUNTER_NUM', 'TAKEN_HRS_FROM_ADMIT'], ascending=[True, True, True])
    pin = df_map["PIN"]
    pin = pin.sort_values(
        by=['STUDY_ID', 'ENCOUNTER_NUM', 'DISP_DAYS_PRIOR'], ascending=[True, True, True])

    clean_encounters(encounters)
    clean_admit(admit_dx)
    clean_or_proc_orders(or_proc_orders)
    clean_pin(pin)

    exclusion_ranges = clean_med_admin(med_admin)
    orders_activity = clean_orders_activiy(orders_activity, exclusion_ranges)
    orders_nutrition = clean_orders_nutrition(
        orders_nutrition, exclusion_ranges)
    labs = clean_labs(labs, exclusion_ranges)

    process_meal_time(labs)

    write_to_csv(encounters, "ENCOUNTERS")
    write_to_csv(admit_dx, "ADMIT_DX")
    write_to_csv(med_admin, "MEDICATION_ADMINISTRATIONS")
    write_to_csv(labs, "LABS")

    # dataset_tcn(encounters, labs, admit_dx, or_proc_orders, orders_nutrition)


def encoding(name, df, column_list):
    def preprocess_column(column):
        # Convert numerical values to strings
        column = column.astype(str)
        return column
    if column_list[0] == "CURRENT_ICD10_LIST":
        # Example usage
        df["CURRENT_ICD10_LIST"] = preprocess_column(df["CURRENT_ICD10_LIST"])

    for column in column_list:
        # Perform one-hot encoding
        categories = df[column].unique()
        np_column = df[column].values.flatten()  # numpy array
        categories = np.unique(np_column)  # categories
        category_index = {category: index for index,
                          category in enumerate(categories)}
        one_hot_encoded = []

        for i in range(len(np_column)):
            data = np_column[i]
            one_hot = [0] * len(categories)  # [0,0]
            index = np.where(data == categories)[0][0]
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
                df.at[i, 'RESULT_HRS_FROM_ADMIT'] - df.at[i-1, 'RESULT_HRS_FROM_ADMIT'] < 12:  # also check that the hours from admit are not more than 12h apart to avoid classifying different days as the same meal
            df.at[i-1, 'MEAL'] = "other"
    # drop rows with MEAL == "other"
    len_before = len(df)
    df.query('MEAL != "other"', inplace=True)
    print(
        f"*\t[process_meal_time] Dropped {len_before - len(df)} rows with meal_time == other, out of {len_before} rows.")


def dataset_tcn(encounters, labs, admit_dx, or_proc_orders, orders_nutrition):
    merged_df = pd.merge(encounters, labs, on='STUDY_ID')
    id_counts = merged_df['STUDY_ID'].value_counts()
    merged_df = merged_df[merged_df["STUDY_ID"].isin(
        id_counts[id_counts > 3].index)]
    merged_df = merged_df.drop(columns=[
                               "ENCOUNTER_NUM_x", 'HOSP_ADMSN_TOD', 'HOSP_DISCHRG_TOD', 'ENCOUNTER_NUM_y', 'MEAL'])

    tod1, tod2, tod3 = [], [], []
    glucose1, glucose2, glucose3 = [], [], []
    td1, td2, td3 = [], [], []

    df = 0
    new_data = []
    merged_df.reset_index(drop=True, inplace=True)

    # TODO: static features:
    # disease DONE
    # OR_PROC_ID_onehot done
    # ORDERS_NUTRITION_ONEHOT
    # ATC

    # TODO: time sequence features:
    # medication TAKEN_TOD, TAKEN_HRS_FROM_ADMIT,MAR_ACTION(one_hot),SIG,DOSE_UNIT,ROUTE
    # activity ( 0/1) and activity time seq?

    for group_key, group_indices in merged_df.groupby("STUDY_ID").groups.items():
        first_index = group_indices[0]

        for i in range(len(group_indices)-3):  # (0,2)  0,1 [3,4,5,6]
            tod1 = merged_df.iloc[first_index+i]["RESULT_TOD"]
            tod2 = merged_df.iloc[first_index+i+1]["RESULT_TOD"]
            tod3 = merged_df.iloc[first_index+i+2]["RESULT_TOD"]
            tod4 = merged_df.iloc[first_index+i+3]["RESULT_TOD"]
            glucose1 = merged_df.iloc[first_index+i]["GLUCOSE (mmol/L)"]
            glucose2 = merged_df.iloc[first_index+i+1]["GLUCOSE (mmol/L)"]
            glucose3 = merged_df.iloc[first_index+i+2]["GLUCOSE (mmol/L)"]
            td1 = 0
            td2 = merged_df.iloc[first_index+i+1]["RESULT_HRS_FROM_ADMIT"] - \
                merged_df.iloc[first_index+i]["RESULT_HRS_FROM_ADMIT"]
            td3 = merged_df.iloc[first_index+i+2]["RESULT_HRS_FROM_ADMIT"] - \
                merged_df.iloc[first_index+i+1]["RESULT_HRS_FROM_ADMIT"]
            td4 = merged_df.iloc[first_index+i+3]["RESULT_HRS_FROM_ADMIT"] - \
                merged_df.iloc[first_index+i+2]["RESULT_HRS_FROM_ADMIT"]

            # prediction : y
            glucose4 = merged_df.loc[first_index+i+3]["GLUCOSE (mmol/L)"]

            # Extract static feature from -- encounter
            encounters = merged_df.loc[first_index + i]
            hosp_dischrg_hrs = encounters['HOSP_DISCHRG_HRS_FROM_ADMIT']
            weight = encounters['WEIGHT_KG']
            height = encounters['HEIGHT_CM']
            age = encounters['AGE']
            sex = encounters['SEX']

            new_data.append({"STUDY_ID": group_key,
                             'HOSP_DISCHRG_HRS_FROM_ADMIT': hosp_dischrg_hrs,
                             'WEIGHT_KG': weight, 'HEIGHT_CM': height,
                             'AGE': age, 'SEX': sex,
                             'TOD1': tod1, 'GLUCOSE1': glucose1, 'TD1': td1,
                            'TOD2': tod2, 'GLUCOSE2': glucose2, 'TD2': td2,
                             'TOD3': tod3, 'GLUCOSE3': glucose3, 'TD3': td3,
                             'TOD4': tod4,  'TD4': td4,
                             "GLUCOSE4": glucose4})

    new_df = pd.DataFrame(new_data)
    print(new_df)
    # Convert lists to tuples in the 'CURRENT_ICD10_LIST' column
    admit_dx['CURRENT_ICD10_LIST'] = admit_dx['CURRENT_ICD10_LIST'].apply(
        tuple)

    # admit_dx
    admit_dx = admit_dx[['STUDY_ID', 'CURRENT_ICD10_LIST']].drop_duplicates(
        subset="STUDY_ID").reset_index(drop=True)
    print(admit_dx)

    print(new_df)
    print(new_df.shape[0])
    new_df = pd.merge(new_df, admit_dx, on='STUDY_ID', how='inner')
    new_df = new_df[['STUDY_ID', 'CURRENT_ICD10_LIST'] +
                    [col for col in new_df.columns if col != 'STUDY_ID' and col != 'CURRENT_ICD10_LIST']]

    print(new_df)
    print(new_df.shape[0])

    # OR_PROC_ORDER
    or_proc_orders = or_proc_orders[['STUDY_ID', 'OR_PROC_ID']].drop_duplicates(
        subset="STUDY_ID").reset_index(drop=True)
    print(or_proc_orders)

    print(new_df)
    print(new_df.shape[0])
    new_df = pd.merge(new_df, or_proc_orders, on='STUDY_ID', how='inner')
    new_df = new_df[['STUDY_ID', 'OR_PROC_ID'] +
                    [col for col in new_df.columns if col != 'STUDY_ID' and col != 'OR_PROC_ID']]

    print(new_df)
    print(new_df.shape[0])

    # order_nutrition
    # orders_nutrition = orders_nutrition[['STUDY_ID', 'PROC_ID']].drop_duplicates(subset = "STUDY_ID").reset_index(drop = True)
    # print(orders_nutrition)

    # print(new_df)
    # print(new_df.shape[0])
    # print("-------------------------- 3")
    # new_df = pd.merge(new_df, orders_nutrition, on='STUDY_ID', how='inner')
    # print(new_df)
    # print(new_df.shape[0])

    write_to_csv(new_df, "new_dt")
    print("Length of unique STUDY_ID values in new_df:",
          new_df['STUDY_ID'].nunique())


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
             r"EXTRACTION, ELECTRODE LEAD, CARDIAC, USING LASER; \Reimplant of CRT-D with new RV and LV leads": "I05.9",
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


def clean_orders_activiy(df, exclusion_ranges):
    # drop PROC_NAME, ORDER_PROC_ID
    df.drop(['PROC_NAME', 'ORDER_PROC_ID'], axis=1, inplace=True)
    df = df[df.apply(
        lambda row: is_within_exclusion_range(row, 'ORDER_HRS_FROM_ADMIT', exclusion_ranges), axis=1)]

    return df


def clean_orders_nutrition(df, exclusion_ranges):
    # drop PROC_NAME, ORDER_PROC_ID
    df.drop(['PROC_NAME', 'ORDER_PROC_ID'], axis=1, inplace=True)

    # find/remove rows with empty DISCON_HRS
    empty_discon = df[(df['ORDER_DISCON_TOD'].notnull()) == False].index

    df.drop(empty_discon, axis=0, inplace=True)
    df = df[df.apply(
        lambda row: is_within_exclusion_range(row, 'ORDER_HRS_FROM_ADMIT', exclusion_ranges), axis=1)]
    return df


def clean_labs(df, exclusion_ranges):
    pd.DataFrame.dropna(df)
    # drop COMPONENT_NAME, EXTERNAL_NAME, REFERENCE_UNIT
    df.drop(['COMPONENT_NAME', 'EXTERNAL_NAME',
            'REFERENCE_UNIT'], axis=1, inplace=True)

    # find/remove rows with empty ORD_VALUE
    empty_ord = df[(df['ORD_VALUE'].notnull()) == False].index

    df.drop(empty_ord, axis=0, inplace=True)
    df = df[df.apply(
        lambda row: is_within_exclusion_range(row, 'RESULT_HRS_FROM_ADMIT', exclusion_ranges), axis=1)]
    return df


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

    # get rows where route is empty
    list = df[(df['ROUTE'].notnull()) == False].index

    # fill in empty values
    for i in range(len(list)):
        df.loc[list[i], "ROUTE"] = med_routes[df.loc[list[i], "MEDICATION_ID"]]

    # find/remove rows with empty values
    empty_vals = df[(df['SIG'].notnull()) == False].index
    df.drop(empty_vals, axis=0, inplace=True)

    exclusion_ranges = find_IV_times(df)

    # drop columns
    df.drop(['MEDICATION_ID', 'MEDICATION_NAME', 'INFUSION_RATE',
            'INFUSION_RATE_UNIT', 'STRENGTH'], axis=1, inplace=True)

    return exclusion_ranges


def find_IV_times(med_admin_df):

    # IV insulins we want to exclude
    insulin_names = ["INSULIN REGULAR (HUMULIN R) 1 UNIT/ML (100 UNIT) IN NACL 0.9% 100 ML INFUSION BAG",
                     "ZZZ_INSULIN REGULAR (HUMULIN R) 1 UNIT/ML IN NACL 0.9% 100 ML (RN)", "INSULIN REGULAR (HUMULIN R) 1 UNIT/ML IN D5W 100 ML", "INSULIN LISPRO (HUMALOG) IN NACL 0.9% 50 ML BAG (FLOOR)"]

    # dataframe with only rows with those meds
    insulin_df = med_admin_df[(med_admin_df['MEDICATION_NAME'] ==
                               insulin_names[0]) | (med_admin_df['MEDICATION_NAME'] ==
                                                    insulin_names[1])]

    # actions which can mean a new medication was started
    start_continuous_actions = ["Restarted", "Continued from OR",
                                "Continued from Pre-op", "Given", "New Bag", "Rate Change"]

    start_time, stop_time = 0, 0
    exclusion_ranges = {}
    start_med = ""
    id = ""
    for _, row in insulin_df.iterrows():
        # if the ID or encounter number changes, reset
        if id != str(row["STUDY_ID"])+'_'+str(row["ENCOUNTER_NUM"]):
            id = str(row["STUDY_ID"])+'_'+str(row["ENCOUNTER_NUM"])
            start_time, stop_time = 0, 0
            exclusion_ranges[id] = []

        action = row["MAR_ACTION"]
        start_med = row["MEDICATION_NAME"]
        # case 1: start action
        if action == "Given":   # Non-continouous case
            start_time = float(row["TAKEN_HRS_FROM_ADMIT"])
            stop_time = start_time + 1
            exclusion_ranges[id].append([start_time, stop_time])
        if action in start_continuous_actions:
            if start_time == 0:  # ensures first administration, or given after a full start, stop cycle occured
                start_time = float(row["TAKEN_HRS_FROM_ADMIT"])
                stop_time = -1
                if start_med == insulin_names[-1]:
                    stop_time = start_time + 1
        # case 2: stop action
        if action == "Stopped":
            if start_time == 0:  # error cases
                try:
                    if stop_time == 0:  # case a: "Stopped" after admission, before any starts
                        start_time = 0
                    else:               # case b: else
                        start_time = exclusion_ranges[id][-1][1]
                        exclusion_ranges.pop()
                except:
                    pass
            stop_time = float(row["TAKEN_HRS_FROM_ADMIT"])
            exclusion_ranges[id].append([start_med, start_time, stop_time])
            start_time = 0
    return exclusion_ranges


def is_within_exclusion_range(row, col, exclusion_ranges):
    key = str(row['STUDY_ID'])+'_'+str(row['ENCOUNTER_NUM'])
    time = row[col]
    if key in exclusion_ranges:
        for _, start_time, stop_time in exclusion_ranges[key]:
            if start_time <= time <= stop_time:
                return False  # Exclude this row
    return True  # Keep this row


def clean_pin(df):
    # drop everything but STUDY_ID, DISP_DAYS_PRIOR, SUPP_DRUG_ATC_CODE
    df.drop(['ENCOUNTER_NUM', 'DRUG_DIN', 'BRAND_NAME', 'DSPN_AMT_QTY', 'DSPN_AMT_UNT_MSR_CD',
            'DSPN_DAY_SUPPLY_QTY', 'DSPN_DAY_SUPPLY_UNT_MSR_CD'], axis=1, inplace=True)


def write_to_csv(df_file, name):
    df_file.to_csv("data/"+name+".csv", index=None, header=True)


if __name__ == "__main__":
    main()
