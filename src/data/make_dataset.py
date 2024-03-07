import pandas as pd
import json


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

    for key in df_map.keys():
        write_to_csv(df_map[key], key)


def clean_encounters(df):
    # drop A1C columns
    df.drop(['PRE_ADMISSION_A1C', 'A1C_DAYS'], axis=1, inplace=True)

    # find/remove rows with empty heights
    empty_heights = df[(df['HEIGHT_CM'].notnull()) == False].index

    df.drop(empty_heights, axis=0, inplace=True)


def clean_admit(df):

    # missing codes
    codes = {"Heart Failure": "I50.9", "Heart Failure, Pericardial Effusion": "I31.3, I50.9", "Critical Aortic Stenosis with Heart Failure": "I35.0, I50.9", "CHF": "I50.0",
             "AORTIC STENOSIS": "I35.0",   # 10 & 11
             "STEMI": "I21.3, R94.30", "NSTEMI exacerbation": "I21.4, R94.31", "NSTEMI": "I21.4, R94.31", "Acute MI": "I21.9",
             "Chest pain and SOB": "R07.4, R06.0", "Chest Pain": "R07.4", "SOB": "R06.0",  # 2
             "Infected Endocarditis": "I33.0", "Endocarditis": "I38",

             "Triple Vessel Disease": "I25.19", "3-vessel CAD": "I25.19", "Left Main CAD": "I25.19", "Coronary Artery Disease": "I25.19", "multivessel disease": "I25.19",
             "Lung Transplant": "Z94.2", "Lung Transplant.": "Z94.2", "Lung Tx.": "Z94.2", "Transplantation": "Z94.9",
             "mitral regurgitation": "I34.0",  # 6
             "Postop Sternal Pain": "R07.3", "Sternal wound infection": "S21.11", "Sternum infection": "S21.11", "Sternal infection": "S21.11",  # 3, 9
             "Wound Infection": "T14.1, T79.3", "post op infection": "T81.4", "Driveline Infection": "T82.79, Y83.1",   # 4

             # 12
             # 12
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
             "Fluid overload VAD": "Y71.2",
             "STERNAL WOUND": "S21.11", "Sternal wire infected": "T82.79",
             "Pulmonary Fibrosis": "J84.9", "Pulmonary": "J44.9", "REPAIR, PARTIAL ANOMALOUS PULMONARY VENOUS RETURN": "Q24.8",  # check
             "Gram Positive Sepsis": "A41.9",
             "HEART Transplant": "Z94.9", "Heart tx": "Z94.9",
             "Acute MI and decompensated heart failure": "I50.0, I21.9",
             "Double Lung Transplant": "Z94.2", "Preop lung tx": "Z76.803",
             "MVR": "Z95.4",
             "Fungal infection": "B99",
             "TAVR work up": "Z95.4",
             "low hgb": "D64.9",
             "Generator change": "Z45.00",
             "Pacemaker Problem": "Z45.00",
             "Valvular heart failure": "I51.9",
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

    # QUESTION #2: Is SOB the same as SOBOE?    Not necesserily the same
    # QUESTION #3: Is post op Postop Sternal Pain the same as sternal pain? probably
    # QUESTION #4: Is Driveline Infection same as Infection associated with driveline of left ventricular assist device (LVAD)? probably
    # QUESTION #5: Is Fluid overload VAD same as Fluid overload?    different
    # QUESTION #6: Is Bacteremia/Mitral Vegetation same as Mitral regurgitation?    different
    # QUESTION #7: what does MVR stand for (repair, replacement, regurgitation)?    send it here
    # QUESTION #8: Is NSTEMI same as NSTEMI exacerbation?   probably the same
    # QUESTION #9: Is sternum infection same as sternum wound infection?    yes
    # QUESTION #10: "Heart Failure, Pericardial Effusion" same as composite of two?
    # QUESTION #11: Is Valvular heart failure same as Valvular heart disease?   different
    # QUESTION #12: Is "Cardioverter defibrillator subcutaneous insertion (SICD)" same code as Fitting or adjustment of automatic implantable cardioverter-defibrillator?   same

    # "Coronary angiography +/- PCI", "Coronary angiography W&R Cath from LAC LA BICHE; - non isolated, heparin gtt, 15/15, independent, RA", "Coronary angiography/possible PCI; Patient in Westlock hospital W&R Cath; - non isolated, RA, SL, able to lay flat, 15/15, independent","Coronary angiography W&R Cath from Northern Lights Hospital 7807916296; - 48 Iso due to SOB and no Cough covid negative, independent, able to lay flat with pain management, 15/15",
    # "LEFT HEART CATHETERIZATION +/- PCI": "", "Right heart catheterization":"",
    # "VAD Workup", "AVR May 11 - +/- VAD. VAD workup":"", "VAD / bradycardia",  # 5
    # "DEBRIDEMENT, STERNUM, WITH REPAIR USING PLATE",
    # "Lung disease",
    # "Pre op MVR and lead extraction":"Z95.4",   # 7
    # "EDPD",
    # "Abd pain",
    # "Enlarging hematoma to left chest wall",
    # "Pump thrombosis",
    # "Bacteremia/Mitral Vegetation": "I34.0",
    # "Penetrating Ulcer - Distal Aortic Arch",

    # extra:
    # "EXTRACTION, ELECTRODE LEAD, CARDIAC, USING LASER; \Reimplant of CRT-D with new RV and LV leads"
    # "liver biopsy VAD patient", "VAD patient for generator change Monday", "REMOVAL, ELECTRODE LEAD, ICD [1072379]", "VAD work- up", "REMOVAL, ELECTRODE LEAD, ICD [1072379]"


def clean_labs(df):
    # The ratio of missing data is really small
    # drop all the None value
    return pd.DataFrame.dropna(df)


def clean_or_proc_orders(df):
    # drop all columns except STUDY_ID, ENCOUNTER_NUM, and OR_PROC_ID
    df.drop(['PROC_DISPLAY_NAME', 'ANESTHESIA_START_TOD', 'ANESTHESIA_START_HRS_FROM_ADMIT', 'PROCEDURE_START_TOD', 'PROCEDURE_START_HRS_FROM_ADMIT', 'PROCEDURE_COMP_TOD', 'PROCEDURE_COMP_HRS_FROM_ADMIT', 'ANESTHESIA_STOP_TOD', 'ANESTHESIA_STOP_HRS_FROM_ADMIT'], axis=1, inplace=True)


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
    # drop COMPONENT_NAME, EXTERNAL_NAME, REFERENCE_UNIT
    df.drop(['COMPONENT_NAME', 'EXTERNAL_NAME',
            'REFERENCE_UNIT'], axis=1, inplace=True)

    # find/remove rows with empty ORD_VALUE
    empty_ord = df[(df['ORD_VALUE'].notnull()) == False].index

    df.drop(empty_ord, axis=0, inplace=True)


def clean_med_admin(df):

    # drop ATC codes
    df.drop('MEDICATION_ATC', axis=1, inplace=True)

    df.drop('MEDICATION_NAME', axis=1, inplace=True)
    df.drop('STRENGTH', axis=1, inplace=True)

    # missing routes for meds
    med_routes = {4000287: "oral", 124838: "subcutaneous", 2365: "intravenous", 4002245: "intravenous",
                  6000183: "intravenous", 174845: "oral", 2365: "intravenous", 33009: "oral"}

    # insulin_list = ["17405", "28534", "30080", "124838", "124845", "124847", "124854", "124857", "125482", "130342", "134056", "162674", "166114", "169138", "199429", "4002243",
    #                "4002245", "4002455", "4002541", "4002722", "4002723", "4002884", "4002908", "4002909", "6000598", "6001625", "6002910", "6004503", "6004606"]

    # QUESTION: what to do when columns I-M are empty?

    # get rows where route is empty
    list = df[(df['ROUTE'].notnull()) == False].index

    for i in range(len(list)):
        df.loc[list[i], "CURRENT_ICD10_LIST"] = med_routes[df.loc[list[i],
                                                                  "MEDICATION_ID"]]    # fill in missing routes


def clean_pin(df):
    # drop everything but STUDY_ID, DISP_DAYS_PRIOR, SUPP_DRUG_ATC_CODE
    df.drop(['ENCOUNTER_NUM', 'DRUG_DIN', 'BRAND_NAME', 'DSPN_AMT_QTY', 'DSPN_AMT_UNT_MSR_CD', 'DSPN_DAY_SUPPLY_QTY', 'DSPN_DAY_SUPPLY_UNT_MSR_CD'], axis=1, inplace=True)


def write_to_csv(df_file, name):
    df_file.to_csv(name+".csv", index=None, header=True)


def preprocess_data(df_file, name):
    pass


if __name__ == "__main__":
    main()
