import pandas as pd
import numpy as np


def main():

    df_map = pd.read_excel(
        "data/ACCESS 1853 Dataset.xlsx", sheet_name=None)

    encounters = df_map["ENCOUNTERS"]
    admit_dx = df_map["ADMIT_DX"]
    OR_orders = df_map["OR_PROC_ORDERS"]
    orders = df_map["ORDERS"]
    orders_qs = df_map["ORDER_QUESTIONS"]
    labs = df_map["LABS"]
    med_admin = df_map["MEDICATION_ADMINISTRATIONS"]
    med_orders = df_map["MEDICATION_ORDERS"]
    pin = df_map["PIN"]

    clean_admit(admit_dx)

    for key in df_map.keys():
        write_to_csv(df_map[key], key)


def clean_admit(df):
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
             "Abscess to left thigh": "L02.4",
             "Unstable angine": "I20.0",
             "Symptomatic Bradycardia": "R00.1",
             "Cardiogenic shock": "R57.0",
             "SAH": "I60.9",
             "ACS": "I24.9",
             "Afib, new onset": "I48.90",
             }

    codes = {k.lower(): v for k, v in codes.items()}

    df.drop('DX_ID', axis=1, inplace=True)

    list = df[(df['CURRENT_ICD10_LIST'].notnull()) == False].index

    for i in range(len(list)):
        if not isinstance(df.at[list[i], "ADMIT_DIAG_TEXT"], float):
            dx = df.at[list[i], "ADMIT_DIAG_TEXT"].lower()
        else:
            dx = df.at[list[i], "DX_NAME"].lower()
        if dx in codes:
            df.loc[list[i], "CURRENT_ICD10_LIST"] = codes[dx]
            # print(df.loc[list[i]])
        else:
            df.drop(list[i], axis=0, inplace=True)

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

    # "stroke":"",
    # "CABG (CORONARY ARTERY BYPASS GRAFT) [1070528]":"", "CABG, WITH AORTIC VALVE REPLACEMENT":"",
    # "Aortic Dissection":"", "MINI-STERNOTOMY AORTIC VALVE REPLACEMENT", "Penetrating Ulcer - Distal Aortic Arch", "REOPERATION, WITH AORTIC VALVE REPAIR OR REPLACEMENT [1072465]", "Valve repair post op complication",
    # "Coronary angiography +/- PCI", "Coronary angiography W&R Cath from LAC LA BICHE; - non isolated, heparin gtt, 15/15, independent, RA", "Coronary angiography/possible PCI; Patient in Westlock hospital W&R Cath; - non isolated, RA, SL, able to lay flat, 15/15, independent", "90% LM stenosis", "Coronary angiography W&R Cath from Northern Lights Hospital 7807916296; - 48 Iso due to SOB and no Cough covid negative, independent, able to lay flat with pain management, 15/15",
    # "VAD Workup", "Fluid overload VAD", "AVR May 11 - +/- VAD. VAD workup":"", "VAD / bradycardia",  # 5
    # "STERNAL WOUND", "DEBRIDEMENT, STERNUM, WITH REPAIR USING PLATE", "Sternal wire infected",
    # "Pulmonary Fibrosis", "Pulmonary","REPAIR, PARTIAL ANOMALOUS PULMONARY VENOUS RETURN",
    # "LEFT HEART CATHETERIZATION +/- PCI": "", "Right heart catheterization":"",
    # "HEART Transplant", "Acute MI and decompensated heart failure"
    # "Lung disease", "Double Lung Transplant", "Preop lung tx",
    # "Pre op MVR and lead extraction", "MVR",  # 7
    # "Gram Positive Sepsis":"",
    # "Fungal infection",
    # "EDPD",
    # "TAVR work up":"",
    # "low hgb",
    # "Generator change",
    # "Pacemaker Problem",
    # "Abd pain",
    # "Enlarging hematoma to left chest wall",
    # "Pump thrombosis",
    # "Bacteremia/Mitral Vegetation": "I34.0",
    # "Valvular heart failure": "I38",

    # extra:
    # "EXTRACTION, ELECTRODE LEAD, CARDIAC, USING LASER; \Reimplant of CRT-D with new RV and LV leads"
    # "liver biopsy VAD patient", "CABG", "VAD patient for generator change Monday", "REMOVAL, ELECTRODE LEAD, ICD [1072379]", "VAD work- up", "Heart tx", "NSTEMI/CABG", "REMOVAL, ELECTRODE LEAD, ICD [1072379]", "NSTEMI/wtg CABG"


def write_to_csv(df_file, name):
    df_file.to_csv("data/"+name+".csv", header=True, index=False)


def preprocess_data(df_file, name):
    pass


if __name__ == "__main__":
    main()
