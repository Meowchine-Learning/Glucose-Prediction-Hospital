import csv
import json
import numpy as np
import nltk
from gensim.models.doc2vec import TaggedDocument, Doc2Vec


def _dataInput_csv(filePath: str) -> list:
    with open(filePath, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader, None)
        IN = list(reader)
    return IN


def _dataOutput_json(DATA, dirPath: str = "src/features/output/") -> None:
    FEATURE_DATA = {}
    SEQUENCE_DATA = {}
    for ID in DATA.keys():
        DATAForCurrID = DATA[ID]

        # FEATURE DATASET:
        FEATURE_DATA[ID] = {}
        FEATURE_DATA[ID]["WEIGHT_KG"] = DATAForCurrID["WEIGHT_KG"]
        FEATURE_DATA[ID]["HEIGHT_CM"] = DATAForCurrID["HEIGHT_CM"]
        FEATURE_DATA[ID]["AGE"] = DATAForCurrID["AGE"]
        FEATURE_DATA[ID]["SEX"] = DATAForCurrID["SEX"]
        FEATURE_DATA[ID]["DISEASES"] = DATAForCurrID["DISEASES"]
        FEATURE_DATA[ID]["OR_PROC_ID"] = DATAForCurrID["OR_PROC_ID"]
        FEATURE_DATA[ID]["OR_PROC_ID_ONEHOT"] = DATAForCurrID["OR_PROC_ID_ONEHOT"]
        FEATURE_DATA[ID]["ORDERS_NUTRITION"] = DATAForCurrID["ORDERS_NUTRITION"]
        FEATURE_DATA[ID]["ORDERS_NUTRITION_ONEHOT"] = DATAForCurrID["ORDERS_NUTRITION_ONEHOT"]
        FEATURE_DATA[ID]["LAB_COMPONENT_ID"] = DATAForCurrID["LAB_COMPONENT_ID"]
        FEATURE_DATA[ID]["LAB_COMPONENT_ID_ONEHOT"] = DATAForCurrID["LAB_COMPONENT_ID_ONEHOT"]
        FEATURE_DATA[ID]["LAB_ORD_VALUE"] = DATAForCurrID["LAB_ORD_VALUE"]
        FEATURE_DATA[ID]["MEDICATION_ATC_ENCODED"] = DATAForCurrID["MEDICATION_ATC_ENCODED"]
        FEATURE_DATA[ID]["MEDICATION_ACTIONS"] = DATAForCurrID["MEDICATION_ACTIONS"]
        FEATURE_DATA[ID]["MEDICATION_ACTIONS_ENCODED"] = DATAForCurrID["MEDICATION_ACTIONS_ENCODED"]
        FEATURE_DATA[ID]["PRIOR_MEDICATION_ATC_ENCODED"] = DATAForCurrID["PRIOR_MEDICATION_ATC_ENCODED"]
        FEATURE_DATA[ID]["PRIOR_MEDICATION_DISP_DAYS_NORM"] = DATAForCurrID["PRIOR_MEDICATION_DISP_DAYS_NORM"]
        FEATURE_DATA[ID]["PRIOR_MEDICATION_DISP_DAYS_NORM_ONEHOT"] = DATAForCurrID["PRIOR_MEDICATION_DISP_DAYS_NORM_ONEHOT"]

        # SEQUENCE DATASET:
        SEQUENCE_DATA[ID] = {}
        temp = {str(DATAForCurrID["HOSP_ADMIT_TIME"]): "INIT"}

        for idx, time in enumerate(DATAForCurrID["ORDERS_ACTIVITY_START_TIME"]):
            temp[str(time)] = "ACTIVITY_START"
        for idx, time in enumerate(DATAForCurrID["ORDERS_ACTIVITY_STOP_TIME"]):
            temp[str(time)] = "ACTIVITY_STOP"

        for idx, time in enumerate(DATAForCurrID["ORDERS_NUTRITION_START_TIME"]):
            NUTRITION_TYPE = str(DATAForCurrID["ORDERS_NUTRITION_START_TIME"][idx])
            temp[str(time)] = f"NUTRITION_START_TYPE={NUTRITION_TYPE}"
        for idx, time in enumerate(DATAForCurrID["ORDERS_NUTRITION_STOP_TIME"]):
            NUTRITION_TYPE = str(DATAForCurrID["ORDERS_NUTRITION_START_TIME"][idx])
            temp[str(time)] = f"NUTRITION_STOP_TYPE={NUTRITION_TYPE}"

        for idx, time in enumerate(DATAForCurrID["LAB_RESULT_HRS_FROM_ADMIT"]):
            LAB_TYPE = str(DATAForCurrID["LAB_COMPONENT_ID"][idx])
            LAB_RESULT = str(DATAForCurrID["LAB_ORD_VALUE"][idx])
            temp[str(time)] = f"LAB_TYPE={LAB_TYPE}_RESULT={LAB_RESULT}"

        for idx, time in enumerate(DATAForCurrID["MEDICATION_TAKEN_HRS_FROM_ADMIT"]):
            MEDICATION_TYPE = str(DATAForCurrID["MEDICATION_ATC"][idx])
            MEDICATION_SIG = str(DATAForCurrID["MEDICATION_SIG"][idx])
            temp[str(time)] = f"MEDICATION_TYPE={MEDICATION_TYPE}_SIG={MEDICATION_SIG}"

        temp[str(DATAForCurrID["HOSP_DISCHRG_HRS_FROM_ADMIT"])] = "END"

        SEQUENCE = []
        for time in temp.keys():
            if time == "None":
                SEQUENCE.append(99999.9)
                continue
            SEQUENCE.append(float(time))
        try:
            SEQUENCE = sorted(SEQUENCE)
        except:
            a = 0
        ACTIONS = []
        for time in SEQUENCE:
            if time != 99999.9:
                ACTIONS.append(temp[str(time)])
            else:
                ACTIONS.append(temp["None"])
        SEQUENCE_DATA[ID] = {}
        SEQUENCE_DATA[ID]["SEQUENCE"] = SEQUENCE
        SEQUENCE_DATA[ID]["ACTIONS"] = ACTIONS

    with open(dirPath + '/FEATURE_DATA.json', mode='w') as file:
        json.dump(FEATURE_DATA, file, indent=4)
    with open(dirPath + '/SEQUENCE_DATA.json', mode='w') as file:
        json.dump(SEQUENCE_DATA, file, indent=4)

    # Overall DATASET:
    with open(dirPath + '/DATA.json', mode='w') as file:
        json.dump(DATA, file, indent=4)

def _getTableColumn(data: list, index: int) -> list:
    return [row[index] for row in data if len(row) > index]


def _initiateIDCase(DATA, ID):
    # todo: see __main__ for instructions
    # TABLE 01
    DATA[ID] = {}
    DATA[ID]["HOSP_ADMIT_TIME"] = 0.0
    DATA[ID]["HOSP_DISCHRG_HRS_FROM_ADMIT"] = None

    # TABLE 02
    DATA[ID]["WEIGHT_KG"] = None
    DATA[ID]["HEIGHT_CM"] = None
    DATA[ID]["AGE"] = None
    DATA[ID]["SEX"] = None

    # TABLE 04
    DATA[ID]["OR_PROC_ID"] = []
    DATA[ID]["OR_PROC_ID_ONEHOT"] = []

    # TABLE 05 & 06
    DATA[ID]["ORDERS_ACTIVITY_START_TIME"] = []
    DATA[ID]["ORDERS_ACTIVITY_STOP_TIME"] = []
    DATA[ID]["ORDERS_NUTRITION"] = []
    DATA[ID]["ORDERS_NUTRITION_ONEHOT"] = []
    DATA[ID]["ORDERS_NUTRITION_START_TIME"] = []
    DATA[ID]["ORDERS_NUTRITION_STOP_TIME"] = []

    # TABLE 09
    DATA[ID]["LAB_RESULT_HRS_FROM_ADMIT"] = []
    DATA[ID]["LAB_COMPONENT_ID"] = []
    DATA[ID]["LAB_COMPONENT_ID_ONEHOT"] = []
    DATA[ID]["LAB_ORD_VALUE"] = []

    # TABLE 10
    DATA[ID]["MEDICATION_ATC"] = []
    DATA[ID]["MEDICATION_ATC_ENCODED"] = []
    DATA[ID]["MEDICATION_TAKEN_HRS_FROM_ADMIT"] = []
    DATA[ID]["MEDICATION_SIG"] = []
    DATA[ID]["MEDICATION_ACTIONS"] = []
    DATA[ID]["MEDICATION_ACTIONS_ENCODED"] = []

    # TABLE 12
    DATA[ID]["PRIOR_MEDICATION_ATC_ENCODED"] = []
    DATA[ID]["PRIOR_MEDICATION_DISP_DAYS_NORM"] = []
    DATA[ID]["PRIOR_MEDICATION_DISP_DAYS_NORM_ONEHOT"] = []


def preprocess_01_ENCOUNTERS(DATA, filePath_01_ENCOUNTERS) -> dict:
    data_01_ENCOUNTERS = _dataInput_csv(filePath_01_ENCOUNTERS)

    STUDY_ID = _getTableColumn(data_01_ENCOUNTERS, 0)
    ENCOUNTER_NUM = _getTableColumn(data_01_ENCOUNTERS, 1)
    HOSP_DISCHRG_HRS_FROM_ADMIT = list(map(float, _getTableColumn(data_01_ENCOUNTERS, 4)))
    HOSP_DISCHRG_HRS_FROM_ADMIT = [round(time, 4) for time in HOSP_DISCHRG_HRS_FROM_ADMIT]
    WEIGHT_KG = list(map(float, _getTableColumn(data_01_ENCOUNTERS, 5)))
    HEIGHT_CM = list(map(float, _getTableColumn(data_01_ENCOUNTERS, 6)))
    AGE = list(map(int, _getTableColumn(data_01_ENCOUNTERS, 7)))
    SEX_str = _getTableColumn(data_01_ENCOUNTERS, 8)
    SEX = [0 if case == "MALE" else 1 for case in SEX_str]
    del SEX_str

    for idx, value in enumerate(STUDY_ID):
        ID = str(STUDY_ID[idx] + ENCOUNTER_NUM[idx])
        _initiateIDCase(DATA, ID)
        DATA[ID]["HOSP_DISCHRG_HRS_FROM_ADMIT"] = HOSP_DISCHRG_HRS_FROM_ADMIT[idx]
        DATA[ID]["WEIGHT_KG"] = WEIGHT_KG[idx]
        DATA[ID]["HEIGHT_CM"] = HEIGHT_CM[idx]
        DATA[ID]["AGE"] = AGE[idx]
        DATA[ID]["SEX"] = SEX[idx]

    print("√ 01_ENCOUNTERS")
    return DATA


def preprocess_02_ADMIT_DX(DATA, filePath_02_ADMIT_DX) -> dict:
    data_02_ADMIT_DX = _dataInput_csv(filePath_02_ADMIT_DX)

    STUDY_ID = _getTableColumn(data_02_ADMIT_DX, 0)
    ENCOUNTER_NUM = _getTableColumn(data_02_ADMIT_DX, 1)
    CURRENT_ICD10_LIST = _getTableColumn(data_02_ADMIT_DX, 3)

    diseaseIDs = dict()
    num = 0
    for idx, value in enumerate(STUDY_ID):
        for disease in CURRENT_ICD10_LIST[idx].split(", "):
            if not disease in diseaseIDs:
                diseaseIDs[disease] = num
                num += 1

    for idx, value in enumerate(STUDY_ID):
        ID = str(STUDY_ID[idx] + ENCOUNTER_NUM[idx])
        diseases = np.zeros(len(diseaseIDs))
        for disease in CURRENT_ICD10_LIST[idx].split(", "):
            diseases[diseaseIDs[disease]] = 1
        if DATA.get(ID) is None:
            _initiateIDCase(DATA, ID)
        DATA[ID]["DISEASES"] = list(diseases)

    print("√ 02_ADMIT_DX")
    return DATA


def preprocess_04_OR_PROC_ORDERS(DATA, filePath_04_OR_PROC_ORDERS) -> dict:
    data_04_OR_PROC_ORDERS = _dataInput_csv(filePath_04_OR_PROC_ORDERS)

    STUDY_ID = _getTableColumn(data_04_OR_PROC_ORDERS, 0)
    ENCOUNTER_NUM = _getTableColumn(data_04_OR_PROC_ORDERS, 1)
    OR_PROC_ID = _getTableColumn(data_04_OR_PROC_ORDERS, 2)

    procs = dict()
    num = 0
    for idx, value in enumerate(STUDY_ID):
        if not OR_PROC_ID[idx] in procs:
            procs[OR_PROC_ID[idx]] = num
            num += 1

    for idx, value in enumerate(STUDY_ID):
        ID = str(STUDY_ID[idx] + ENCOUNTER_NUM[idx])
        prod_ids = np.zeros(len(procs))
        prod_ids[procs[OR_PROC_ID[idx]]] = 1
        if DATA.get(ID) is None:
            _initiateIDCase(DATA, ID)
        DATA[ID]["OR_PROC_ID"].append(OR_PROC_ID[idx])
        if DATA[ID]["OR_PROC_ID_ONEHOT"] == []:
            DATA[ID]["OR_PROC_ID_ONEHOT"] = list(prod_ids)
        else:
            DATA[ID]["OR_PROC_ID_ONEHOT"] = [a + b for a, b in zip(DATA[ID]["OR_PROC_ID_ONEHOT"], list(prod_ids))]
    print("√ 04_OR_PROC_ORDERS")
    return DATA


def preprocess_05_ORDERS_ACTIVITY(DATA, filePath_05_ORDERS_ACTIVITY) -> dict:
    data_05_ORDERS_ACTIVITY = _dataInput_csv(filePath_05_ORDERS_ACTIVITY)

    STUDY_ID = _getTableColumn(data_05_ORDERS_ACTIVITY, 0)
    ENCOUNTER_NUM = _getTableColumn(data_05_ORDERS_ACTIVITY, 1)
    PROC_ID = _getTableColumn(data_05_ORDERS_ACTIVITY, 2)
    PROC_START_HRS_FROM_ADMIT = _getTableColumn(data_05_ORDERS_ACTIVITY, 6)
    ORDER_DISCON_HRS_FROM_ADMIT = _getTableColumn(data_05_ORDERS_ACTIVITY, 8)

    for idx, value in enumerate(STUDY_ID):
        ID = str(STUDY_ID[idx] + ENCOUNTER_NUM[idx])
        if DATA.get(ID) is None:
            _initiateIDCase(DATA, ID)

        DATA[ID]["ORDERS_ACTIVITY_START_TIME"].append(round(float(PROC_START_HRS_FROM_ADMIT[idx]), 4))
        DATA[ID]["ORDERS_ACTIVITY_STOP_TIME"].append(round(float(ORDER_DISCON_HRS_FROM_ADMIT[idx]), 4))

    print("√ 05_ORDERS_ACTIVITY")
    return DATA


def preprocess_06_ORDERS_NUTRITION(DATA, filePath_06_ORDERS_NUTRITION) -> dict:
    data_06_ORDERS_NUTRITION = _dataInput_csv(filePath_06_ORDERS_NUTRITION)

    STUDY_ID = _getTableColumn(data_06_ORDERS_NUTRITION, 0)
    ENCOUNTER_NUM = _getTableColumn(data_06_ORDERS_NUTRITION, 1)
    PROC_ID = _getTableColumn(data_06_ORDERS_NUTRITION, 2)
    PROC_START_HRS_FROM_ADMIT = _getTableColumn(data_06_ORDERS_NUTRITION, 6)
    ORDER_DISCON_HRS_FROM_ADMIT = _getTableColumn(data_06_ORDERS_NUTRITION, 8)

    procs = dict()
    num = 0
    for idx, value in enumerate(STUDY_ID):
        if not PROC_ID[idx] in procs:
            procs[PROC_ID[idx]] = num
            num += 1

    for idx, value in enumerate(STUDY_ID):
        ID = str(STUDY_ID[idx] + ENCOUNTER_NUM[idx])
        if DATA.get(ID) is None:
            _initiateIDCase(DATA, ID)

        prod_ids = np.zeros(len(procs))
        prod_ids[procs[PROC_ID[idx]]] = 1

        DATA[ID]["ORDERS_NUTRITION"].append(PROC_ID[idx])
        if DATA[ID]["ORDERS_NUTRITION_ONEHOT"] == []:
            DATA[ID]["ORDERS_NUTRITION_ONEHOT"] = list(prod_ids)
        else:
            DATA[ID]["ORDERS_NUTRITION_ONEHOT"] = [a + b for a, b in zip(DATA[ID]["ORDERS_NUTRITION_ONEHOT"], list(prod_ids))]
        DATA[ID]["ORDERS_NUTRITION_START_TIME"].append(round(float(PROC_START_HRS_FROM_ADMIT[idx]), 4))
        DATA[ID]["ORDERS_NUTRITION_STOP_TIME"].append(round(float(ORDER_DISCON_HRS_FROM_ADMIT[idx]), 4))

    print("√ 06_ORDERS_NUTRITION")
    return DATA


def preprocess_07_ACTIVITY_ORDER_QUESTIONS(DATA, filePath_07_ACTIVITY_ORDER_QUESTIONS) -> dict:
    # todo
    return DATA


def preprocess_08_NUTRITION_ORDER_QUESTIONS(DATA, filePath_08_NUTRITION_ORDER_QUESTIONS) -> dict:
    # todo
    return DATA


def preprocess_09_LABS(DATA, filePath_09_LABS) -> dict:
    data_09_LABS = _dataInput_csv(filePath_09_LABS)

    STUDY_ID = _getTableColumn(data_09_LABS, 0)
    ENCOUNTER_NUM = _getTableColumn(data_09_LABS, 1)
    RESULT_HRS_FROM_ADMIT = _getTableColumn(data_09_LABS, 3)
    COMPONENT_ID = _getTableColumn(data_09_LABS, 4)
    ORD_VALUE = _getTableColumn(data_09_LABS, 5)

    labs = dict()
    num = 0
    for idx, value in enumerate(STUDY_ID):
        if not COMPONENT_ID[idx] in labs:
            labs[COMPONENT_ID[idx]] = num
            num += 1

    for idx, value in enumerate(STUDY_ID):
        ID = str(STUDY_ID[idx] + ENCOUNTER_NUM[idx])
        if DATA.get(ID) is None:
            _initiateIDCase(DATA, ID)

        lab_ids = np.zeros(len(labs))
        lab_ids[labs[COMPONENT_ID[idx]]] = 1

        DATA[ID]["LAB_RESULT_HRS_FROM_ADMIT"].append(round(float(RESULT_HRS_FROM_ADMIT[idx]), 4))
        DATA[ID]["LAB_COMPONENT_ID"].append(COMPONENT_ID[idx])
        if DATA[ID]["LAB_COMPONENT_ID_ONEHOT"] == []:
            DATA[ID]["LAB_COMPONENT_ID_ONEHOT"] = list(lab_ids)
        else:
            DATA[ID]["LAB_COMPONENT_ID_ONEHOT"] = [a + b for a, b in zip(DATA[ID]["LAB_COMPONENT_ID_ONEHOT"], list(lab_ids))]
        if (ORD_VALUE[idx][0] == '>') or (ORD_VALUE[idx][0] == '<'):
            ORD_VALUE[idx] = ORD_VALUE[idx][1:]
        DATA[ID]["LAB_ORD_VALUE"].append(float(ORD_VALUE[idx]))
    print("√ 09_LABS")
    return DATA


def preprocess_10_MEDICATION_ADMINISTRATIONS_and_12_PIN(DATA, filePath_10_MEDICATION_ADMINISTRATIONS, filePath_12_PIN, toReload_ACTIONS_d2vModel=False, toReload_DRUGS_d2vModel=False) -> dict:
    data_10_MEDICATION_ADMINISTRATIONS = _dataInput_csv(filePath_10_MEDICATION_ADMINISTRATIONS)
    data_12_PIN = _dataInput_csv(filePath_12_PIN)

    STUDY_ID_10 = _getTableColumn(data_10_MEDICATION_ADMINISTRATIONS, 0)
    ENCOUNTER_NUM_10 = _getTableColumn(data_10_MEDICATION_ADMINISTRATIONS, 1)
    MEDICATION_ATC = _getTableColumn(data_10_MEDICATION_ADMINISTRATIONS, 2)
    TAKEN_HRS_FROM_ADMIT = _getTableColumn(data_10_MEDICATION_ADMINISTRATIONS, 4)
    MAR_ACTION = _getTableColumn(data_10_MEDICATION_ADMINISTRATIONS, 5)
    SIG = _getTableColumn(data_10_MEDICATION_ADMINISTRATIONS, 6)
    DOSE_UNIT = _getTableColumn(data_10_MEDICATION_ADMINISTRATIONS, 7)
    ROUTE = _getTableColumn(data_10_MEDICATION_ADMINISTRATIONS, 8)

    STUDY_ID_12 = _getTableColumn(data_12_PIN, 0)
    ENCOUNTER_NUM_12 = _getTableColumn(data_12_PIN, 1)
    DISP_DAYS_PRIOR = _getTableColumn(data_12_PIN, 2)
    SUPP_DRUG_ATC_CODE = _getTableColumn(data_12_PIN, 3)

    # Pre-merging:
    for idx, value in enumerate(STUDY_ID_10):
        ID = str(STUDY_ID_10[idx] + ENCOUNTER_NUM_10[idx])
        if DATA.get(ID) is None:
            _initiateIDCase(DATA, ID)

        DATA[ID]["MEDICATION_ATC"].append(str(MEDICATION_ATC[idx]))
        DATA[ID]["MEDICATION_SIG"].append(float(SIG[idx]))
        DATA[ID]["MEDICATION_TAKEN_HRS_FROM_ADMIT"].append(round(float(TAKEN_HRS_FROM_ADMIT[idx]), 4))
        DATA[ID]["MEDICATION_ACTIONS"].append(str(MAR_ACTION[idx] + " " + DOSE_UNIT[idx] + " " + ROUTE[idx]))

    for idx, value in enumerate(STUDY_ID_12):
        ID = str(STUDY_ID_12[idx] + ENCOUNTER_NUM_12[idx])
        if DATA.get(ID) is None:
            _initiateIDCase(DATA, ID)

        DATA[ID]["PRIOR_MEDICATION_ATC_ENCODED"].append(str(SUPP_DRUG_ATC_CODE[idx]))
        a = int(DISP_DAYS_PRIOR[idx])
        DISP_DAYS_PRIOR_NORM = int(DISP_DAYS_PRIOR[idx]) // 73     # --> Map 2 yr range to 0~10;
        DATA[ID]["PRIOR_MEDICATION_DISP_DAYS_NORM"].append(int(DISP_DAYS_PRIOR_NORM))

        if DATA[ID]["PRIOR_MEDICATION_DISP_DAYS_NORM_ONEHOT"] == []:
            DATA[ID]["PRIOR_MEDICATION_DISP_DAYS_NORM_ONEHOT"] = list(np.zeros(11))
        else:
            DATA[ID]["PRIOR_MEDICATION_DISP_DAYS_NORM_ONEHOT"][int(DISP_DAYS_PRIOR_NORM)] += 1


    # Create Encoders
    if toReload_ACTIONS_d2vModel:
        ACTIONS_d2vModel = Doc2Vec.load("src/features/encoders/ACTIONS_d2vModel.pkl")
    else:
        ACTIONS_Documents = []
        for ID in DATA.keys():
            ACTIONS_Documents.append(nltk.word_tokenize(' '.join(DATA[ID]['MEDICATION_ACTIONS']).strip()))
        ACTIONS_TDocuments = [TaggedDocument(words=[str(doc)], tags=[idx]) for idx, doc in enumerate(ACTIONS_Documents)]
        ACTIONS_d2vModel = Doc2Vec(documents=ACTIONS_TDocuments,
                                   vector_size=5,
                                   window=6,
                                   min_count=1,
                                   workers=10)
        ACTIONS_d2vModel.save("src/features/encoders/ACTIONS_d2vModel.pkl")

    if toReload_DRUGS_d2vModel:
        DRUGS_d2vModel = Doc2Vec.load("src/features/encoders/DRUGS_d2vModel.pkl")
    else:
        DRUGS_Documents = []
        for ID in DATA.keys():
            DRUGS_Documents.append(DATA[ID]['MEDICATION_ATC'])
            DRUGS_Documents.append(DATA[ID]['PRIOR_MEDICATION_ATC_ENCODED'])

        DRUGS_TDocuments = [TaggedDocument(words=[str(doc)], tags=[idx]) for idx, doc in enumerate(DRUGS_Documents)]
        DRUGS_d2vModel = Doc2Vec(documents=DRUGS_TDocuments,
                                 vector_size=5,
                                 window=6,
                                 min_count=1,
                                 workers=10)
        DRUGS_d2vModel.save("src/features/encoders/DRUGS_d2vModel.pkl")

    # Encoding Processes:
    for idx, value in enumerate(STUDY_ID_10):
        ID = str(STUDY_ID_10[idx] + ENCOUNTER_NUM_10[idx])
        DATA[ID]["MEDICATION_ATC_ENCODED"] = DRUGS_d2vModel.infer_vector(list(map(str, DATA[ID]["MEDICATION_ATC"]))).tolist()
        DATA[ID]["MEDICATION_ACTIONS_ENCODED"] = ACTIONS_d2vModel.infer_vector(nltk.word_tokenize(' '.join(DATA[ID]['MEDICATION_ACTIONS']).strip())).tolist()
    print("√ 10_MEDICATION_ADMINISTRATIONS")

    for idx, value in enumerate(STUDY_ID_12):
        ID = str(STUDY_ID_12[idx] + ENCOUNTER_NUM_12[idx])
        DATA[ID]["PRIOR_MEDICATION_ATC_ENCODED"] = DRUGS_d2vModel.infer_vector(list(map(str, DATA[ID]["PRIOR_MEDICATION_ATC_ENCODED"]))).tolist()
    print("√ 12_PIN")

    return DATA


def preprocess_11_MEDICATION_ORDERS(DATA, filePath_11_MEDICATION_ORDERS) -> dict:
    return DATA


def preprocessData() -> dict:
    DATA = {}

    """ Target File Paths """
    filePath_01_ENCOUNTERS = 'data/ENCOUNTERS.csv'
    filePath_02_ADMIT_DX = 'data/ADMIT_DX.csv'
    filePath_03_DX_LOOKUP = 'data/DX_LOOKUP.csv'
    filePath_04_OR_PROC_ORDERS = 'data/OR_PROC_ORDERS.csv'
    filePath_05_ORDERS_ACTIVITY = 'data/ORDERS_ACTIVITY.csv'
    filePath_06_ORDERS_NUTRITION = 'data/ORDERS_NUTRITION.csv'
    filePath_07_ACTIVITY_ORDER_QUESTIONS = 'data/ACTIVITY_ORDER_QUESTIONS.csv'  # currently ignored
    filePath_08_NUTRITION_ORDER_QUESTIONS = 'data/NUTRITION_ORDER_QUESTIONS.csv'  # currently ignored
    filePath_09_LABS = 'data/LABS.csv'
    filePath_10_MEDICATION_ADMINISTRATIONS = 'data/MEDICATION_ADMINISTRATIONS.csv'
    filePath_11_MEDICATION_ORDERS = 'data/MEDICATION_ORDERS.csv'
    filePath_12_PIN = 'data/PIN.csv'

    """ Data Preprocessing """
    # 01_ENCOUNTERS
    DATA = preprocess_01_ENCOUNTERS(DATA, filePath_01_ENCOUNTERS)
    # 02_ADMIT_DX & 03_DX_LOOKUP
    DATA = preprocess_02_ADMIT_DX(DATA, filePath_02_ADMIT_DX)
    # 04_OR_PROC_ORDERS
    DATA = preprocess_04_OR_PROC_ORDERS(DATA, filePath_04_OR_PROC_ORDERS)
    # 05_ORDERS_ACTIVITY
    DATA = preprocess_05_ORDERS_ACTIVITY(DATA, filePath_05_ORDERS_ACTIVITY)
    # 06_ORDERS_NUTRITION
    DATA = preprocess_06_ORDERS_NUTRITION(DATA, filePath_06_ORDERS_NUTRITION)
    # 07_ACTIVITY_ORDER_QUESTIONS       # todo: relations needed b/w tables, currently ignored;
    DATA = preprocess_07_ACTIVITY_ORDER_QUESTIONS(DATA, filePath_07_ACTIVITY_ORDER_QUESTIONS)
    # 08_NUTRITION_ORDER_QUESTIONS      # todo: relations needed b/w tables, currently ignored;
    DATA = preprocess_08_NUTRITION_ORDER_QUESTIONS(DATA, filePath_08_NUTRITION_ORDER_QUESTIONS)
    # 09_LABS
    DATA = preprocess_09_LABS(DATA, filePath_09_LABS)
    # 10_MEDICATION_ADMINISTRATIONS & 12_PIN
    DATA = preprocess_10_MEDICATION_ADMINISTRATIONS_and_12_PIN(DATA, filePath_10_MEDICATION_ADMINISTRATIONS,
                                                               filePath_12_PIN)
    # 11_MEDICATION_ORDERS              # todo: almost same to table 10, currently ignored;
    DATA = preprocess_11_MEDICATION_ORDERS(DATA, filePath_11_MEDICATION_ORDERS)

    return DATA


if __name__ == '__main__':
    """ Data Structure """
    # TODO
    #  dataPreprocessed =
    #   {
    #       ID_1: {
    #           HOSP_ADMIT_TIME: 0,                         <INIT_TIME: Relative time, which is ALWAYS 0>
    #           HOSP_DISCHRG_HRS_FROM_ADMIT: time,          <END_TIME: Relative time>
    #           WEIGHT_KG: float,
    #           HEIGHT_CM: float,
    #           AGE: int,
    #           SEX: int,
    #           DISEASES: [                                 * ENCODING: ONE-HOT APPLIED, TOTAL_VARIETIES=150 *
    #               diseaseID_1: int (idx),
    #               diseaseID_2: int (idx),
    #               ...
    #           ],
    #           OR_PROC_ID: [                               * ENCODING: ONE-HOT APPLIED, TOTAL_VARIETIES=108 *
    #               OR_PROC_ID_01,
    #               OR_PROC_ID_02,
    #               ...
    #           ],
    #           ORDERS_ACTIVITY_START_TIME: [               <ACTIVITY_TIME: Relative start time, all activities have same ID -> ignored>
    #               activity_01_startTime: time,
    #               activity_02_startTime: time,
    #               ...
    #           ],
    #           ORDERS_ACTIVITY_STOP_TIME: [                <ACTIVITY_TIME: Relative end time>
    #               activity_01_stopTime: time,
    #               activity_02_stopTime: time,
    #               ...
    #           ],
    #           ORDERS_NUTRITION: [                         * ENCODING: ONE-HOT APPLIED, TOTAL_VARIETIES=15 *
    #               nutrition_01: int,
    #               nutrition_02: int,
    #               ...
    #           ],
    #           ORDERS_NUTRITION_START_TIME: [              <NUTRITION_TIME: Relative start time>
    #               nutrition_01_startTime: time,
    #               nutrition_02_startTime: time,
    #               ...
    #           ],
    #           ORDERS_NUTRITION_STOP_TIME: [               <NUTRITION_TIME: Relative end time>
    #               nutrition_01_stopTime: time,
    #               nutrition_02_stopTime: time,
    #               ...
    #           ],
    #           LAB_RESULT_HRS_FROM_ADMIT: [                <TEST_TIME: Relative time>
    #               LAB_RESULT_TIME_1: time,
    #               LAB_RESULT_TIME_2: time,
    #               ...
    #           ],
    #           LAB_COMPONENT_ID: [                         * ENCODING: ONE-HOT APPLIED, TOTAL_VARIETIES=16 *
    #               LAB_COMPONENT_TIME_1: time,
    #               LAB_COMPONENT_TIME_2: time,
    #               ...
    #           ],
    #           LAB_ORD_VALUE: [                            ?????
    #               LAB_ORD_VALUE_TIME_1: float，
    #               LAB_ORD_VALUE_TIME_2: float，
    #               ...
    #           ],
    #           MEDICATION_ATC: [
    #               MEDICATION_ATC_1: str;
    #               MEDICATION_ATC_2: str;
    #               ...
    #           ],
    #           MEDICATION_ATC_ENCODED: [                   * (NEW) ENCODING: Doc2Vec APPLIED -> `DRUGS_d2vModel.pkl` *
    #               <Doc2Vec_ENCODED_VECTOR, Size=5>
    #               <Same as MEDICATION_ATC (kept for time checking)>
    #           ],
    #           MEDICATION_TAKEN_HRS_FROM_ADMIT: [          <TREATMENT_TIME: Relative time>
    #               MEDICATION_TAKEN_TIME_1: time;
    #               MEDICATION_TAKEN_TIME_2: time;
    #               ...
    #           ],
    #           MEDICATION_SIG: [                           ?????
    #               MEDICATION_SIG_1: float;
    #               MEDICATION_SIG_2: float;
    #               ...
    #           ],
    #           MEDICATION_ACTIONS: [                       * ENCODING: Doc2Vec *
    #               MEDICATION_ACTION_1: str;
    #               MEDICATION_ACTION_1: str;
    #               ...
    #           ],
    #           MEDICATION_ACTIONS_ENCODED: [               * ENCODING: Doc2Vec APPLIED -> `ACTIONS_d2vModel.pkl` *
    #               <Doc2Vec_ENCODED_VECTOR, Size=5>
    #               <Same as MEDICATION_ACTIONS (kept for time checking)>
    #           ],
    #           PRIOR_MEDICATION_ATC_ENCODED: [             * ENCODING: Doc2Vec APPLIED -> `DRUGS_d2vModel.pkl` *
    #              <Doc2Vec_ENCODED_VECTOR, Size=5>
    #           ],
    #           PRIOR_MEDICATION_DISP_DAYS_NORM: [          * ENCODING: ONE-HOT, TOTAL_VARIETIES=10, just as features not time *
    #               PRIOR_MEDICATION_DISP_DAYS_NORM_1: int;     --> [0, 10], indicates how far the time prior
    #               PRIOR_MEDICATION_DISP_DAYS_NORM_2: int;
    #               ...
    #           ],
    #       },
    #       ID_2: {
    #           ...
    #       },
    #       ...
    #   }

    """ Sequential Logics """
    # TODO
    #  ------> ADMIT_TIME ------> NUTRITION/ACTIVITY/TREATMENT/TEST_TIME ------> DISCHRG_TIME ------>
    #     [INIT, 0_hr]       [Comparing to ADMIT_TIME, i_hr, 0 < i < n]      [END, n_hr, Comparing to ADMIT_TIME]

    """ Data Cleaning and Preprocessing """
    dataPreprocessed = preprocessData()
    _dataOutput_json(dataPreprocessed)
