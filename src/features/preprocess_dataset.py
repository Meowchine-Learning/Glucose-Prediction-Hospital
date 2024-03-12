import csv
import json
from gensim.models.doc2vec import TaggedDocument, Doc2Vec


def _dataInput_csv(filePath: str) -> list:
    with open(filePath, mode='r', newline='') as file:
        reader = csv.reader(file)
        next(reader, None)
        IN = list(reader)
    return IN


def _dataOutput_json(DATA, filePath: str = "output/DATA.json") -> None:
    with open(filePath, mode='w') as file:
        json.dump(DATA, file, indent=4)


def _getTableColumn(data: list, index: int) -> list:
    return [row[index] for row in data if len(row) > index]


def _initiateIDCase(DATA, ID):
    # todo: see __main__ for instructions
    # TABLE 01
    DATA[ID] = {}
    DATA[ID]["HOSP_ADMIT_TIME"] = 0
    DATA[ID]["HOSP_DISCHRG_HRS_FROM_ADMIT"] = None

    # TABLE 02
    DATA[ID]["WEIGHT_KG"] = None
    DATA[ID]["HEIGHT_CM"] = None
    DATA[ID]["ACGE"] = None
    DATA[ID]["SEX"] = None

    # TABLE 03
    DATA[ID]["DISEASES"] = []

    # TABLE 04
    DATA[ID]["OR_PRO_ID"] = []

    # TABLE 05 & 06
    DATA[ID]["ORDERS_ACTIVITY"] = []
    DATA[ID]["ORDERS_ACTIVITY_START_TIME"] = []
    DATA[ID]["ORDERS_ACTIVITY_STOP_TIME"] = []
    DATA[ID]["ORDERS_NUTRITION"] = []
    DATA[ID]["ORDERS_NUTRITION_START_TIME"] = []
    DATA[ID]["ORDERS_NUTRITION_STOP_TIME"] = []

    # TABLE 09
    DATA[ID]["LAB_RESULT_HRS_FROM_ADMIT"] = []
    DATA[ID]["COMPONENT_ID"] = []
    DATA[ID]["ORD_VALUE"] = []

    # TABLE 10
    DATA[ID]["MEDICATION_ATC"] = []
    DATA[ID]["MEDICATION_TAKEN_HRS_FROM_ADMIT"] = []
    DATA[ID]["MEDICATION_SIG"] = []
    DATA[ID]["MEDICATION_ACTIONS"] = []

    # TABLE 12
    DATA[ID]["PRIOR_MEDICATION_ATC_CODE"] = []
    DATA[ID]["PRIOR_MEDICATION_DISP_DAYS_NORM"] = []


def preprocess_01_ENCOUNTERS(DATA, filePath_01_ENCOUNTERS) -> dict:
    data_01_ENCOUNTERS = _dataInput_csv(filePath_01_ENCOUNTERS)

    STUDY_ID = _getTableColumn(data_01_ENCOUNTERS, 0)
    ENCOUNTER_NUM = _getTableColumn(data_01_ENCOUNTERS, 1)
    HOSP_DISCHRG_HRS_FROM_ADMIT = _getTableColumn(data_01_ENCOUNTERS, 4)
    WEIGHT_KG = _getTableColumn(data_01_ENCOUNTERS, 5)
    HEIGHT_CM = _getTableColumn(data_01_ENCOUNTERS, 6)
    AGE = _getTableColumn(data_01_ENCOUNTERS, 7)
    SEX_str = _getTableColumn(data_01_ENCOUNTERS, 8)
    SEX = [0 if case == "MALE" else 1 for case in SEX_str]
    del SEX_str

    for idx, value in enumerate(STUDY_ID):
        ID = str(STUDY_ID[idx] + ENCOUNTER_NUM[idx])
        _initiateIDCase(DATA, ID)
        DATA[ID]["HOSP_DISCHRG_HRS_FROM_ADMIT"] = HOSP_DISCHRG_HRS_FROM_ADMIT
        DATA[ID]["WEIGHT_KG"] = WEIGHT_KG
        DATA[ID]["HEIGHT_CM"] = HEIGHT_CM
        DATA[ID]["AGE"] = AGE
        DATA[ID]["SEX"] = SEX
    return DATA


def preprocess_02_ADMIT_DX_and_03_DX_LOOKUP(DATA, filePath_02_ADMIT_DX, filePath_03_DX_LOOKUP) -> dict:
    data_02_ADMIT_DX = _dataInput_csv(filePath_02_ADMIT_DX)
    data_03_DX_LOOKUP = _dataInput_csv(filePath_03_DX_LOOKUP)

    diseaseIDs = {}
    for idx, value in enumerate(_getTableColumn(data_03_DX_LOOKUP, 0)):
        diseaseIDs[value] = idx

    STUDY_ID = _getTableColumn(data_02_ADMIT_DX, 0)
    ENCOUNTER_NUM = _getTableColumn(data_02_ADMIT_DX, 1)
    CURRENT_ICD10_LIST = _getTableColumn(data_02_ADMIT_DX, 3)

    for idx, value in enumerate(STUDY_ID):
        ID = str(STUDY_ID[idx] + ENCOUNTER_NUM[idx])
        diseases = [diseaseIDs[disease] for disease in CURRENT_ICD10_LIST[idx].split(", ")]
        if DATA.get(ID) is None:
            _initiateIDCase(DATA, ID)
        DATA[ID]["DISEASES"] = diseases
    return DATA


def preprocess_04_OR_PROC_ORDERS(DATA, filePath_04_OR_PROC_ORDERS) -> dict:
    data_04_OR_PROC_ORDERS = _dataInput_csv(filePath_04_OR_PROC_ORDERS)

    STUDY_ID = _getTableColumn(data_04_OR_PROC_ORDERS, 0)
    ENCOUNTER_NUM = _getTableColumn(data_04_OR_PROC_ORDERS, 1)
    OR_PROC_ID = _getTableColumn(data_04_OR_PROC_ORDERS, 2)

    for idx, value in enumerate(STUDY_ID):
        ID = str(STUDY_ID[idx] + ENCOUNTER_NUM[idx])
        if DATA.get(ID) is None:
            _initiateIDCase(DATA, ID)
        DATA[ID]["OR_PROC_ID"].append(OR_PROC_ID[idx])
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

        DATA[ID]["ORDERS_ACTIVITY"].append(PROC_ID)
        DATA[ID]["ORDERS_ACTIVITY_START_TIME"].append(PROC_START_HRS_FROM_ADMIT)
        DATA[ID]["ORDERS_ACTIVITY_STOP_TIME"].append(ORDER_DISCON_HRS_FROM_ADMIT)

    return DATA


def preprocess_06_ORDERS_NUTRITION(DATA, filePath_06_ORDERS_NUTRITION) -> dict:
    data_06_ORDERS_NUTRITION = _dataInput_csv(filePath_06_ORDERS_NUTRITION)

    STUDY_ID = _getTableColumn(data_06_ORDERS_NUTRITION, 0)
    ENCOUNTER_NUM = _getTableColumn(data_06_ORDERS_NUTRITION, 1)
    PROC_ID = _getTableColumn(data_06_ORDERS_NUTRITION, 2)
    PROC_START_HRS_FROM_ADMIT = _getTableColumn(data_06_ORDERS_NUTRITION, 6)
    ORDER_DISCON_HRS_FROM_ADMIT = _getTableColumn(data_06_ORDERS_NUTRITION, 8)

    for idx, value in enumerate(STUDY_ID):
        ID = str(STUDY_ID[idx] + ENCOUNTER_NUM[idx])
        if DATA.get(ID) is None:
            _initiateIDCase(DATA, ID)

        DATA[ID]["ORDERS_NUTRITION"].append(PROC_ID)
        DATA[ID]["ORDERS_NUTRITION_START_TIME"].append(PROC_START_HRS_FROM_ADMIT)
        DATA[ID]["ORDERS_NUTRITION_STOP_TIME"].append(ORDER_DISCON_HRS_FROM_ADMIT)

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

    for idx, value in enumerate(STUDY_ID):
        ID = str(STUDY_ID[idx] + ENCOUNTER_NUM[idx])
        if DATA.get(ID) is None:
            _initiateIDCase(DATA, ID)

        DATA[ID]["LAB_RESULT_HRS_FROM_ADMIT"].append(RESULT_HRS_FROM_ADMIT)
        DATA[ID]["LAB_COMPONENT_ID"].append(COMPONENT_ID)
        DATA[ID]["LAB_ORD_VALUE"].append(ORD_VALUE)

    return DATA


def preprocess_10_MEDICATION_ADMINISTRATIONS_and_12_PIN(DATA, filePath_10_MEDICATION_ADMINISTRATIONS, filePath_12_PIN) -> dict:
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

    drugMenu = list(set(MEDICATION_ATC + SUPP_DRUG_ATC_CODE))
    # drugMenu --> One-hot MODEL

    for idx, value in enumerate(STUDY_ID_10):
        ID = str(STUDY_ID_10[idx] + ENCOUNTER_NUM_10[idx])
        if DATA.get(ID) is None:
            _initiateIDCase(DATA, ID)

        DATA[ID]["MEDICATION_ATC"].append(MEDICATION_ATC)                  # todo One-hot encoding, same drugMenu
        DATA[ID]["MEDICATION_SIG"].append(SIG)
        DATA[ID]["MEDICATION_TAKEN_HRS_FROM_ADMIT"].append(TAKEN_HRS_FROM_ADMIT)
        DATA[ID]["MEDICATION_ACTIONS"].append(str(MAR_ACTION + DOSE_UNIT + ROUTE))

    for idx, value in enumerate(STUDY_ID_12):
        ID = str(STUDY_ID_12[idx] + ENCOUNTER_NUM_12[idx])      # todo reclean
        if DATA.get(ID) is None:
            _initiateIDCase(DATA, ID)

        DATA[ID]["PRIOR_MEDICATION_ATC_CODE"].append(MEDICATION_ATC)        # todo One-hot encoding, same drugMenu
        DISP_DAYS_PRIOR_NORM = [int(int(days) / 730 * 10) for days in DISP_DAYS_PRIOR]     # --> Map 2 yr range to 0~10;
        DATA[ID]["PRIOR_MEDICATION_DISP_DAYS_NORM"].append(DISP_DAYS_PRIOR_NORM)

    return DATA


def preprocess_11_MEDICATION_ORDERS(DATA, filePath_11_MEDICATION_ORDERS) -> dict:
    # todo
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
    DATA = preprocess_02_ADMIT_DX_and_03_DX_LOOKUP(DATA, filePath_02_ADMIT_DX, filePath_03_DX_LOOKUP)
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
    DATA = preprocess_10_MEDICATION_ADMINISTRATIONS_and_12_PIN(DATA, filePath_10_MEDICATION_ADMINISTRATIONS, filePath_12_PIN)
    # 11_MEDICATION_ORDERS              # todo: almost same to table 10, currently ignored;
    DATA = preprocess_11_MEDICATION_ORDERS(DATA, filePath_11_MEDICATION_ORDERS)

    return DATA


# * ENCODING: APPLY Doc2Vec *


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
    #           SEX: int,                                   * ENCODING: APPLY ONE-HOT, TOTAL_VARIETIES=2 *
    #           DISEASES: [                                 * ENCODING: APPLY ONE-HOT, TOTAL_VARIETIES=150 *
    #               diseaseID_1: int (idx),
    #               diseaseID_2: int (idx),
    #               ...
    #           ],
    #           OR_PROC_ID: [                               * ENCODING: APPLY ONE-HOT, TOTAL_VARIETIES=108 *
    #               OR_PROC_ID_01,
    #               OR_PROC_ID_02,
    #               ...
    #           ],
    #           ORDERS_ACTIVITY: [                          * ENCODING: ALL SAME, TOTAL_VARIETIES=1 *
    #               activity_01: int (idx),
    #               activity_02: int (idx),
    #               ...
    #           ],
    #           ORDERS_ACTIVITY_START_TIME: [               <ACTIVITY_TIME: Relative start time>
    #               activity_01_startTime: time,
    #               activity_02_startTime: time,
    #               ...
    #           ],
    #           ORDERS_ACTIVITY_STOP_TIME: [                <ACTIVITY_TIME: Relative end time>
    #               activity_01_stopTime: time,
    #               activity_02_stopTime: time,
    #               ...
    #           ],
    #           ORDERS_NUTRITION: [                         * ENCODING: APPLY ONE-HOT, TOTAL_VARIETIES=15 *
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
    #           LAB_COMPONENT_ID: [                         * ENCODING: APPLY ONE-HOT, TOTAL_VARIETIES=16 *
    #               LAB_COMPONENT_TIME_1: time,
    #               LAB_COMPONENT_TIME_2: time,
    #               ...
    #           ],
    #           LAB_ORD_VALUE: [                            ?????
    #               LAB_ORD_VALUE_TIME_1: float，
    #               LAB_ORD_VALUE_TIME_2: float，
    #               ...
    #           ],
    #           MEDICATION_ATC: [                           * ENCODING: APPLY ONE-HOT, share the same drugMenu, TOTAL_VARIETIES=??*
    #               MEDICATION_ATC_1: str;
    #               MEDICATION_ATC_2: str;
    #               ...
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
    #           MEDICATION_ACTIONS: [                       * ENCODING: APPLY Doc2Vec *
    #               MEDICATION_ACTION_1: str;
    #               MEDICATION_ACTION_1: str;
    #               ...
    #           ],
    #           PRIOR_MEDICATION_ATC_CODE: [                 * ENCODING: APPLY ONE-HOT, share the same drugMenu, TOTAL_VARIETIES=??*
    #               PRIOR_DRUG_TAKEN_ATC_CODE_1: str;
    #               PRIOR_DRUG_TAKEN_ATC_CODE_2: str;
    #               ...
    #           ],
    #           PRIOR_MEDICATION_DISP_DAYS_NORM: [          <TREATMENT_TIME_NORMED: As personal features, like height>
    #               PRIOR_MEDICATION_DISP_DAYS_NORM_1: int;     --> [0, 10]
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
    #  PRIOR_TAKEN_TIME ------> ADMIT_TIME ------> NUTRITION/ACTIVITY/TREATMENT/TEST_TIME ------> DISCHRG_TIME
    #      [INIT, <0 h]        [INIT, 0 h]             [Comparing to ADMIT_TIME]          [END, Comparing to ADMIT_TIME]

    """ Data Cleaning and Preprocessing """
    dataPreprocessed = preprocessData()
