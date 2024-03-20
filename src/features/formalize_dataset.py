import json
import math


def _dataInput_json(path) -> dict:
    with open(path, 'r') as f:
        return json.load(f)


def _dataOutput_json(DATA):
    with open("/output/DATASET.json", mode='w') as file:
        json.dump(DATA, file, indent=4)


def formalizeSequenceData(DATA, FEATURE_DATA, SEQUENCE_DATA):
    for sampleKey in SEQUENCE_DATA.keys():
        print(sampleKey)
        sequences = SEQUENCE_DATA[sampleKey]["SEQUENCE"]
        actions = SEQUENCE_DATA[sampleKey]["ACTIONS"]

        try:
            ATime = math.ceil(FEATURE_DATA[sampleKey]["HOSP_ADMIT_TOD"])
        except:
            break
        RTimeMax = math.ceil(max(sequences)) + 1
        for RTime in range(RTimeMax):
            # Compute Time Slots:
            dayNum = (ATime + RTime) // 24
            dayTime = (ATime + RTime) % 24
            uniqueSampleID = f"{sampleKey}|{str(dayNum)}|{str(dayTime)}"

            # Initialize Time Windows:
            DATA[uniqueSampleID] = {}
            DATA[uniqueSampleID]["#Day"] = dayNum
            DATA[uniqueSampleID]["#Time"] = dayTime
            DATA[uniqueSampleID]["Activity"] = 0

        for idx, sequence in enumerate(sequences):
            action = actions[idx]
            action_RTime = sequences[idx]
            if action_RTime >= RTimeMax:
                break
            if action == 'ACTIVITY_STOP':
                time = math.ceil(ATime + action_RTime)
                action_dayNum = time // 24
                action_dayTime = time % 24
                uniqueSampleID = f"{sampleKey}|{str(action_dayNum)}|{str(action_dayTime)}"
                DATA[uniqueSampleID]["Activity"] = 1
    test = 0
    return DATA


def formalizeFeatureData(DATA, FEATURE_DATA, SEQUENCE_DATA):
    for uniqueSampleID in DATA.keys():
        sampleKey = uniqueSampleID.split('|')[0]
        DATA[uniqueSampleID]["Weight"] = math.ceil(FEATURE_DATA[sampleKey]["WEIGHT_KG"])
        DATA[uniqueSampleID]["Height"] = math.ceil(FEATURE_DATA[sampleKey]["HEIGHT_CM"])
        DATA[uniqueSampleID]["Age"] = FEATURE_DATA[sampleKey]["AGE"]
        DATA[uniqueSampleID]["Sex"] = FEATURE_DATA[sampleKey]["SEX"]

        DATA[uniqueSampleID]["Operations"] = FEATURE_DATA[sampleKey]["OR_PROC_ID_ONEHOT"]
        DATA[uniqueSampleID]["MedActs"] = FEATURE_DATA[sampleKey]["MEDICATION_ACTIONS_ENCODED"]
        DATA[uniqueSampleID]["Diseases"] = FEATURE_DATA[sampleKey]["DISEASES"]
        DATA[uniqueSampleID]["PriorMed"] = FEATURE_DATA[sampleKey]["PRIOR_MEDICATION_ATC_ENCODED"]

    test = 1
    return DATA


def generateDataset(FEATURE_DATA_FilePath, SEQUENCE_DATA_FilePath):
    """ Input Preprocessed data """
    FEATURE_DATA = _dataInput_json(FEATURE_DATA_FilePath)
    SEQUENCE_DATA = _dataInput_json(SEQUENCE_DATA_FilePath)

    """ Formalize Datasets """
    DATA = {}
    DATA = formalizeSequenceData(DATA, FEATURE_DATA, SEQUENCE_DATA)
    DATA = formalizeFeatureData(DATA, FEATURE_DATA, SEQUENCE_DATA)

    #_dataOutput_json(FORMAL_DATA)


if __name__ == '__main__':
    generateDataset(
        FEATURE_DATA_FilePath := "output/FEATURE_DATA.json",
        SEQUENCE_DATA_FilePath := "output/SEQUENCE_DATA.json"
    )
