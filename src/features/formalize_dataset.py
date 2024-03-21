import csv
import json
import math
import numpy as np
from copy import deepcopy


def _dataInput_json(inputPath) -> dict:
    print(f"\n>> Got Preprocessed Data from {inputPath}.")
    with open(inputPath, 'r') as f:
        return json.load(f)


def _dataOutput_json(DATA, outputPath="output/FormalizedDATA.json"):
    print("\n>> Printing Formalized Data to JSON...")
    with open(outputPath, mode='w') as file:
        json.dump(DATA, file, indent=4)
    print(f"\t> Formalized Data printed to {outputPath}...")


def _dataOutput_csv(DATA, outputPath="output/FormalizedDATA.csv"):
    print("\n>> Printing Formalized Data to CSV...")
    FEATURES = [
        "UniqueSampleID",
        "LabTests",
        "#Day",
        "#Time",
        "Med",
        "Activity",
        "Nutrition",
        "Weight",
        "Height",
        "Age",
        "Sex",
        "Operations",
        "MedActs",
        "Diseases",
        "PriorMed"
    ]

    with open(outputPath, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(FEATURES)

        for uniqueSampleID in sorted(DATA.keys()):
            dataline = [  # 1, 4, 6, 11, 12, 13, 14
                uniqueSampleID,
                DATA[uniqueSampleID].get(FEATURES[1], ""),
                DATA[uniqueSampleID].get(FEATURES[2], ""),
                DATA[uniqueSampleID].get(FEATURES[3], ""),
                DATA[uniqueSampleID].get(FEATURES[4], ""),
                DATA[uniqueSampleID].get(FEATURES[5], ""),
                DATA[uniqueSampleID].get(FEATURES[6], ""),
                DATA[uniqueSampleID].get(FEATURES[7], ""),
                DATA[uniqueSampleID].get(FEATURES[8], ""),
                DATA[uniqueSampleID].get(FEATURES[9], ""),
                DATA[uniqueSampleID].get(FEATURES[10], ""),
                DATA[uniqueSampleID].get(FEATURES[11], ""),
                DATA[uniqueSampleID].get(FEATURES[12], ""),
                DATA[uniqueSampleID].get(FEATURES[13], ""),
                DATA[uniqueSampleID].get(FEATURES[14], "")
            ]
            writer.writerow(dataline)
    print(f"\t> Formalized Data printed to {outputPath}...")


def _dataOutput_npy(DATA, outputPath="output/FormalizedDATA.npy"):
    print("\n>> Printing Formalized Data to NPY...")
    FEATURES = [
        "UniqueSampleID",
        "LabTests",
        "#Day",
        "#Time",
        "Med",
        "Activity",
        "Nutrition",
        "Weight",
        "Height",
        "Age",
        "Sex",
        "Operations",
        "MedActs",
        "Diseases",
        "PriorMed"
    ]

    DATAMATRIX = []
    for uniqueSampleID in sorted(DATA.keys()):
        # 1, 4, 6, 11, 12, 13, 14
        DATALINE = [
            uniqueSampleID,
            *DATA[uniqueSampleID].get(FEATURES[1]),
            DATA[uniqueSampleID].get(FEATURES[2]),
            DATA[uniqueSampleID].get(FEATURES[3]),
            *DATA[uniqueSampleID].get(FEATURES[4]),
            DATA[uniqueSampleID].get(FEATURES[5]),
            *DATA[uniqueSampleID].get(FEATURES[6]),
            DATA[uniqueSampleID].get(FEATURES[7]),
            DATA[uniqueSampleID].get(FEATURES[8]),
            DATA[uniqueSampleID].get(FEATURES[9]),
            DATA[uniqueSampleID].get(FEATURES[10]),
            *DATA[uniqueSampleID].get(FEATURES[11]),
            *DATA[uniqueSampleID].get(FEATURES[12]),
            *DATA[uniqueSampleID].get(FEATURES[13]),
            *DATA[uniqueSampleID].get(FEATURES[14])
        ]
        print(f"Array at index {uniqueSampleID} has shape: {len(DATALINE)}, size={len(DATA[uniqueSampleID].get(FEATURES[1]))}")

        dataline = [x if np.isscalar(x) else x for x in DATALINE]
        DATAMATRIX.append(dataline)

    np.save(outputPath, np.array(DATAMATRIX))
    print(f"\t> Formalized Data printed to {outputPath}...")


def formalizeSequenceData(DATA, FEATURE_DATA, SEQUENCE_DATA):
    print("\n>> Formalizing Sequence Data...")

    ONEHOT_DICT = _dataInput_json("output/ONEHOT_DICT.json")
    lab_num = len(ONEHOT_DICT["Lab"])
    med_num = len(ONEHOT_DICT["Med"])
    nutri_num = len(ONEHOT_DICT["Nutri"])

    for sampleKey in SEQUENCE_DATA.keys():
        sequences = SEQUENCE_DATA[sampleKey]["SEQUENCE"]
        actions = SEQUENCE_DATA[sampleKey]["ACTIONS"]

        try:
            ATime = math.ceil(FEATURE_DATA[sampleKey]["HOSP_ADMIT_TOD"])
        except:
            break
        RTimeMax = math.ceil(max(sequences)) + 1
        for RTime in range(RTimeMax):
            dayNum = (ATime + RTime) // 24
            dayTime = (ATime + RTime) % 24
            uniqueSampleID = f"{sampleKey}|{str(dayNum)}|{str(dayTime)}"

            DATA[uniqueSampleID] = {}
            DATA[uniqueSampleID]["#Day"] = dayNum
            DATA[uniqueSampleID]["#Time"] = dayTime
            DATA[uniqueSampleID]["Activity"] = 0
            DATA[uniqueSampleID]["LabTests"] = list(np.zeros(lab_num))
            DATA[uniqueSampleID]["Med"] = list(np.zeros(med_num))
            DATA[uniqueSampleID]["Nutrition"] = list(np.zeros(nutri_num))

        for idx, sequence in enumerate(sequences):
            action = actions[idx]
            action_RTime = sequences[idx]
            if action_RTime >= RTimeMax:
                break
            elif action == 'ACTIVITY_STOP':
                time = math.ceil(ATime + action_RTime)
                action_dayNum = time // 24
                action_dayTime = time % 24
                uniqueSampleID = f"{sampleKey}|{str(action_dayNum)}|{str(action_dayTime)}"
                DATA[uniqueSampleID]["Activity"] = 1
            elif action.split("=")[0] == "NUTRITION_STOP_TYPE":
                # Nutrition
                idxNutriDict = ONEHOT_DICT["Nutri"].index(action.split("=")[1])  # find the index of this nutrition
                DATA[uniqueSampleID]["Nutrition"][idxNutriDict] = 1  # assign the nutrition onehot
            elif action.split("|")[0].split("=")[0] == "LAB_TYPE":
                # LabTests
                idxLabDict = ONEHOT_DICT["Lab"].index(
                    action.split("|")[0].split("=")[1])  # find the index of this Lab,
                DATA[uniqueSampleID]["LabTests"][idxLabDict] = float(
                    action.split("|")[1].split("=")[1])  # and assign the corresponding result
            elif action.split("|")[0].split("=")[0] == "MEDICATION_TYPE":
                # Med
                idxMedDict = ONEHOT_DICT["Med"].index(action.split("|")[0].split("=")[1])  # find the index of this Med
                DATA[uniqueSampleID]["Med"][idxMedDict] = float(
                    action.split("|")[1].split("=")[1])  # and assign the corresponding sig

        for RTime in range(1, RTimeMax):
            prDayNum = (ATime + RTime - 1) // 24
            prDayTime = (ATime + RTime - 1) % 24
            prUniqueSampleID = f"{sampleKey}|{str(prDayNum)}|{str(prDayTime)}"
            dayNum = (ATime + RTime) // 24
            dayTime = (ATime + RTime) % 24
            uniqueSampleID = f"{sampleKey}|{str(dayNum)}|{str(dayTime)}"
            if all(x == 0 for x in DATA[uniqueSampleID]["LabTests"]):
                # if all the value in current time is 0, use the previous value
                DATA[uniqueSampleID]["LabTests"] = DATA[prUniqueSampleID]["LabTests"]

    print("\t> Sequence Data Formalized.")
    return DATA


def formalizeFeatureData(DATA, FEATURE_DATA, SEQUENCE_DATA):
    print("\n>> Formalizing Feature Data...")
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

    print("\t> Feature Data Formalized.")
    return DATA


def pruneData(DATA):
    DATA_pruned = deepcopy(DATA)
    for uniqueSampleID in DATA.keys():
        if DATA[uniqueSampleID]["#Time"] < 7:
            del DATA_pruned[uniqueSampleID]
    return DATA_pruned


def generateDataset(FEATURE_DATA_FilePath, SEQUENCE_DATA_FilePath):
    """ Input Preprocessed data """
    FEATURE_DATA = _dataInput_json(FEATURE_DATA_FilePath)
    SEQUENCE_DATA = _dataInput_json(SEQUENCE_DATA_FilePath)

    """ Formalize Datasets """
    DATA = {}
    DATA = formalizeSequenceData(DATA, FEATURE_DATA, SEQUENCE_DATA)
    DATA = formalizeFeatureData(DATA, FEATURE_DATA, SEQUENCE_DATA)
    DATA = pruneData(DATA)

    # _dataOutput_json(DATA, outputPath="output/FormalizedDATA.json")
    #_dataOutput_csv(DATA, outputPath="output/FormalizedDATA.csv")
    _dataOutput_npy(DATA)


if __name__ == '__main__':
    generateDataset(
        FEATURE_DATA_FilePath := "output/FEATURE_DATA.json",
        SEQUENCE_DATA_FilePath := "output/SEQUENCE_DATA.json",
    )
