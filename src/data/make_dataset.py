import pandas as pd
import json


def main():

    df_map = pd.read_excel(
        "ACCESS 1853 Dataset.xlsx", sheet_name=None)

    encounters = df_map["ENCOUNTERS"]
    admit_dx = df_map["ADMIT_DX"]
    OR_orders = df_map["OR_PROC_ORDERS"]
    orders = df_map["ORDERS"]
    orders_qs = df_map["ORDER_QUESTIONS"]
    labs = df_map["LABS"]
    med_admin = df_map["MEDICATION_ADMINISTRATIONS"]
    med_orders = df_map["MEDICATION_ORDERS"]
    pin = df_map["PIN"]

    for key in df_map.keys():
        write_to_csv(df_map[key], key)


def write_to_csv(df_file, name):
    df_file.to_csv(name+".csv", index=None, header=True)


def preprocess_data(df_file, name):
    pass


if __name__ == "__main__":
    main()
