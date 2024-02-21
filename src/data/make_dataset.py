import pandas as pd
import json


def main():

    df_map = pd.read_excel(
        "data/ACCESS 1853 Dataset.xlsx", sheet_name=None, index_col=0)

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


def clean_admit(df_file):
    pass


def write_to_csv(df_file, name):
    df_file.to_csv("data/"+name+".csv", header=True)


def preprocess_data(df_file, name):
    pass


if __name__ == "__main__":
    main()
