import csv
import copy

def read_csv(filename, array):
    with open(filename) as csvfile: 
        reader = csv.reader(csvfile, delimiter=',') 
        for row in reader:
            array.append(row)

def combine(data, comb_sheet):
    length = len(data)
    for i in range(2, len(comb_sheet[0])):
        data[0].append(comb_sheet[0][i])

    missing_list = []
    for i in range(1, length):
        x = 0
        unmodified_row = copy.deepcopy(data[i])
        found = False
        for j in range(len(comb_sheet)):
            if data[i][0] == comb_sheet[j][0] and data[i][1] == comb_sheet[j][1]:
                found = True
                row = copy.deepcopy(unmodified_row)
                for k in range(2, len(comb_sheet[j])):
                    if x == 0:
                        data[i].append(comb_sheet[j][k])
                    else:
                        row.append(comb_sheet[j][k])
                if x != 0:
                    data.append(row)
                x = x + 1
        if not found:
            missing_list.append(i)
    
    for item in reversed(missing_list):
        data.pop(item)
    return data


if __name__ == '__main__':
    encounters_file = 'data/processed_encounters.csv'
    encounters = []
    read_csv(encounters_file, encounters)

    med_admin_file = 'data/processed_med_admin.csv'
    med_admin = []
    read_csv(med_admin_file, med_admin)

    admit_file = 'data/processed_admit.csv'
    admit = []
    read_csv(admit_file, admit)

    labs_file = 'data/processed_labs.csv'
    labs = []
    read_csv(labs_file, labs)

    data = combine(encounters, admit)
    data = combine(data, med_admin)
    data = combine(data, labs)

    with open('data/combined_data.csv', 'w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerows(data)