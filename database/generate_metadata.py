#%% read from Info.cfg
header = ["id","ED","ES","Group","Height","NbFrame","Weight"]
rows = []

for num in range(1,151):
    id = "%03d" % num

    patient_mode = ['training', 'testing'][num > 100]
    config_file_path = f'database/{patient_mode}/patient{id}/Info.cfg'
    patient_data = {}

    with open(config_file_path) as f_in:
        for line in f_in:
            l = line.rstrip().split(": ")
            patient_data[l[0]] = l[1]

    row = [id] + list(patient_data.values())
    rows.append(row)

#%% save metadata to csv
import pandas as pd
df = pd.DataFrame(rows, columns=header)
df.to_csv('database/metadata.csv', index=False)