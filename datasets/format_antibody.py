import numpy as np
import csv

raw = list()
with open('antibody.csv', 'rt') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        raw.append(row)

headers = raw[0]
raw = raw[1:]

n_entries = len(raw)
data = np.zeros([n_entries, 6])

for ii in range(0, n_entries):
    chain_H = raw[ii][1]
    chain_L = raw[ii][2]
    model = int(raw[ii][3])

    HL = float(raw[ii][4]) * np.pi / 180
    HC1 = float(raw[ii][5]) * np.pi / 180
    HC2 = float(raw[ii][6]) * np.pi / 180
    LC1 = float(raw[ii][7]) * np.pi / 180
    LC2 = float(raw[ii][8]) * np.pi / 180
    dc = float(raw[ii][9])

    data[ii, :] = np.array([HL, HC1, HC2, LC1, LC2, dc])

np.save('antibody.npy', data)
