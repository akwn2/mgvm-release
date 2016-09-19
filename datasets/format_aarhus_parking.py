from datetime import datetime
import numpy as np
import csv

raw = list()
with open('aarhus_parking.csv', 'rt') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        raw.append(row)

# Geographical location of the parking spaces
locations = {
    'NORREPORT': (56.16184, 10.21284),
    'BUSGADEHUSET': (56.15561, 10.206),
    'BRUUNS': (56.14951, 10.20596),
    'SKOLEBAKKEN': (56.15659, 10.21353),
    'SCANDCENTER': (56.1527, 10.197),
    'SALLING': (56.15441, 10.20818),
    'MAGASIN': (56.15679, 10.2049),
    'KALKVAERKSVEJ': (56.14952, 10.21149)
}

# The Aarhus dataset has the following entries
# Vehicle count, Update time (time of measurement), ID (datapoint number), Total number of spaces, Transmission time
headers = raw[0]
raw = raw[1:]

n_entries = len(raw)
data = np.zeros([n_entries, 6])  # We're only interested in number of spaces, time of the day and location of the garage

for ii in range(0, n_entries):
    timestamp = datetime.strptime(raw[ii][1], '%d/%m/%y %H:%M')
    n_spaces = int(raw[ii][0])
    max_spaces = int(raw[ii][3])
    lat, lon = locations[raw[ii][4]]
    day = timestamp.day
    month = timestamp.month

    totd = (3600 * timestamp.hour + 60 * timestamp.minute) / (24 * 3600) * 2 * np.pi

    if n_spaces > max_spaces:  # Any overflows in counting are disconsidered
        n_spaces = 0

    data[ii, :] = np.array([totd, n_spaces, lon, lat, day, month])

np.save('aarhus_parking.npy', data)
