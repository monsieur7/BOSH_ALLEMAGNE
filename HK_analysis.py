import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import csv
import scipy.signal # signal
import scipy.stats
import math
file = open("C:/Users/nolan/Desktop/Data-Bosch/Data-Bosch/Daten.CSV")
csvreader = csv.reader(file)
header = next(csvreader)
print(header[7], header[8])
rows = []
time = []
HEAT = []
POWER = []
TEMP = []
i = 0
offset = 1
#TODO : CHANGE TO SECONDS
fs = 1/(15*60)
for row in csvreader:
    rows.append(row)
    #DeltaT = (round(float(row[5].replace(",", ".")), 2)) - round(float(row[4].replace(",", ".")), 2)
    HEAT.append((round(float(row[7].replace(",", ".")), 2)))
    POWER.append((round(float(row[8].replace(",", ".")), 2)))
    #TEMP.append((round(float(row[1].replace(",", ".")), 5)))
    time.append(i)
    i = i + 1
    if(i > 1000):
        break
plt.figure(1)
plt.scatter(HEAT, POWER)
slope, intercept, r, p, se = scipy.stats.linregress(HEAT, POWER)
t = np.linspace(min(HEAT), max(HEAT))
plt.plot(t, slope*t+intercept, "r")
plt.plot(t, slope*t+intercept+offset, "-g")
plt.plot(t, slope*t+intercept-offset, "-g")
plt.xlabel("FLUX")
plt.ylabel("POWER")
print("r value", r)

plt.show()


