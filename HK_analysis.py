import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import csv
import scipy.signal # signal
import scipy.stats
import math
from scipy.stats import multivariate_normal

def c_k(X, Y):
    c = []
    for i in range(0, len(Y)):
        if(Y[i] == 0):
            c.append(0)
        else:
            c.append(X[i]/Y[i])
    return c
file = open("Data-Bosch/Data-Bosch/Daten.CSV")
csvreader = csv.reader(file)
header = next(csvreader)
print(header[6], header[8])

rows = []
time = []
HEAT = []
POWER = []
TEMP = []
i = 0
offset = 1
time_step = 500*1e6 #every hour
#each 1 is 15min = 900sec
#TODO : CHANGE TO SECONDS
#FLOOR DATA
fs = 1/(15*60)
for row in csvreader:
    rows.append(row)
    DeltaT = (round(float(row[5].replace(",", ".")), 2)) - round(float(row[4].replace(",", ".")), 2)
    HEAT.append((round(float(row[3].replace(",", ".")), 2)*DeltaT))
    POWER.append((round(float(row[6].replace(",", ".")), 2)) + (round(float(row[6].replace(",", ".")), 2)))
    TEMP.append(round(DeltaT, 2))
    #TEMP.append((round(float(row[1].replace(",", ".")), 5)))
    time.append(i)
    i = i + 1
    if(i > 1000e6):
        break

plt.figure(1)
time_step = len(POWER)
C = c_k(POWER[0:time_step-1], HEAT[0:time_step-1])
plt.hist(np.array(C), density=True, alpha=0.6, color='g')

mu, sigma = scipy.stats.norm.fit(C)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = scipy.stats.norm.pdf(x, mu, sigma)
plt.plot(x, p, 'k', linewidth=2)
title = "Fit results: mu = %.2f,  std = %.2f" % (mu, sigma)
plt.title(title)
plt.ylabel("POWER")
plt.figure(2)
error = np.array(POWER[0:time_step-1]) - C*np.array(HEAT[0:time_step-1])
mu_error, sigma_error = scipy.stats.norm.fit(error)
plt.plot(time[0:time_step-1], POWER[0:time_step-1])

#plt.plot(time[0:time_step-1], np.array(HEAT[0:time_step-1])*mu+sigma, "r")

plt.plot(time[0:time_step-1], np.array(HEAT[0:time_step-1])*mu+sigma*2, "g")
plt.plot(time[0:time_step-1], np.array(HEAT[0:time_step-1])*mu-sigma*2, "g")
POWER_FALSE = []
TIME_FALSE = []
for i in range(0, len(POWER[0:time_step-1])):
    if(POWER[i] > (HEAT[i]*mu+sigma*3)) or (POWER[i] < (HEAT[i]*mu-sigma*3)):
        print("i", i, "POWER", POWER[i], "HEAT", HEAT[i], "-3*sigma", HEAT[i]*mu-sigma*3, "3sigma", HEAT[i]*mu+sigma*3)
        POWER_FALSE.append(POWER[i])
        TIME_FALSE.append(i)
plt.scatter(TIME_FALSE, POWER_FALSE, label='warning', c='r')
plt.show()



