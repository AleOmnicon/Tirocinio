from ModelloSolignano import ModelloSolignano
import os
import pandas as pd
import matplotlib.pyplot as plt

DATE_LIMITER = "2023-08-01 00:00:00"
TRAIN_SET_PERC = 0.8
fig, ax = plt.subplots()
# DATA PREP

#all data
PATH = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(PATH, "data", "trendLvSolignanoUlia.csv")

allData = pd.read_csv(PATH,delimiter=";")
allData.columns = ["Time", "Value"]
allData["Time"] = pd.to_datetime(allData["Time"])
allData.set_index("Time", inplace=True)
allData = allData.resample(rule="60T").mean().ffill()

ax.plot(allData, label="Rilevazioni")

#stable data & training/validation 
stableData = allData[allData.index < DATE_LIMITER]

setDivisor = int(len(stableData) * TRAIN_SET_PERC)

trainingSet = stableData.iloc[:setDivisor]
validationSet = stableData.iloc[setDivisor:]

#strange data
strangeData = allData[allData.index >= DATE_LIMITER]

modelSOL = ModelloSolignano()

modelSOL.fit(trainingSet["Value"])

pred, warn = modelSOL.predict(48)
print(warn)

# SCORING SU TUTTO IL DATASET
# score, warn, preds = modelSOL.score(allData, 24)
# print(score)

# anom = []
# date = []
# i = 0
# n = len(warn)
# while i < n:
#     w = warn.iloc[i]
#     if w["anomaly"] != None:
#         anom.append(w)
#         date.append(warn.index[i])
#     i += 1

# if len(anom) > 0:
#     alies = pd.Series(anom, date)
#     ax.plot(preds.loc[alies.iloc[:-24].index], "ro", label="Anomalie")

# ax.plot(preds, label="Predizioni")

# SCORING SUL DATASET STABILE
score_stable, warns_stable, preds_stable = modelSOL.score(stableData, 24)
print(f"score su dataset stabile = {score_stable}")


plt.legend()
plt.show()