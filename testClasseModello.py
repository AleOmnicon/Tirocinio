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
allData = allData.resample(rule="15T").mean().ffill()

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

predNwarn = modelSOL.predict(96)
print(predNwarn)