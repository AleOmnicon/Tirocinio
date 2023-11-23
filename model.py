import pandas as pd
import numpy as np
import os
import sys
from skforecast.ForecasterAutoreg import ForecasterAutoreg  ## fare riferimento alla documentazione di skforecast
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import time
from sklearn.metrics import r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt

# DATA PREP
PATH = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(PATH, "data", "trendLvSolignanoUlia.csv")

data = pd.read_csv(PATH,delimiter=";")
data.columns = ["Time", "Value"]

print(data)

dateLimiter = "2023-08-01 00:00:00"

stableData = data[data["Time"] < dateLimiter]
stableData["Time"] = pd.to_datetime(stableData["Time"])
stableData.set_index("Time", inplace=True)

resampledData = stableData.resample(rule="15T").mean().ffill()

setDivisor = int(len(resampledData) * 0.8)

trainSet = resampledData.iloc[:setDivisor]
validationSet = resampledData.iloc[setDivisor:]

strangeSet = data[data["Time"] >= dateLimiter]
strangeSet["Time"] = pd.to_datetime(strangeSet["Time"])
strangeSet.set_index("Time", inplace=True)
strangeSet = strangeSet.resample(rule="15T").mean().ffill()
# print(trainSet.index.to_series().diff().unique())
# print(trainSet)

# MODEL TRAIN
regr = XGBRegressor()
lags = 24*4

model = ForecasterAutoreg(regr, lags=lags)

start=time.time()
model.fit(trainSet["Value"])
end= time.time()
print(f"elapsed {int(end - start)} seconds")

preds = model.predict(len(validationSet)).to_frame()
preds["Real"] = validationSet
# print(preds)

print(preds["Real"])

score_r2 = r2_score(preds["Real"], preds["pred"])
# print(score_r2)

score_mape = mean_absolute_percentage_error(preds["Real"], preds["pred"])
# print(score_mape)

# plt.plot(preds["pred"], color="red", label="predizione")
# plt.plot(preds["Real"], color="blue", label="reali")
# plt.legend()
# plt.show()


# # MODELL MA MEGLIO
# i=0
# lenVal = len(validationSet)
# newPreds = []
# newTimes = []
# while(i + lags < lenVal):
#     print(f"step {i+1}/{lenVal - lags}", end="\r")
#     batch = validationSet.iloc[i:i+lags]["Value"]
#     betterPred = model.predict(4 , last_window=batch)
#     newPreds.append(betterPred.iloc[-1])
#     newTimes.append(betterPred.index[-1])
#     i += 1

# predSeries = pd.Series(newPreds, newTimes)

# plt.plot(predSeries, label="pred")
# plt.plot(validationSet, label="real")
# plt.legend()
# plt.show()

## STRANGE SET

i=0
lenVal = len(strangeSet)
newPreds = []
newTimes = []
while(i + lags < lenVal):
    print(f"step {i+1}/{lenVal - lags}", end="\r")
    batch = strangeSet.iloc[i:i+lags]["Value"]
    betterPred = model.predict(4 , last_window=batch)
    newPreds.append(betterPred.iloc[-1])
    newTimes.append(betterPred.index[-1])
    i += 1

predSeries = pd.Series(newPreds, newTimes)

# score_r2 = r2_score(preds["Real"], preds["pred"])
# print(score_r2)

# score_mape = mean_absolute_percentage_error(preds["Real"], preds["pred"])
# print(score_mape)

plt.plot(predSeries, label="pred")
plt.plot(strangeSet, label="real")
plt.legend()
plt.show()





