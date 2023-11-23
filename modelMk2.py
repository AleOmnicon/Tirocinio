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
from skforecast.model_selection import grid_search_forecaster

# CONST

DATE_LIMITER = "2023-08-01 00:00:00"
TRAIN_SET_PERC = 0.8
LAGS = [96, 192, 288, 384, 480] # 1,2,3,4,5 giorni
STEPS = 4 # 1h

# FUNC

def trainForecasters(regressors, lags, trSet):
    models = []
    for reg in regressors:
        for lag in lags:
            m = ForecasterAutoreg(regressor=reg, lags=lag)
            start=time.time()
            m.fit(trSet["Value"])
            end= time.time()
            ttime = end - start
            models.append([m, reg, lag ,ttime])
            print(f" {len(models)}/{len(regressors)*len(lags)}", end="\r")
    return pd.DataFrame(data=models, columns=["Model", "Regressor", "Lag", "Training Time"])

def predict(setToPredict ,lags, steps):
    i=0
    lenVal = len(setToPredict)
    newPreds = []
    newTimes = []
    while(i + lags < lenVal):
        print(f"step {i+1}/{lenVal - lags}", end="\r")
        batch = setToPredict.iloc[i:i + lags]["Value"]
        betterPred = model.predict(steps , last_window=batch)
        newPreds.append(betterPred.iloc[-1])
        newTimes.append(betterPred.index[-1])
        i += 1
    pred = pd.Series(newPreds, newTimes)
    print("\n")
    return pred

# DATA PREP

#all data
PATH = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(PATH, "data", "trendLvSolignanoUlia.csv")

allData = pd.read_csv(PATH,delimiter=";")
allData.columns = ["Time", "Value"]
allData["Time"] = pd.to_datetime(allData["Time"])
allData.set_index("Time", inplace=True)
allData = allData.resample(rule="15T").mean().ffill()

print("TUTTI I DATI:")
print(allData)

plt.plot(allData, label="Rilevazioni")

#stable data & training/validation 
stableData = allData[allData.index < DATE_LIMITER]

setDivisor = int(len(stableData) * TRAIN_SET_PERC)

trainingSet = stableData.iloc[:setDivisor]
validationSet = stableData.iloc[setDivisor:]

print("\nVALIDATION SET:")
print(validationSet)

#strange data
strangeData = allData[allData.index >= DATE_LIMITER]

print("\nSTRANGE SET:")
print(strangeData)


# MODEL TRAINING

# xgb = XGBRegressor()
# rf = RandomForestRegressor()
# regs = [xgb]

# fs = trainForecasters(regs, LAGS, trainingSet)

# print("\nTRAINED MODELS:")
# print(fs)

#base
xgb = XGBRegressor()
lags = LAGS[0]

model = ForecasterAutoreg(regressor=xgb, lags=lags)

start=time.time()
model.fit(trainingSet["Value"])
end= time.time()
print(f"\nTRAINING TIME: {(end - start):.3} s")

#aaa
# xgb = XGBRegressor()
# rf = RandomForestRegressor()
# lags = 24*4

# model = ForecasterAutoreg(regressor=xgb, lags=lags)

# results_grid = grid_search_forecaster(
#     forecaster  = model,
#     y           = trainingSet["Value"],
#     param_grid  = {},
#     lags_grid   = [LAGS[0]],
#     steps       = 4,
#     refit       = False,
#     metric      = ['mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error', 'mean_squared_log_error'],
#     initial_train_size = int(len(trainingSet["Value"])*0.5),
#     return_best = True,
#     verbose     = False
# )

# PREDIZIONI

# #sul training set
# start = time.time()
# predTraining = predict(trainingSet, LAGS[0], STEPS)
# plt.plot(trainingSet, label="predizione set training")
# end = time.time()
# print(f"predizione sul training set: {int(end-start)} s")

#sul validation set
start = time.time()
predVal = predict(validationSet, LAGS[0], STEPS)
plt.plot(predVal, label="predizione validation set")
end = time.time()
print(f"predizione sul validation set: {int(end-start)} s")
print(predVal.values)



# #sullo strange set
# start = time.time()
# predStrange = predict(strangeData, LAGS[0], STEPS)
# plt.plot(predStrange, label="predizione set anomalo")
# end = time.time()
# print(f"predizione sullo strange set: {int(end-start)} s")





plt.legend()
plt.show()
