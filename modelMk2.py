import pandas as pd
import numpy as np
import os
import sys
from skforecast.ForecasterAutoreg import ForecasterAutoreg  ## fare riferimento alla documentazione di skforecast
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
import time
from sklearn.metrics import r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
from skforecast.model_selection import grid_search_forecaster

# CONST

DATE_LIMITER = "2023-08-01 00:00:00"
TRAIN_SET_PERC = 0.8
LAGS = [48, 96, 144, 192, 240, 288, 384, 480] # guardando le n/4 ore precedenti
STEPS = 4*48 # predici le *n ore successive
lags = LAGS[6]


fig, ax = plt.subplots()

# FUNC

def makePred(model, setToPredict ,lags, steps):
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

def printScore(realData, predictedData):
    real = realData.values
    pred = predictedData.values
    mape = mean_absolute_percentage_error(real, pred)
    r2 = r2_score(real, pred)

    info = f"___SCORE___\nmape: {mape:.5}\nr2: {r2:.5}"    
    ax.annotate(info, (predictedData.index[int(len(pred)*0.5)],50))    
    ax.fill_between(predictedData.index, pred+pred*mape, pred-pred*mape, color='C1', alpha=0.3)
    
    print("___SCORE___")
    print(f"mape: {mape:.5}")
    print(f"r2: {r2:.5}\n")


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

# MODEL TRAINING

#base
xgb = XGBRegressor(colsample_bytree=1, eta=0.1, max_depth=9, subsample=0.75)
regr = Pipeline([
    ("scaler", StandardScaler()),
    ("reg", xgb)
])

model = ForecasterAutoreg(regressor=regr, lags=lags)

start=time.time()
model.fit(trainingSet["Value"])
end= time.time()
print(f"\nTRAINING TIME: {(end - start):.3} s")

# PREDIZIONI

# #sul training set
# start = time.time()
# predVal = makePred(model, trainingSet, lags, STEPS)
# end = time.time()
# ax.plot(predVal, label="predizione Training set")
# print(f"predizione sul Training set: {int(end-start)} s")
# printScore(allData.loc[predVal.index], predVal)


#sul validation set
start = time.time()
predVal = makePred(model, validationSet, lags, STEPS)
end = time.time()
ax.plot(predVal, label="predizione Validation set")
print(f"predizione sul validation set: {int(end-start)} s")
printScore(allData.loc[predVal.index], predVal)


# #sul strange set
# start = time.time() 
# predVal = makePred(model, strangeData, lags, STEPS)
# ax.plot(predVal, label="predizione Strange set")
# end = time.time()
# print(f"predizione sulo Strange set: {int(end-start)} s")
# #evito di inserire le previsioni non presenti nei dati per fare lo score
# printScore(allData.loc[predVal.iloc[:-STEPS+1].index], predVal.iloc[:-STEPS+1])

print(f"Predette le {STEPS/4} ore successive\nguardando le {lags/4} ore precedenti")

plt.legend()
plt.show()
