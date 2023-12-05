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


#FIT
modelSOL = ModelloSolignano()
modelSOL.fit(trainingSet["Value"])


# # TEST ANOMALIE  non dovrebbe vederle  e infatti non le vede
# A_1st = "2023-08-16 10:00:00"# primo picco anomalo negativo
# A_2nd = "2023-09-15 23:00:00"#secondo
# A_3d = "2023-10-06 10:00:00"#terzo

# A_1_pred, A_1_warn = modelSOL.predict(24, strangeData[strangeData.index < A_1st].squeeze("columns"))
# A_2_pred, A_2_warn = modelSOL.predict(24, strangeData[strangeData.index < A_2nd].squeeze("columns"))
# A_3_pred, A_3_warn = modelSOL.predict(24, strangeData[strangeData.index < A_3d].squeeze("columns"))

# print(f"ANOMALY 1 = {A_1_warn}")
# print(f"ANOMALY 2 = {A_2_warn}")
# print(f"ANOMALY 3 = {A_3_warn}")

# ax.plot(A_1_pred)
# ax.plot(A_2_pred)
# ax.plot(A_3_pred)

d = [130.0, 300.0]
p = pd.to_datetime(["2023-08-01T00:00:00", "2023-08-01T00:00:00"])
alto = pd.Series(d, p)
ax.plot(alto, "k", label="Stabile | Instabile")

ax.hlines(y=[130.0, 300.0], xmin=allData.index[0], xmax=allData.index[-1], colors="r", linestyles="--")



# SCORING SUL DATASET INSTABILE
score_unst, warn_unst, preds_unst = modelSOL.score(strangeData, 24)
print(score_unst)

anom = []
date = []
i = 0
n = len(warn_unst)
while i < n:
    w = warn_unst.iloc[i]
    if w["anomaly"] != None:
        anom.append(w)
        date.append(warn_unst.index[i])
    i += 1

if len(anom) > 0:
    alies = pd.Series(anom, date)
    ax.plot(preds_unst.loc[alies.iloc[:-24].index], "ro", label="Anomalie")

ax.plot(preds_unst, label="Pred. strangeData")

scores = f"Score dati instabili\nmape = {score_unst['mape']:.3}\nr_2 = {score_unst['r_2']:.5}"
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.85, 0.15, scores, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', bbox=props)

pred = preds_unst.values
mape = score_unst["mape"]
ax.fill_between(preds_unst.index, pred+pred*mape, pred-pred*mape, color='C1', alpha=0.3)



# SCORING SUL DATASET STABILE
score_stable, warns_stable, preds_stable = modelSOL.score(stableData, 24)
print(f"score su dataset stabile = {score_stable}")

ax.plot(preds_stable, label="Pred. stableData")

scores = f"Score dati stabili\nmape = {score_stable['mape']:.3}\nr_2 = {score_stable['r_2']:.5}"
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.25, 0.5, scores, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', bbox=props)

pred = preds_stable.values
mape = score_stable["mape"]
ax.fill_between(preds_stable.index, pred+pred*mape, pred-pred*mape, color='C1', alpha=0.3)


plt.legend()
plt.show()