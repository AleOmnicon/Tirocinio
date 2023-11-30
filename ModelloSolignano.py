import pandas as pd
import numpy as np
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.metrics import r2_score, mean_absolute_percentage_error

class ModelloSolignano:
    def __init__(self):
        self.reg = None # Inserire il regressore scelto con i suoi iperparametri
        self.lags = 192 # inserire il lag generalmente ottimale
        self.model = ForecasterAutoreg(self.reg, self.lags)

    def fit(self, trainingData):
        self.model.fit(trainingData)

    def predict(self, steps, lastWindow):
        preds = self.model.predict(steps, lastWindow)
        warnings = self._checkWarnings(preds)
        return preds, warnings

    def _checkWarnings(self, preds):
        warnings = []

        for p in preds:
            if p < 140:
                warnings.append("Previsto avvicinamento alla soglia inferiore di allarme")
                break
            if p > 290:
                warnings.append("Previsto avvicinamento alla soglia superiore di allarme")
                break

        m = int(preds[-1] - preds[0])
        if m < 0:
            warnings.append(f"Previsto un abbassamento del livello di {m}cm")
        if m > 0:
            warnings.append(f"Previsto un innalzamento del livello di {m}cm")
        if m == 0:
            warnings.append("Previsto un andamento stabile")

        i=1
        mms = []
        while i < len(preds):
            mms.append(abs(preds[i] - preds[i - 1]))
        if sum(mms) < 1:
            warnings.append("!Possibile anomalia rilevata")

        return warnings
    
    def score(self, validationSet, steps):        
        # per ora nella score ignoro i warnings
        i=0
        lenVal = len(validationSet)
        newPreds = []
        newTimes = []
        lags = self.lags[-1]
        while(i + lags < lenVal):
            batch = validationSet.iloc[i:i + lags]["Value"]
            betterPred = self.predict(steps , batch)
            newPreds.append(betterPred.iloc[-1])
            newTimes.append(betterPred.index[-1])
            i += 1
        preds = pd.Series(newPreds, newTimes)

        real = validationSet.iloc[steps:].values
        pred = preds.iloc[:-steps].values
        mape = mean_absolute_percentage_error(real, pred)
        r_2 = r2_score(real, pred)

        return mape, r_2