import pandas as pd
import numpy as np
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_percentage_error

class ModelloSolignano:
    """Prova documentazione: Classe base di forecasting per il livello della cisterna di Solignano"""
    def __init__(self):
        self.reg = XGBRegressor(colsample_bytree=0.5, eta=0.1, max_depth=8, subsample=0.75) # Inserire il regressore scelto con i suoi iperparametri. colsample_bytree=0.5, eta=0.1, max_depth=8, subsample=0.75 sono i migliori per ora
        self.lags = 48 # inserire il lag generalmente ottimale
        self.model = ForecasterAutoreg(self.reg, self.lags)

    def fit(self, trainingData:pd.Series):
        """
        Come il metodo fit() di ForecasterAutoreg, con solo i dati di training come input.
        
        Parameters
        ----------
        trainingData : pandas Series
            Training time series
        """
        self.model.fit(trainingData)

    def predict(self, steps:int, lastWindow:pd.Series = None):
        """
        Chiama predict() di ForecasterAutoreg.

        Parameters
        ----------
        steps : int
            Number of future steps predicted
        lastWindow : pandas Series, default None
            Series values used to create the predictors (lags) needed in the first iteration 
            of the prediction (t + 1). If lastWindow = None, the values stored 
            in self.last_window are used to calculate the initial predictors, 
            and the predictions start right after training data.

        Returns
        ----------
        pandas Series
            a Series with the predictions
        dict
            a dictionary containing warnings
        """
        preds = self.model.predict(steps, lastWindow)
        warnings = self._checkWarnings(preds)
        return preds, warnings

    def _checkWarnings(self, preds:pd.Series):
        """
        Metodo custom per i dati di solignano.
        Esplicita l'andamento previsto e prova ad informare riguardo possibili eventi.
        Credo si potrebbe addestrare un altro modello di classificazione solo per i warnings.

        Parameters
        ----------
        preds : pandas Series
            Series containing the model predictions

        Returns
        ---------
        dict
            a dictionary with information about the predicted trend and possible warnings about wter level
        """
        warnings = {"approaching": None, "trend": None, "delta_sum": None, "anomaly": None}

        for p in preds:
            if p < 140:
                warnings["approaching"] = "Previsto avvicinamento alla soglia inferiore di allarme"
                break
            if p > 290:
                warnings["approaching"] = "Previsto avvicinamento alla soglia superiore di allarme"
                break

        m = int(preds[-1] - preds[0])
        if m < 0:
            warnings["trend"] = f"Previsto un abbassamento del livello di {abs(m)}cm"
        if m > 0:
            warnings["trend"] = f"Previsto un innalzamento del livello di {m}cm"
        if m == 0:
            warnings["trend"] = "Previsto un andamento stabile"

        i=1
        mms = []
        while i < len(preds):
            mms.append(abs(preds[i] - preds[i - 1]))
            i+=1
        delta_sum = sum(mms)
        warnings["delta_sum"] = delta_sum
        if delta_sum < 1:
            warnings["anomaly"] = "!Possibile anomalia rilevata"

        return warnings
    
    def score(self, validationSet:pd.DataFrame, steps:int, warnings:bool = False):
        """
        Metodo per lo scoring del modello.

        Parameters
        ----------
        validationSet : pandas DataFrame
            data to score the prediction on
        steps : int
            Number of future steps predicted

        Returns
        ----------
        dict
            a dictionary containing mean absolute percentage error "mape" and
            the coefficient of determination "r_2"
        dict
            a dictionary containing warnings
        pandas Series
            a Series containing the predictions
        """
        i=0
        lenVal = len(validationSet)
        newPreds = []
        newTimes = []
        newWarns = []
        lags = self.lags
        while(i + lags < lenVal):
            print(f"step {i+1}/{lenVal - lags}", end="\r")
            batch = pd.Series(validationSet.iloc[i:i + lags].values.ravel(), validationSet.iloc[i:i + lags].index)
            betterPred, warnings = self.predict(steps , batch) # ignoro i warnings, ma non credo servano nella score
            newPreds.append(betterPred.iloc[-1])
            newTimes.append(betterPred.index[-1])
            newWarns.append(warnings)
            i += 1
        preds = pd.Series(newPreds, newTimes)
        warns = pd.Series(newWarns, newTimes)

        
        real = validationSet.loc[preds.iloc[:-steps].index].values
        pred = preds.iloc[:-steps].values
        mape = mean_absolute_percentage_error(real, pred)
        r_2 = r2_score(real, pred)

        return {"mape":mape, "r_2":r_2}, warns, preds