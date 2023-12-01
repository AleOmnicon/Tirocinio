import pandas as pd
import numpy as np
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_percentage_error

class ModelloSolignano:
    """Prova documentazione: Classe base di forecasting per il livello della cisterna di Solignano"""
    def __init__(self):
        self.reg = XGBRegressor() # Inserire il regressore scelto con i suoi iperparametri
        self.lags = 192 # inserire il lag generalmente ottimale
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
        tuple
            a tuple containing a Series with the predictions and a list of warnings
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
        warnings : list
            a list with information about the predicted trend and possible warnings about wter level
        """
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
            warnings.append(f"Previsto un abbassamento del livello di {abs(m)}cm")
        if m > 0:
            warnings.append(f"Previsto un innalzamento del livello di {m}cm")
        if m == 0:
            warnings.append("Previsto un andamento stabile")

        i=1
        mms = []
        while i < len(preds):
            mms.append(abs(preds[i] - preds[i - 1]))
            i+=1
        warnings.append(sum(mms))
        if sum(mms) < 5:
            warnings.append("!Possibile anomalia rilevata")

        return warnings
    
    def score(self, validationSet:pd.DataFrame, steps:int):
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
        """
        i=0
        lenVal = len(validationSet)
        newPreds = []
        newTimes = []
        lags = self.lags
        while(i + lags < lenVal):
            print(f"step {i+1}/{lenVal - lags}", end="\r")
            batch = validationSet.iloc[i:i + lags].values
            betterPred = self.model.predict(steps , batch) # ignoro i warnings, ma non credo servano nella score
            newPreds.append(betterPred.iloc[-1])
            newTimes.append(betterPred.index[-1])
            i += 1
        preds = pd.Series(newPreds, newTimes)

        
        real = validationSet.loc[preds.iloc[:-steps].index].values
        pred = preds.iloc[:-steps].values
        mape = mean_absolute_percentage_error(real, pred)
        r_2 = r2_score(real, pred)

        return {"mape":mape, "r_2":r_2}