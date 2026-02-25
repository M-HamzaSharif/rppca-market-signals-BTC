#  Prob-averaging ensemble that aligns class order across models.
#-----------------------------------------------------------------------------------------------#


#Imports (find in requirements.txt)
import numpy as np



class SoftVotingEnsemble:



    def __init__(self, models):
        self.models = models
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = self.models[0].classes_
        return self


    def _proba_aligned(self, model, X):
        proba = model.predict_proba(X)
        if np.array_equal(model.classes_, self.classes_):
            return proba
        idx_map = [np.where(model.classes_ == c)[0][0] for c in self.classes_]
        return proba[:, idx_map]


    def predict_proba(self, X):
        probas = [self._proba_aligned(m, X) for m in self.models]
        return np.mean(probas, axis=0)



    def predict(self, X):
        p = self.predict_proba(X)
        return self.classes_[np.argmax(p, axis=1)]