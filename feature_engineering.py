from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, cap_engagement=100):
        self.cap_engagement = cap_engagement

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["thumbs_ratio"] = X["thumbs_up"] / (X["thumbs_up"] + X["thumbs_down"] + 1)
        X["engagement"] = (X["reply_count"] + X["thumbs_up"] + X["thumbs_down"]).clip(upper=self.cap_engagement)
        X["stars_weighted"] = X["stars"] * X["engagement"]
        return X
