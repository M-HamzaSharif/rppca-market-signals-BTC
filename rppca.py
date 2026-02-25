#Imports (find in requirements.txt)
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.preprocessing import RobustScaler


class RPPCA:

    def __init__(self, n_components=12, rp_dim=64, power_iter=2, random_state=42):
        self.n_components = n_components
        self.rp_dim = rp_dim
        self.power_iter = power_iter
        self.random_state = random_state

        self.scaler = RobustScaler(quantile_range=(10.0, 90.0))
        self.rp = None
        self.pca = None

        self.feature_names_ = None
        self.orig_loadings_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X: pd.DataFrame):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("RPPCA.fit expects a pandas DataFrame")

        self.feature_names_ = list(X.columns)

        Xs = self.scaler.fit_transform(X)
        rp_dim = min(self.rp_dim, X.shape[1])

        self.rp = GaussianRandomProjection(
            n_components=rp_dim,
            random_state=self.random_state
        )
        Z = self.rp.fit_transform(Xs)

        self.pca = PCA(
            n_components=min(self.n_components, rp_dim),
            svd_solver="randomized",
            iterated_power=self.power_iter,
            random_state=self.random_state
        )
        self.pca.fit(Z)

        self.explained_variance_ratio_ = float(self.pca.explained_variance_ratio_.sum())

        self.orig_loadings_ = self.rp.components_.T @ self.pca.components_.T  # (n_features, k)

        return self

    def _transform_core(self, X: pd.DataFrame):
        Xs = self.scaler.transform(X)
        Z = self.rp.transform(Xs)
        scores = self.pca.transform(Z)
        Z_hat = self.pca.inverse_transform(scores)
        recon_err = np.linalg.norm(Z - Z_hat, axis=1)
        return scores, recon_err

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("RPPCA.transform expects a pandas DataFrame")

        scores, recon_err = self._transform_core(X)
        k = scores.shape[1]

        out = pd.DataFrame(scores, index=X.index, columns=[f"RPPC{i+1}" for i in range(k)])
        out["RP_RECON_ERROR"] = recon_err
        return out

    def top_features_by_pc(self, pc_index=0, top_n=10):
        if self.orig_loadings_ is None or self.feature_names_ is None:
            return []
        w = np.abs(self.orig_loadings_[:, pc_index])
        idx = np.argsort(w)[::-1][:top_n]
        return [(self.feature_names_[i], float(w[i])) for i in idx]