# Imports (find in requirement.txt)
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier

from rppca import RPPCA
from ensemble import SoftVotingEnsemble
from qa import save_model_qa



# Demo dataset (no MT5 needed for live data, code may require further sophistication)

def make_synthetic_btc_dataset(n=6000, seed=42):

    rng = np.random.default_rng(seed)

    # Fake price
    returns = rng.normal(0, 1, size=n) * 0.5
    price = 50000 + np.cumsum(returns)

    # Fake indicators
    df = pd.DataFrame({
        "RSI": np.clip(50 + rng.normal(0, 12, size=n), 0, 100),
        "MACD": rng.normal(0, 30, size=n),
        "Signal": rng.normal(0, 30, size=n),
        "EMA_Cross": rng.normal(0, 50, size=n),
        "Stoch_K": np.clip(50 + rng.normal(0, 20, size=n), 0, 100),
        "Stoch_D": np.clip(50 + rng.normal(0, 20, size=n), 0, 100),
        "BB_Width": np.abs(rng.normal(200, 50, size=n)),
        "ATR": np.abs(rng.normal(120, 25, size=n)),
        "ATR_Ratio": np.abs(rng.normal(0.002, 0.0007, size=n)),
        "OBV": rng.normal(0, 1, size=n).cumsum(),
        "Close_Pct_Change": np.concatenate([[0], np.diff(price) / price[:-1] * 100]),
        "Net_News_Sentiment": rng.normal(0, 0.15, size=n),
        "Net_Social_Hype_Sentiment": rng.normal(0, 0.15, size=n),
    })


    labels = []
    for i in range(n):
        if df.loc[i, "RSI"] < 40 and df.loc[i, "MACD"] > df.loc[i, "Signal"]:
            labels.append("BUY")
        elif df.loc[i, "RSI"] > 60 and df.loc[i, "MACD"] < df.loc[i, "Signal"]:
            labels.append("SELL")
        else:
            labels.append("WAIT")

    df["Signal_Label"] = labels
    return df


def main():

    # Load dataset

    df = make_synthetic_btc_dataset(n=6000)

    y = df["Signal_Label"]
    X = df.drop(columns=["Signal_Label"]).apply(pd.to_numeric, errors="coerce").fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )


    # Class weights

    classes_sorted = np.array(sorted(pd.unique(y_train)))
    weights = compute_class_weight(class_weight="balanced", classes=classes_sorted, y=y_train)
    cw = {cls: float(w) for cls, w in zip(classes_sorted, weights)}


    # RP-PCA feature augmentation

    rp = RPPCA(n_components=8, rp_dim=16, power_iter=2, random_state=42).fit(X_train)
    X_train_rp = rp.transform(X_train)
    X_test_rp = rp.transform(X_test)

    X_train_aug = pd.concat([X_train, X_train_rp], axis=1)
    X_test_aug = pd.concat([X_test, X_test_rp], axis=1)

    print(f"[RP-PCA] Explained variance (proj space, summed): {rp.explained_variance_ratio_:.3f}")
    print("[RP-PCA] PC1 top drivers:", rp.top_features_by_pc(0, top_n=8))


    # Models

    rf = RandomForestClassifier(
        n_estimators=200, random_state=42, class_weight=cw, n_jobs=-1
    )
    lgbm = lgb.LGBMClassifier(
        n_estimators=300, random_state=42, class_weight=cw, verbose=-1
    )
    cat = CatBoostClassifier(
        iterations=400,
        learning_rate=0.05,
        depth=6,
        random_state=42,
        verbose=0,
        loss_function="MultiClass",
        class_weights=[cw.get(c, 1.0) for c in classes_sorted]
    )

    for m in (rf, lgbm, cat):
        m.fit(X_train_aug, y_train)

    # voting ensemble
    ensemble = SoftVotingEnsemble([rf, lgbm, cat]).fit(X_train_aug, y_train)


    # Evaluate

    y_pred = ensemble.predict(X_test_aug)
    proba = ensemble.predict_proba(X_test_aug)

    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc:.4f}\n")
    print(classification_report(y_test, y_pred, labels=sorted(pd.unique(y_test)), zero_division=0))


    # Quality assurance (qa) checks

    os.makedirs("outputs", exist_ok=True)
    qa = save_model_qa(
        y_true=y_test,
        y_pred=y_pred,
        proba=proba,
        labels=sorted(pd.unique(y_test)),
        out_dir="outputs/qa",
        rolling_window=200
    )
    print("\nSaved QA assets:", qa)
    print("\nDone. Check outputs/qa/ folder.")


if __name__ == "__main__":
    main()