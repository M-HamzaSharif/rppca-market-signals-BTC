# Imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_fscore_support


def save_model_qa(
    y_true,
    y_pred,
    proba,
    labels,
    out_dir="outputs/qa",
    rolling_window=200
):

    os.makedirs(out_dir, exist_ok=True)

    y_true = np.array(list(y_true))
    y_pred = np.array(list(y_pred))

    # Rolling accuracy
    acc_series = pd.Series((y_pred == y_true).astype(int))
    rolling_acc = acc_series.rolling(
        rolling_window,
        min_periods=max(10, rolling_window // 5)
    ).mean()

    plt.figure(figsize=(7, 3.5))
    plt.plot(rolling_acc.values)
    plt.title(f"Rolling Accuracy (window={rolling_window})")
    plt.ylabel("Accuracy")
    plt.xlabel("Step")
    plt.tight_layout()
    rolling_path = os.path.join(out_dir, "rolling_accuracy.png")
    plt.savefig(rolling_path, dpi=160)
    plt.close()

    # Confusion matrix
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=labels, display_labels=labels
    )
    plt.title("Confusion Matrix")
    plt.tight_layout()
    cm_path = os.path.join(out_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=160)
    plt.close()

    # class metrics
    prf = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    pr_table = pd.DataFrame({
        "Label": labels,
        "Precision": prf[0],
        "Recall": prf[1],
        "F1": prf[2],
        "Support": prf[3],
    })
    pr_csv = os.path.join(out_dir, "per_class_metrics.csv")
    pr_table.to_csv(pr_csv, index=False)

    # Calibration plots
    calib_paths = []
    if proba is not None:
        proba = np.asarray(proba)
        for i, lab in enumerate(labels):
            y_true_i = (y_true == lab).astype(int)
            prob_i = proba[:, i]

            frac_pos, mean_pred = calibration_curve(
                y_true_i, prob_i, n_bins=10, strategy="uniform"
            )

            plt.figure(figsize=(4.6, 4.0))
            plt.plot(mean_pred, frac_pos, marker="o", label=str(lab))
            plt.plot([0, 1], [0, 1], linestyle="--")
            plt.title(f"Reliability – {lab}")
            plt.xlabel("Predicted probability")
            plt.ylabel("Observed frequency")
            plt.legend()
            plt.tight_layout()

            pth = os.path.join(out_dir, f"reliability_{lab}.png")
            plt.savefig(pth, dpi=160)
            plt.close()
            calib_paths.append(pth)

    return {
        "rolling_accuracy": rolling_path,
        "confusion_matrix": cm_path,
        "per_class_metrics": pr_csv,
        "calibration": calib_paths
    }