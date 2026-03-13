from sklearn.calibration import calibration_curve
import numpy as np

def compute_calibration(y_true, y_prob, bins=10):
    """
    Compute calibration curve values.
    """

    prob_true, prob_pred = calibration_curve(
        y_true,
        y_prob,
        n_bins=bins
    )

    return prob_true, prob_pred
