"""Podium predictor of the assistant: the Milestone 1 shallow NN, loaded from disk, with live SHAP.

Part of the prediction view. `PodiumPredictor` loads the artifacts exported by the last cell of
F1_podium_prediction.ipynb (milestone3/models/) once and never fits anything. For one set of
pre-race feature values it returns the podium probability, the class at the M1 threshold and the
local SHAP contributions, computed with the same `shap.DeepExplainer` and background as in M1.

SHAP values are on the log-odds (logit) scale, the network's output before the sigmoid:
base value + sum of contributions = the driver's logit. A positive value pushes towards a
podium, a negative one away; the sizes rank the features but are not probability points.

Self-test: python -m milestone3.prediction.predictor
"""
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
from torch import nn

from milestone3 import config
from milestone3.prediction import features


def build_net(n_inputs, hidden, dropout):
    """Rebuild the M1 ShallowNN network: Linear + ReLU (+ Dropout) per hidden layer, one output logit."""
    layers, width = [], n_inputs
    for units in hidden:
        layers += [nn.Linear(width, units), nn.ReLU()] + ([nn.Dropout(dropout)] if dropout else [])
        width = units
    return nn.Sequential(*layers, nn.Linear(width, 1))


class PodiumPredictor:
    """The saved M1 network plus its preprocessor and SHAP explainer."""

    def __init__(self, models_dir=config.MODELS_DIR):
        # artifacts exported by the notebook: metadata, preprocessing, weights, SHAP background
        self.meta = json.loads((models_dir / "meta.json").read_text())
        self.features = self.meta["features"]
        self.threshold = self.meta["threshold"]
        self.prep = joblib.load(models_dir / "preprocessor.joblib")
        saved = torch.load(models_dir / "nn.pt")
        self.net = build_net(saved["n_inputs"], saved["hidden"], saved["dropout"])
        self.net.load_state_dict(saved["state_dict"])
        self.net.eval()
        background = torch.tensor(np.load(models_dir / "shap_background.npy"))
        self.explainer = shap.DeepExplainer(self.net, background)
        self.base_value = float(np.asarray(self.explainer.expected_value).reshape(-1)[-1])
        self.global_importance = (pd.read_csv(models_dir / "global_shap.csv")
                                  [["feature", "mean absolute SHAP"]].to_dict("records"))

    def predict(self, values: dict) -> dict:
        """Probability, class and local SHAP for one driver-race; raises ValueError on invalid values."""
        errors = features.validate(values)
        if errors:
            raise ValueError(" ".join(errors))
        raw = pd.DataFrame([values])[self.features]
        z = torch.tensor(np.asarray(self.prep.transform(raw), dtype=np.float32))
        with torch.no_grad():
            probability = torch.sigmoid(self.net(z)).item()
        shap_values = np.asarray(self.explainer.shap_values(z)).reshape(-1)

        # (feature, original value, SHAP in log-odds), largest effect first
        contributions = sorted(zip(self.features, raw.iloc[0], shap_values), key=lambda c: -abs(c[2]))
        return {
            "probability": probability,
            "predicted_podium": probability >= self.threshold,
            "threshold": self.threshold,
            "contributions": [(f, float(v), float(s)) for f, v, s in contributions],
            "base_value": self.base_value,
            "global_importance": self.global_importance,
        }

    def waterfall_figure(self, prediction) -> plt.Figure:
        """SHAP waterfall of one prediction, labelled with the original (unscaled) feature values."""
        labels = {f.name: f.label for f in features.FEATURES}
        names, data, contribution = zip(*[(labels[f], v, s) for f, v, s in prediction["contributions"]])
        explanation = shap.Explanation(values=np.array(contribution), base_values=prediction["base_value"],
                                       data=np.array(data), feature_names=list(names))
        shap.plots.waterfall(explanation, max_display=len(names), show=False)
        fig = plt.gcf()
        fig.set_size_inches(7, 4.5)   # compact, so the text stays readable when the app scales it to a column
        fig.axes[0].set_title("Contributions in log-odds of a podium (red = towards a podium)", fontsize=10)
        return fig


if __name__ == "__main__":
    predictor = PodiumPredictor()
    cases = json.loads((config.EVAL_DIR / "explanation_cases.json").read_text())
    print(f"threshold {predictor.threshold}, base value {predictor.base_value:.3f} (log-odds)")

    # predict every explanation case from the pre-race values stored in the M1 model table
    for case in cases:
        values, filled, info = features.prefill(case["driverId"], case["race_name"], case["year"])
        result = predictor.predict(values)
        top = ", ".join(f"{f} {s:+.2f}" for f, _, s in result["contributions"][:3])
        print(f"{case['driver']:<18} {case['race_name']:<24} {case['year']}  p={result['probability']:.2f} "
              f"podium={result['predicted_podium']!s:<5} actual P{info['actual_position']}  top: {top}")
    predictor.waterfall_figure(result).savefig(config.RESULTS_DIR / "waterfall_example.png", bbox_inches="tight")

    # an out-of-range value is rejected: no probability is returned
    try:
        predictor.predict({**values, "grid_pos": 50})
    except ValueError as error:
        print("Rejected invalid input:", error)
