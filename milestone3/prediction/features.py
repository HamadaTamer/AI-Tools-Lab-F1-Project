"""The nine pre-race features of the Milestone 1 podium model: definitions, valid ranges, validation, prefill.

Part of the prediction view: the chatbot collects these values (from the user, from
`prefill`, or from the defaults below), checks them with `validate`, and only then hands
them to `PodiumPredictor`. Definitions follow `build_features` in F1_podium_prediction.ipynb.

Valid ranges are the minimum and maximum observed in outputs/model_table.csv (1950-2024, 24,724
driver-races), with the shares widened to 0-1: a value outside them has never happened, so the
model would be extrapolating. Defaults are the training-season medians, which is what the M1
imputer fills for a missing value. `python -m milestone3.prediction.features` rewrites
eval/results/feature_ranges.csv (min, 1st/99th percentile, median, max) as the evidence.
"""
from dataclasses import dataclass
from numbers import Real

import pandas as pd

from milestone3 import config

RACES_CSV = config.REPO_DIR / "milestone2" / "races.csv"
RESULTS_CSV = config.REPO_DIR / "milestone2" / "results.csv"


@dataclass(frozen=True)
class FeatureSpec:
    name: str      # column name used by the model
    label: str     # short name shown to the user
    unit: str
    help: str      # plain-language definition
    kind: str      # "int", "float" or "bool"
    min: float
    max: float
    default: float


FEATURES = [
    FeatureSpec("grid_pos", "Starting grid position", "position",
                "Where the driver lines up (1 = pole); a pit-lane start counts as the back of the field.",
                "int", 1, 34, 12),
    FeatureSpec("pit_lane_start", "Pit-lane start", "0 / 1",
                "1 if the driver starts from the pit lane, 0 otherwise.", "bool", 0, 1, 0),
    FeatureSpec("quali_pos", "Qualifying position", "position",
                "Position in the qualifying session (1 = fastest); can differ from the grid after penalties.",
                "int", 1, 26, 11),
    FeatureSpec("q1_gap_pct", "Q1 gap to the fastest", "%",
                "How much slower the driver's Q1 lap was than the fastest Q1 lap of the race (0 = fastest).",
                "float", 0, 44.8, 1.5),
    FeatureSpec("drv_standing", "Driver championship position", "position",
                "Driver's place in the championship after the previous round (1 = leader); none before round 1.",
                "int", 1, 102, 12),
    FeatureSpec("con_standing", "Team championship position", "position",
                "The team's place in the constructors' championship after the previous round (1 = leader).",
                "int", 1, 20, 6),
    FeatureSpec("drv_podium_last", "Recent podium rate", "share 0-1",
                "Share of the driver's last 3 races that ended on the podium (0, 0.33, 0.67 or 1).",
                "float", 0, 1, 0),
    FeatureSpec("drv_finish_last", "Recent finishing level", "share 0-1",
                "Average finishing place in the driver's last 3 races as a share of the field "
                "(near 0 = at the front, 1 = last).", "float", 0, 1, 0.53),
    FeatureSpec("drv_career_podium", "Career podium rate", "share 0-1",
                "Share of all the driver's earlier races that ended on the podium.", "float", 0, 1, 0.05),
]
SPECS = {f.name: f for f in FEATURES}


def validate(values: dict) -> list[str]:
    """Error messages for missing, non-numeric, non-integer or out-of-range values (empty list = valid)."""
    errors = []
    for f in FEATURES:
        v = values.get(f.name)
        if v is None:
            errors.append(f"{f.label} is missing.")
        elif isinstance(v, bool) or not isinstance(v, Real) or v != v:
            errors.append(f"{f.label} must be a number (got {v!r}).")
        elif f.kind in ("int", "bool") and v != int(v):
            errors.append(f"{f.label} must be a whole number (got {v}).")
        elif not f.min <= v <= f.max:
            errors.append(f"{f.label} must be between {f.min:g} and {f.max:g} (got {v:g}).")
    return errors


def _load(path, **kwargs):
    return pd.read_csv(path, na_values=["\\N"], **kwargs)


def prefill(driver_id, race_name=None, year=None):
    """Feature values for one driver-race taken from the M1 model table (leakage-safe, pre-race only).

    Returns (values, names_filled_from_data, race_info). Features missing in the data (e.g. no
    qualifying record in old seasons) and every feature of an unknown race get the default and are
    not listed as filled; race_info is None when the driver-race is not in the table. Without a
    year, the driver's most recent race of that name is used. The finishing position comes from
    the Milestone 2 results CSV (the same data the graph is built from).
    """
    values = {f.name: f.default for f in FEATURES}
    if race_name is None:
        return values, set(), None
    races = _load(RACES_CSV, usecols=["raceId", "year", "name"])
    races = races[races["name"].str.casefold() == race_name.casefold()]
    if year is not None:
        races = races[races["year"] == year]
    table = _load(config.MODEL_TABLE)
    rows = table[(table["driverId"] == driver_id) & table["raceId"].isin(races["raceId"])]
    if rows.empty:
        return values, set(), None
    row = rows.sort_values("date").iloc[-1]

    # values present in the table replace the defaults
    filled = {f.name for f in FEATURES if pd.notna(row[f.name])}
    values.update({n: row[n].item() for n in filled})
    for n in filled:
        if SPECS[n].kind != "float":
            values[n] = int(values[n])

    results = _load(RESULTS_CSV, usecols=["raceId", "driverId", "positionOrder"])
    finish = results[(results["raceId"] == row["raceId"]) & (results["driverId"] == driver_id)]
    race_info = {
        "raceId": int(row["raceId"]), "race_name": races.set_index("raceId").loc[row["raceId"], "name"],
        "year": int(row["year"]), "date": row["date"], "circuitId": int(row["circuitId"]),
        "constructorId": int(row["constructorId"]),
        "actual_position": int(finish["positionOrder"].iloc[0]) if len(finish) else None,
        "actual_podium": bool(row["podium"]),
    }
    return values, filled, race_info


def write_feature_ranges():
    """Evidence for the ranges and defaults: per-feature statistics of the model table."""
    table = _load(config.MODEL_TABLE)
    names = [f.name for f in FEATURES]
    stats = table[names].describe(percentiles=[0.01, 0.5, 0.99]).T[["min", "1%", "50%", "99%", "max"]]
    stats["train median"] = table.loc[table["split"] == "train", names].median()
    stats["valid min"] = [f.min for f in FEATURES]
    stats["valid max"] = [f.max for f in FEATURES]
    stats["default"] = [f.default for f in FEATURES]
    path = config.RESULTS_DIR / "feature_ranges.csv"
    path.parent.mkdir(exist_ok=True)
    stats.round(4).to_csv(path, index_label="feature")
    return stats


if __name__ == "__main__":
    print(write_feature_ranges().round(3))
