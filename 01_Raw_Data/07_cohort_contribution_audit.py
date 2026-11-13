
import pandas as pd
import numpy as np
from pathlib import Path

# ---------------- CONFIGURATION ----------------

BASE = Path(__file__).resolve().parent.parent
INPUT = BASE / "02_Cleaned_Data" / "04_player_raw_metrics.csv"

# Use corrected ranking, NOT the old baseline file.
RANKING = BASE / "02_Cleaned_Data" / "05_player_pressure_scores.csv"

MINIMUMS = {
    "Batter": {"pressure": 24, "normal": 120},
    "Bowler": {"pressure": 18, "normal": 120}
}

COHORT_KEYS = ["role", "match_format", "season", "phase"]

df = pd.read_csv(INPUT)

# ---------------- COHORT AGGREGATION ----------------

metrics = (
    df.groupby(
        ["role", "player_id", "match_format", "season", "phase"],
        as_index=False
    )
    .agg(
        player_name=("player_name", "first"),
        pressure_runs=("pressure_runs", "sum"),
        pressure_balls=("pressure_balls", "sum"),
        pressure_outs=("pressure_outs", "sum"),
        normal_runs=("normal_runs", "sum"),
        normal_balls=("normal_balls", "sum"),
        normal_outs=("normal_outs", "sum")
    )
)

# Only cohorts with both pressure and normal exposure.
metrics = metrics[
    (metrics.pressure_balls > 0) &
    (metrics.normal_balls > 0)
].copy()

# ---------------- ELIGIBILITY ----------------

exposure = metrics.groupby(
    ["role", "player_id"], as_index=False
).agg(
    pressure_balls=("pressure_balls", "sum"),
    normal_balls=("normal_balls", "sum")
)

exposure["eligible"] = False

for role, limits in MINIMUMS.items():

    mask = (
        (exposure.role == role) &
        (exposure.pressure_balls >= limits["pressure"]) &
        (exposure.normal_balls >= limits["normal"])
    )

    exposure.loc[mask, "eligible"] = True

eligible = exposure[
    exposure.eligible
][["role", "player_id"]]


# ---------------- SHRINKAGE ----------------

def shrink_rate(count, trials):

    count = np.asarray(count, dtype=float)
    trials = np.asarray(trials, dtype=float)

    observed = count / trials
    mean_rate = np.average(observed, weights=trials)

    if mean_rate <= 0:
        return np.zeros_like(observed)

    observed_variance = (
        np.var(observed, ddof=1) if len(observed) > 1 else 0.0
    )

    sampling_variance = np.mean(mean_rate / trials)

    between_variance = max(
        observed_variance - sampling_variance,
        mean_rate**2 / 1000
    )

    alpha = mean_rate**2 / between_variance
    beta = mean_rate / between_variance

    return (count + alpha) / (trials + beta)


def shrink_probability(successes, trials):

    successes = np.asarray(successes, dtype=float)
    trials = np.asarray(trials, dtype=float)

    observed = successes / trials
    mean_rate = np.average(observed, weights=trials)

    if len(observed) > 1:

        observed_variance = np.var(observed, ddof=1)

        sampling_variance = np.mean(
            observed * (1 - observed) / trials
        )

        between_variance = max(
            observed_variance - sampling_variance,
            0
        )

    else:
        between_variance = 0

    if between_variance > 0 and 0 < mean_rate < 1:

        concentration = (
            mean_rate * (1 - mean_rate) / between_variance
        ) - 1

        concentration = np.clip(concentration, 2, 1000)

    else:
        concentration = 100

    alpha = max(mean_rate * concentration, 0.01)
    beta = max((1 - mean_rate) * concentration, 0.01)

    return (successes + alpha) / (trials + alpha + beta)


# ---------------- COHORT SCORING ----------------

processed = []

for key, group in metrics.groupby(COHORT_KEYS, sort=False):

    group = group.copy()
    role = key[0]

    group["pressure_run_rate"] = shrink_rate(
        group.pressure_runs,
        group.pressure_balls
    )

    group["normal_run_rate"] = shrink_rate(
        group.normal_runs,
        group.normal_balls
    )

    group["pressure_out_rate"] = shrink_probability(
        group.pressure_outs,
        group.pressure_balls
    )

    group["normal_out_rate"] = shrink_probability(
        group.normal_outs,
        group.normal_balls
    )

    if role == "Batter":

        group["scoring_delta"] = (
            group.pressure_run_rate -
            group.normal_run_rate
        ) * 100

    else:

        group["scoring_delta"] = (
            group.pressure_run_rate -
            group.normal_run_rate
        ) * 6

    group["dismissal_delta"] = (
        group.pressure_out_rate -
        group.normal_out_rate
    )

    processed.append(group)

metrics = pd.concat(processed, ignore_index=True)


# ---------------- STANDARDIZATION ----------------

def standardize(series):

    std = series.std(ddof=0)

    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=series.index)

    return (series - series.mean()) / std


metrics["scoring_z"] = (
    metrics.groupby(COHORT_KEYS)["scoring_delta"]
    .transform(standardize)
)

metrics["dismissal_z"] = (
    metrics.groupby(COHORT_KEYS)["dismissal_delta"]
    .transform(standardize)
)

# Correct direction:
# Batter: higher scoring rate = better.
# Bowler: lower economy rate = better.
metrics["scoring_component"] = np.where(
    metrics["role"] == "Bowler",
    -metrics["scoring_z"],
    metrics["scoring_z"]
)

metrics["second_component"] = np.where(
    metrics.role == "Batter",
    -metrics.dismissal_z,
    metrics.dismissal_z
)

metrics["cohort_score"] = (
    0.5 * metrics.scoring_component +
    0.5 * metrics.second_component
)

metrics["cohort_weight"] = np.sqrt(
    np.minimum(
        metrics.pressure_balls,
        metrics.normal_balls
    )
)

# Keep only eligible players.
metrics = metrics.merge(
    eligible,
    on=["role", "player_id"],
    how="inner"
)


# ---------------- TOP 10 BOWLER AUDIT ----------------

ranking = pd.read_csv(RANKING)

top_bowlers = ranking[
    (ranking.role == "Bowler") &
    (ranking["rank"] <= 10)
][["role", "player_id", "player_name", "rank"]]

# Merge using both role and player ID.
audit = metrics.merge(
    top_bowlers,
    on=["role", "player_id"],
    suffixes=("", "_rank")
)

audit["weighted_contribution"] = (
    audit.cohort_score * audit.cohort_weight
)

audit["weighted_scoring"] = (
    audit.scoring_component * audit.cohort_weight
)

audit["weighted_second"] = (
    audit.second_component * audit.cohort_weight
)


# ---------------- COHORT CONTRIBUTIONS ----------------

print("\nTOP 10 BOWLER COHORT CONTRIBUTIONS")

print(
    audit.sort_values(
        ["rank", "match_format", "season", "phase"]
    )[
        [
            "rank",
            "player_name_rank",
            "match_format",
            "season",
            "phase",
            "pressure_balls",
            "normal_balls",
            "scoring_component",
            "second_component",
            "cohort_weight",
            "cohort_score",
            "weighted_contribution"
        ]
    ].to_string(index=False)
)


# ---------------- RECOMPUTED FINAL SCORES ----------------

summary = audit.groupby(
    ["rank", "player_id", "player_name_rank"],
    as_index=False
).agg(
    total_weight=("cohort_weight", "sum"),
    weighted_score=("weighted_contribution", "sum"),
    weighted_scoring=("weighted_scoring", "sum"),
    weighted_second=("weighted_second", "sum")
)

summary["recomputed_score"] = (
    summary.weighted_score / summary.total_weight
)

print("\nRECOMPUTED TOP 10")

print(
    summary.sort_values("rank").to_string(index=False)
)

# ---------------- CONSISTENCY CHECK ----------------

check = summary.merge(
    ranking[ranking.role == "Bowler"][
        ["player_id", "pressure_score"]
    ],
    on="player_id",
    how="left"
)

check["score_difference"] = (
    check["recomputed_score"] - check["pressure_score"]
).abs()

print("\nRANKING CONSISTENCY CHECK")

print(
    check[
        [
            "rank",
            "player_name_rank",
            "recomputed_score",
            "pressure_score",
            "score_difference"
        ]
    ].to_string(index=False)
)

print(
    "\nMaximum score difference:",
    check["score_difference"].max()
)

print("\nCohort contribution audit completed.")
