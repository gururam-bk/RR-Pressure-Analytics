
import pandas as pd
import numpy as np
from pathlib import Path

# ---------------- CONFIGURATION ----------------

BASE = Path(__file__).resolve().parent.parent

INPUT = BASE / "02_Cleaned_Data" / "04_player_raw_metrics.csv"
OUTPUT = BASE / "02_Cleaned_Data" / "05_player_pressure_scores.csv"

MINIMUMS = {
    "Batter": {"pressure": 24, "normal": 120},
    "Bowler": {"pressure": 18, "normal": 120}
}

COHORT_KEYS = ["role", "match_format", "season", "phase"]

# ---------------- LOAD AND AGGREGATE ----------------

df = pd.read_csv(INPUT, low_memory=False)

group_keys = [
    "role", "player_id", "match_format", "season", "phase"
]

metrics = (
    df.groupby(group_keys, as_index=False)
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

# Keep only cohorts with both pressure and normal exposure.
metrics = metrics[
    (metrics["pressure_balls"] > 0)
    & (metrics["normal_balls"] > 0)
].copy()

# ---------------- PLAYER ELIGIBILITY ----------------

exposure = (
    metrics.groupby(["role", "player_id"], as_index=False)
    .agg(
        pressure_balls=("pressure_balls", "sum"),
        normal_balls=("normal_balls", "sum")
    )
)

exposure["eligible"] = False

for role, limits in MINIMUMS.items():
    mask = (
        (exposure["role"] == role)
        & (exposure["pressure_balls"] >= limits["pressure"])
        & (exposure["normal_balls"] >= limits["normal"])
    )
    exposure.loc[mask, "eligible"] = True

eligible = exposure[exposure["eligible"]].copy()

print("Eligible players after valid-cohort filtering:")
print(eligible.groupby("role")["player_id"].nunique().to_string())

# ---------------- SHRINKAGE FUNCTIONS ----------------

def shrink_rate(count, trials):
    """Gamma-Poisson empirical-Bayes shrinkage for run rates."""

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
    """Beta-Binomial empirical-Bayes shrinkage."""

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
            observed_variance - sampling_variance, 0
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

    return (
        successes + alpha
    ) / (
        trials + alpha + beta
    )


# ---------------- COHORT PERFORMANCE ----------------

processed_groups = []

for cohort_key, group in metrics.groupby(COHORT_KEYS, sort=False):

    group = group.copy()
    role = cohort_key[0]

    # Shrunk run rates
    group["pressure_run_rate"] = shrink_rate(
        group["pressure_runs"],
        group["pressure_balls"]
    )

    group["normal_run_rate"] = shrink_rate(
        group["normal_runs"],
        group["normal_balls"]
    )

    # Shrunk dismissal / wicket probabilities
    group["pressure_out_rate"] = shrink_probability(
        group["pressure_outs"],
        group["pressure_balls"]
    )

    group["normal_out_rate"] = shrink_probability(
        group["normal_outs"],
        group["normal_balls"]
    )

    if role == "Batter":

        # Higher batting strike rate under pressure is better.
        group["scoring_delta"] = (
            group["pressure_run_rate"]
            - group["normal_run_rate"]
        ) * 100

        # Lower dismissal probability under pressure is better.
        group["dismissal_delta"] = (
            group["pressure_out_rate"]
            - group["normal_out_rate"]
        )

    else:

        # Lower bowling economy under pressure is better.
        group["scoring_delta"] = (
            group["pressure_run_rate"]
            - group["normal_run_rate"]
        ) * 6

        # Higher bowler wicket probability under pressure is better.
        group["dismissal_delta"] = (
            group["pressure_out_rate"]
            - group["normal_out_rate"]
        )

    processed_groups.append(group)

metrics = pd.concat(processed_groups, ignore_index=True)

# ---------------- COHORT STANDARDIZATION ----------------

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

# Direction correction:
# Batter: higher scoring improvement is better.
# Bowler: lower economy change is better.
metrics["scoring_component"] = np.where(
    metrics["role"] == "Bowler",
    -metrics["scoring_z"],
    metrics["scoring_z"]
)

# Second component:
# Batter: lower dismissal probability is better.
# Bowler: higher wicket probability is better.
metrics["second_component"] = np.where(
    metrics["role"] == "Batter",
    -metrics["dismissal_z"],
    metrics["dismissal_z"]
)

metrics["cohort_score"] = (
    0.5 * metrics["scoring_component"]
    + 0.5 * metrics["second_component"]
)

# Exposure weighting
metrics["cohort_weight"] = np.sqrt(
    np.minimum(
        metrics["pressure_balls"],
        metrics["normal_balls"]
    )
)

# ---------------- FINAL PLAYER RANKING ----------------

# Only globally eligible players enter the final ranking.
final_data = metrics.merge(
    eligible[["role", "player_id"]],
    on=["role", "player_id"],
    how="inner"
)

final_data["weighted_score"] = (
    final_data["cohort_score"]
    * final_data["cohort_weight"]
)

final_data["weighted_scoring"] = (
    final_data["scoring_component"]
    * final_data["cohort_weight"]
)

final_data["weighted_second"] = (
    final_data["second_component"]
    * final_data["cohort_weight"]
)

final = (
    final_data.groupby(["role", "player_id"], as_index=False)
    .agg(
        player_name=("player_name", "first"),
        total_pressure_balls=("pressure_balls", "sum"),
        total_normal_balls=("normal_balls", "sum"),
        cohorts_used=("cohort_score", "count"),
        total_weight=("cohort_weight", "sum"),
        weighted_score=("weighted_score", "sum"),
        weighted_scoring=("weighted_scoring", "sum"),
        weighted_second=("weighted_second", "sum")
    )
)

final["pressure_score"] = (
    final["weighted_score"] / final["total_weight"]
)

final["scoring_component"] = (
    final["weighted_scoring"] / final["total_weight"]
)

final["second_component"] = (
    final["weighted_second"] / final["total_weight"]
)

final["rank"] = (
    final.groupby("role")["pressure_score"]
    .rank(method="min", ascending=False)
    .astype(int)
)

final = final.sort_values(
    ["role", "rank"]
).reset_index(drop=True)

# ---------------- SAVE OUTPUT ----------------

columns = [
    "role",
    "rank",
    "player_id",
    "player_name",
    "pressure_score",
    "scoring_component",
    "second_component",
    "total_pressure_balls",
    "total_normal_balls",
    "cohorts_used"
]

final[columns].to_csv(OUTPUT, index=False)

# ---------------- SUMMARY ----------------

print("\nPlayers included in final ranking:")
print(final.groupby("role")["player_id"].nunique().to_string())

print("\nTop 10 Batters:")
print(
    final[final["role"] == "Batter"]
    .head(10)[["rank", "player_name", "pressure_score"]]
    .to_string(index=False)
)

print("\nTop 10 Bowlers:")
print(
    final[final["role"] == "Bowler"]
    .head(10)[["rank", "player_name", "pressure_score"]]
    .to_string(index=False)
)

print("\nOutput:", OUTPUT)
