
import pandas as pd
import numpy as np
from pathlib import Path

# ---------------- CONFIGURATION ----------------

BASE = Path(__file__).resolve().parent.parent
CLEAN = BASE / "02_Cleaned_Data"
RAW = BASE / "01_Raw_Data" / "Data Pack"

MINIMUMS = {
    "Batter": {"pressure": 24, "normal": 120},
    "Bowler": {"pressure": 18, "normal": 120}
}

COHORT_KEYS = ["role", "match_format", "season", "phase"]

# ---------------- LOAD DATA ----------------

df = pd.read_csv(CLEAN / "02_predelivery_match_state.csv")
matches = pd.read_csv(RAW / "matches.csv")
competitions = pd.read_csv(RAW / "competitions.csv")

match_info = matches[["match_id", "comp_id"]].merge(
    competitions[["comp_id", "match_format", "season"]],
    on="comp_id", how="left"
)

df = df.merge(match_info, on="match_id", how="left")

df["phase"] = "Middle"
df.loc[df["overid"].between(1, 6), "phase"] = "Powerplay"
df.loc[df["overid"].between(17, 20), "phase"] = "Death"

valid = df[(df["innings"] == 2) & df["target_valid"]]

rrr_base = valid.groupby(
    ["match_format", "season", "phase"]
)["required_run_rate"].agg(["mean", "std"])

df = df.join(
    rrr_base.rename(columns={
        "mean": "rrr_mean",
        "std": "rrr_std"
    }),
    on=["match_format", "season", "phase"]
)

first_scores = (
    df[df["innings"] == 1]
    .groupby(["match_id", "match_format", "season"])["team_runs"]
    .sum()
    .reset_index(name="score")
)

score_base = first_scores.groupby(
    ["match_format", "season"]
)["score"].agg(["mean", "std"])

df = df.join(
    score_base.rename(columns={
        "mean": "score_mean",
        "std": "score_std"
    }),
    on=["match_format", "season"]
)

# ---------------- PARTNERSHIP STATE ----------------

innings_groups = df.groupby(
    ["match_id", "innings"], sort=False
)

df["partnership_id"] = innings_groups["wicket_fall"].transform(
    lambda s: s.cumsum().shift(fill_value=0)
)

partnership_groups = df.groupby(
    ["match_id", "innings", "partnership_id"],
    sort=False
)

df["partnership_runs_before"] = (
    partnership_groups["team_runs"].transform(
        lambda s: s.cumsum().shift(fill_value=0)
    )
)

df["partnership_balls_before"] = (
    partnership_groups["is_batter_ball"].transform(
        lambda s: s.cumsum().shift(fill_value=0)
    )
)

df["partnership_sr_before"] = (
    100 * df["partnership_runs_before"]
    / df["partnership_balls_before"].replace(0, np.nan)
)

# ---------------- PRESSURE FLAGS ----------------

def make_flags(factor):

    high_rrr = (
        (df["innings"] == 2)
        & df["target_valid"]
        & (
            df["required_run_rate"]
            > df["rrr_mean"] + 1.5 * factor * df["rrr_std"]
        )
    )

    early_wickets = (
        df["overid"].between(1, 6)
        & (df["wickets_lost_before"] >= np.ceil(3 * factor))
    )

    middle_wickets = (
        (df["overid"] < 12)
        & (df["wickets_lost_before"] >= np.ceil(5 * factor))
    )

    resource = early_wickets | middle_wickets

    close = (
        (df["innings"] == 2)
        & df["target_valid"]
        & (df["runs_remaining"] > 0)
        & (df["runs_remaining"] <= 24 * factor)
        & (df["legal_balls_remaining"] <= 12 * factor)
    )

    batting = high_rrr | resource | close

    death = (
        (df["innings"] == 2)
        & df["target_valid"]
        & (df["legal_balls_remaining"] <= 30 * factor)
        & (df["wickets_in_hand"] >= np.ceil(5 * factor))
        & (df["required_run_rate"] >= 10.5 * factor)
    )

    low_target = (
        (df["innings"] == 2)
        & df["target_valid"]
        & df["overid"].between(1, 6)
        & (
            df["target"]
            <= df["score_mean"] - 1.5 * factor * df["score_std"]
        )
    )

    partnership = (
        (df["partnership_runs_before"] >= 50 * factor)
        & (df["partnership_sr_before"] >= 150 * factor)
    )

    bowling = death | low_target | partnership

    return batting.fillna(False), bowling.fillna(False)


# ---------------- PLAYER METRICS ----------------

def build_metrics(data, batting_pressure, bowling_pressure):

    data = data.copy()
    data["batting_pressure"] = batting_pressure
    data["bowling_pressure"] = bowling_pressure

    # Batter records
    batter = pd.DataFrame({
        "role": "Batter",
        "player_id": data["striker_id"],
        "player_name": data["striker_name"],
        "match_format": data["match_format"],
        "season": data["season"],
        "phase": data["phase"],
        "pressure": data["batting_pressure"],
        "runs": data["runs"],
        "balls": data["is_batter_ball"].astype(int),
        "outs": (
            data["wicket_fall"]
            & (data["dismissed_player_id"] == data["striker_id"])
        ).astype(int)
    })

    # Attribute non-striker dismissals to the actual dismissed player.
    extra = data[
        data["wicket_fall"]
        & data["dismissed_player_id"].notna()
        & (data["dismissed_player_id"] != data["striker_id"])
    ].copy()

    if not extra.empty:

        name_map = pd.concat([
            data[["striker_id", "striker_name"]].rename(
                columns={
                    "striker_id": "player_id",
                    "striker_name": "player_name"
                }
            ),
            data[["non_striker_id", "non_striker_name"]].rename(
                columns={
                    "non_striker_id": "player_id",
                    "non_striker_name": "player_name"
                }
            )
        ]).dropna().drop_duplicates("player_id")

        name_map = name_map.set_index("player_id")["player_name"]

        extra_batters = pd.DataFrame({
            "role": "Batter",
            "player_id": extra["dismissed_player_id"],
            "player_name": extra["dismissed_player_id"].map(name_map),
            "match_format": extra["match_format"],
            "season": extra["season"],
            "phase": extra["phase"],
            "pressure": extra["batting_pressure"],
            "runs": 0,
            "balls": 0,
            "outs": 1
        })

        batter = pd.concat(
            [batter, extra_batters],
            ignore_index=True
        )

    # Bowler records
    chargeable_types = ["wide", "Overstep", "other"]

    bowler_runs = (
        data["runs"]
        + np.where(
            data["extras_type"].isin(chargeable_types),
            data["extras1_runs"],
            0
        )
    )

    bowler_wickets = (
        data["dismissal_type"].isin(
            ["Bowled", "Caught", "LBW", "Stump", "Hit Wicket"]
        )
    )

    bowler = pd.DataFrame({
        "role": "Bowler",
        "player_id": data["bowler_id"],
        "player_name": data["bowler_name"],
        "match_format": data["match_format"],
        "season": data["season"],
        "phase": data["phase"],
        "pressure": data["bowling_pressure"],
        "runs": bowler_runs,
        "balls": data["is_legal_ball"].astype(int),
        "outs": bowler_wickets.astype(int)
    })

    records = pd.concat(
        [batter, bowler],
        ignore_index=True
    )

    records = records.dropna(subset=["player_id"])

    records["pressure_runs"] = np.where(
        records["pressure"], records["runs"], 0
    )
    records["pressure_balls"] = np.where(
        records["pressure"], records["balls"], 0
    )
    records["pressure_outs"] = np.where(
        records["pressure"], records["outs"], 0
    )

    records["normal_runs"] = np.where(
        ~records["pressure"], records["runs"], 0
    )
    records["normal_balls"] = np.where(
        ~records["pressure"], records["balls"], 0
    )
    records["normal_outs"] = np.where(
        ~records["pressure"], records["outs"], 0
    )

    return (
        records.groupby(
            [
                "role", "player_id", "player_name",
                "match_format", "season", "phase"
            ],
            as_index=False
        )[
            [
                "pressure_runs", "pressure_balls", "pressure_outs",
                "normal_runs", "normal_balls", "normal_outs"
            ]
        ].sum()
    )


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

    return (successes + alpha) / (trials + alpha + beta)


# ---------------- SCORING ENGINE ----------------

def calculate_ranking(raw):

    group_keys = [
        "role", "player_id", "match_format", "season", "phase"
    ]

    metrics = (
        raw.groupby(group_keys, as_index=False)
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

    # Only comparable cohorts.
    metrics = metrics[
        (metrics.pressure_balls > 0)
        & (metrics.normal_balls > 0)
    ].copy()

    exposure = (
        metrics.groupby(["role", "player_id"], as_index=False)
        [["pressure_balls", "normal_balls"]]
        .sum()
    )

    eligible = []

    for role, limits in MINIMUMS.items():

        selected = exposure[
            (exposure.role == role)
            & (exposure.pressure_balls >= limits["pressure"])
            & (exposure.normal_balls >= limits["normal"])
        ]

        eligible.append(selected[["role", "player_id"]])

    eligible = pd.concat(eligible, ignore_index=True)

    processed = []

    for cohort_key, group in metrics.groupby(
        COHORT_KEYS, sort=False
    ):

        group = group.copy()
        role = cohort_key[0]

        group["pressure_rate"] = shrink_rate(
            group.pressure_runs, group.pressure_balls
        )

        group["normal_rate"] = shrink_rate(
            group.normal_runs, group.normal_balls
        )

        group["pressure_out_rate"] = shrink_probability(
            group.pressure_outs, group.pressure_balls
        )

        group["normal_out_rate"] = shrink_probability(
            group.normal_outs, group.normal_balls
        )

        if role == "Batter":
            group["scoring_delta"] = (
                group.pressure_rate - group.normal_rate
            ) * 100
        else:
            group["scoring_delta"] = (
                group.pressure_rate - group.normal_rate
            ) * 6

        group["dismissal_delta"] = (
            group.pressure_out_rate - group.normal_out_rate
        )

        processed.append(group)

    metrics = pd.concat(processed, ignore_index=True)

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

    # Correct direction for both roles:
    # Batter: higher scoring rate is better.
    # Bowler: lower economy rate is better.
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
        0.5 * metrics.scoring_component
        + 0.5 * metrics.second_component
    )

    metrics["cohort_weight"] = np.sqrt(
        np.minimum(
            metrics.pressure_balls,
            metrics.normal_balls
        )
    )

    final_data = metrics.merge(
        eligible,
        on=["role", "player_id"],
        how="inner"
    )

    final_data["weighted_score"] = (
        final_data.cohort_score * final_data.cohort_weight
    )

    final = (
        final_data.groupby(["role", "player_id"], as_index=False)
        .agg(
            player_name=("player_name", "first"),
            weighted_score=("weighted_score", "sum"),
            total_weight=("cohort_weight", "sum")
        )
    )

    final["pressure_score"] = (
        final.weighted_score / final.total_weight
    )

    final["rank"] = (
        final.groupby("role")["pressure_score"]
        .rank(method="min", ascending=False)
    )

    return final.sort_values(
        ["role", "rank"]
    ).reset_index(drop=True)


# ---------------- RUN SCENARIOS ----------------

# Use the newly corrected ranking from 05_calculate_pressure_scores.py.
baseline = pd.read_csv(
    CLEAN / "05_player_pressure_scores.csv"
)

scenario_results = {}

for label, factor in [
    ("Lower", 0.90),
    ("Baseline", 1.00),
    ("Higher", 1.10)
]:

    batting, bowling = make_flags(factor)

    raw = build_metrics(df, batting, bowling)
    ranking = calculate_ranking(raw)

    scenario_results[label] = ranking

    ranking.to_csv(
        CLEAN / f"06_{label.lower()}_sensitivity_ranking.csv",
        index=False
    )

    print(f"\n{label} scenario")
    print("Eligible players:")
    print(ranking.groupby("role")["player_id"].nunique())

# ---------------- RANK STABILITY ----------------

print("\n========== RANK STABILITY ==========")

for role in ["Batter", "Bowler"]:

    base_top = baseline[
        baseline.role == role
    ].sort_values("rank").head(10)

    base_ids = set(base_top.player_id)

    print(f"\n{role.upper()}")

    for label in ["Lower", "Higher"]:

        alternative = scenario_results[label]

        alternative_role = alternative[
            alternative.role == role
        ]

        merged = baseline[
            baseline.role == role
        ][["player_id", "rank"]].merge(
            alternative_role[["player_id", "rank"]],
            on="player_id",
            suffixes=("_base", "_alt")
        )

        correlation = merged[
            ["rank_base", "rank_alt"]
        ].corr(method="spearman").iloc[0, 1]

        alt_top = set(
            alternative_role.sort_values("rank")
            .head(10).player_id
        )

        overlap = len(base_ids & alt_top)

        print(
            f"{label}: Spearman={correlation:.4f}, "
            f"Top-10 overlap={overlap}/10, "
            f"common eligible players={len(merged)}"
        )

        print("Alternative Top 10:")
        print(
            alternative_role.sort_values("rank")
            .head(10)[["rank", "player_name", "pressure_score"]]
            .to_string(index=False)
        )

print("\nSensitivity ranking analysis completed.")
print("Corrected baseline ranking file was not modified.")
