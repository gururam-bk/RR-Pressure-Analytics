
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
CLEAN = BASE / "02_Cleaned_Data"
RAW = BASE / "01_Raw_Data" / "Data Pack"

df = pd.read_csv(CLEAN / "02_predelivery_match_state.csv")
matches = pd.read_csv(RAW / "matches.csv")
competitions = pd.read_csv(RAW / "competitions.csv")

# Add competition format and season
match_info = matches[["match_id", "comp_id"]].merge(
    competitions[["comp_id", "match_format", "season"]],
    on="comp_id",
    how="left"
)

df = df.merge(match_info, on="match_id", how="left")

# Match phase
df["phase"] = "Middle"
df.loc[df["overid"].between(1, 6), "phase"] = "Powerplay"
df.loc[df["overid"].between(17, 20), "phase"] = "Death"

# Batting pressure: RRR cohort threshold
valid_chases = df[
    (df["innings"] == 2) & df["target_valid"]
]

cohort = valid_chases.groupby(
    ["match_format", "season", "phase"]
)["required_run_rate"].agg(["mean", "std"])

df = df.join(
    cohort.rename(columns={
        "mean": "cohort_rrr_mean",
        "std": "cohort_rrr_std"
    }),
    on=["match_format", "season", "phase"]
)

df["high_rrr_pressure"] = (
    (df["innings"] == 2)
    & df["target_valid"]
    & (
        df["required_run_rate"]
        > df["cohort_rrr_mean"] + 1.5 * df["cohort_rrr_std"]
    )
)

# Batting pressure: early resource collapse
df["resource_collapse"] = (
    (
        (df["overid"].between(1, 6))
        & (df["wickets_lost_before"] >= 3)
    )
    |
    (
        (df["overid"] < 12)
        & (df["wickets_lost_before"] >= 5)
    )
)

# Batting pressure: close finish
df["close_finish"] = (
    (df["innings"] == 2)
    & df["target_valid"]
    & (df["runs_remaining"] > 0)
    & (df["runs_remaining"] <= 24)
    & (df["legal_balls_remaining"] <= 12)
)

df["batting_pressure"] = (
    df["high_rrr_pressure"]
    | df["resource_collapse"]
    | df["close_finish"]
)

# First-innings scoring baseline by format and season
first_scores = (
    df[df["innings"] == 1]
    .groupby(["match_id", "match_format", "season"])["team_runs"]
    .sum()
    .reset_index(name="first_innings_score")
)

score_baseline = first_scores.groupby(
    ["match_format", "season"]
)["first_innings_score"].agg(["mean", "std"])

df = df.join(
    score_baseline.rename(columns={
        "mean": "first_innings_mean",
        "std": "first_innings_std"
    }),
    on=["match_format", "season"]
)

# Bowling pressure: death-overs high-RRR defence
df["death_rrr_pressure"] = (
    (df["innings"] == 2)
    & df["target_valid"]
    & (df["legal_balls_remaining"] <= 30)
    & (df["wickets_in_hand"] >= 5)
    & (df["required_run_rate"] >= 10.5)
)

# Bowling pressure: defending an unusually low target
df["low_target_powerplay"] = (
    (df["innings"] == 2)
    & df["target_valid"]
    & (df["overid"].between(1, 6))
    & (
        df["target"]
        <= df["first_innings_mean"] - 1.5 * df["first_innings_std"]
    )
)

df["bowling_pressure"] = (
    df["death_rrr_pressure"]
    | df["low_target_powerplay"]
)

# Partnership momentum before each delivery
innings_groups = df.groupby(["match_id", "innings"], sort=False)

df["partnership_id"] = innings_groups["wicket_fall"].transform(
    lambda s: s.cumsum().shift(fill_value=0)
)

partnership_groups = df.groupby(
    ["match_id", "innings", "partnership_id"],
    sort=False
)

df["partnership_runs_before"] = partnership_groups["team_runs"].transform(
    lambda s: s.cumsum().shift(fill_value=0)
)

df["partnership_balls_before"] = partnership_groups["is_batter_ball"].transform(
    lambda s: s.cumsum().shift(fill_value=0)
)

df["partnership_sr_before"] = (
    100 * df["partnership_runs_before"]
    / df["partnership_balls_before"].replace(0, pd.NA)
)

df["partnership_momentum_pressure"] = (
    (df["partnership_runs_before"] >= 50)
    & (df["partnership_sr_before"] >= 150)
)

df["bowling_pressure"] = (
    df["bowling_pressure"]
    | df["partnership_momentum_pressure"]
)


# Save pressure-flagged deliveries
output = CLEAN / "03_pressure_flagged_deliveries.csv"
df.to_csv(output, index=False)

print("Pressure flags created.")
print("Total deliveries:", len(df))
print("Batting pressure deliveries:", df["batting_pressure"].sum())
print("Bowling pressure deliveries:", df["bowling_pressure"].sum())
print("High RRR pressure:", df["high_rrr_pressure"].sum())
print("Resource collapse:", df["resource_collapse"].sum())
print("Close finishes:", df["close_finish"].sum())
print("Death bowling pressure:", df["death_rrr_pressure"].sum())
print("Low-target powerplay pressure:", df["low_target_powerplay"].sum())
print("Output:", output)
print("Partnership momentum pressure:", df["partnership_momentum_pressure"].sum())