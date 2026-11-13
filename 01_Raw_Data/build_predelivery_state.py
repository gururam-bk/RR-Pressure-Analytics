
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "01_Raw_Data" / "Data Pack"
OUTPUT_DIR = BASE_DIR / "02_Cleaned_Data"

INPUT_FILE = OUTPUT_DIR / "01_filtered_base_deliveries.csv"
MATCH_FILE = DATA_DIR / "matches.csv"
OUTPUT_FILE = OUTPUT_DIR / "02_predelivery_match_state.csv"

df = pd.read_csv(INPUT_FILE)
matches = pd.read_csv(MATCH_FILE)

# Match-level quota and target information
match_info = matches[
    [
        "match_id",
        "first_inn_revised_overs",
        "second_inn_revised_overs",
        "revised_target",
        "result"
    ]
].copy()

df = df.merge(match_info, on="match_id", how="left")

# Revised overs values in this dataset represent BALLS
df["innings_quota"] = 120

first_mask = df["innings"] == 1
second_mask = df["innings"] == 2

df.loc[first_mask, "innings_quota"] = (
    df.loc[first_mask, "first_inn_revised_overs"]
    .fillna(120)
)

df.loc[second_mask, "innings_quota"] = (
    df.loc[second_mask, "second_inn_revised_overs"]
    .fillna(120)
)

df["innings_quota"] = df["innings_quota"].astype(int)

# Sort deliveries
df = df.sort_values(
    ["match_id", "innings", "innings_ball_num"]
).copy()

extras = df["extras_type"].fillna("")

# Provisional extras classification
df["is_wide"] = extras.eq("wide")
df["is_no_ball"] = extras.isin(["Overstep", "other"])

df["is_legal_ball"] = ~(
    df["is_wide"] | df["is_no_ball"]
)

df["is_batter_ball"] = ~df["is_wide"]

# Team runs
df["team_runs"] = (
    df["runs"]
    + df["extras1_runs"]
    + df["extras_2_runs"]
)

# Wickets lost by batting team
df["wicket_fall"] = (
    df["is_wicket"].fillna(False)
    & df["dismissal_type"].ne("Retired hurt")
).astype(int)

# Calculate state BEFORE each delivery
groups = df.groupby(["match_id", "innings"], sort=False)

df["score_before"] = groups["team_runs"].transform(
    lambda s: s.cumsum().shift(fill_value=0)
)

df["wickets_lost_before"] = groups["wicket_fall"].transform(
    lambda s: s.cumsum().shift(fill_value=0)
)

df["legal_balls_before"] = groups["is_legal_ball"].transform(
    lambda s: s.cumsum().shift(fill_value=0)
)

df["batter_balls_before"] = groups["is_batter_ball"].transform(
    lambda s: s.cumsum().shift(fill_value=0)
)

# Resources remaining
df["legal_balls_remaining"] = (
    df["innings_quota"] - df["legal_balls_before"]
).clip(lower=0)

df["wickets_in_hand"] = (
    10 - df["wickets_lost_before"]
).clip(lower=0)

# First innings totals
first_totals = (
    df[first_mask]
    .groupby("match_id")["team_runs"]
    .sum()
)

# Normal target = first innings total + 1
normal_targets = first_totals + 1

# Revised target takes priority when available
revised_targets = (
    matches.set_index("match_id")["revised_target"]
)

df["target"] = pd.NA

df.loc[second_mask, "target"] = (
    df.loc[second_mask, "match_id"]
    .map(normal_targets)
)

df.loc[second_mask, "target"] = (
    df.loc[second_mask, "match_id"]
    .map(revised_targets)
    .combine_first(df.loc[second_mask, "target"])
)

# Do not invent a normal target for No Result matches
no_result_ids = matches.loc[
    matches["result"].fillna("").str.contains(
        "No Result", case=False
    ),
    "match_id"
]

invalid_target_mask = (
    second_mask
    & df["match_id"].isin(no_result_ids)
    & df["match_id"].map(revised_targets).isna()
)

df.loc[invalid_target_mask, "target"] = pd.NA

# Second innings pressure inputs
df["runs_remaining"] = pd.NA

df.loc[second_mask, "runs_remaining"] = (
    df.loc[second_mask, "target"]
    - df.loc[second_mask, "score_before"]
)

df["required_run_rate"] = pd.NA

valid_rrr = (
    second_mask
    & df["target"].notna()
    & (df["legal_balls_remaining"] > 0)
)

df.loc[valid_rrr, "required_run_rate"] = (
    df.loc[valid_rrr, "runs_remaining"]
    / (df.loc[valid_rrr, "legal_balls_remaining"] / 6)
)

# Mark whether second-innings target is usable
df["target_valid"] = (
    second_mask
    & df["target"].notna()
    & ~df["match_id"].isin(no_result_ids)
    & ~df["match_id"].isin([48610])
)

# Save
OUTPUT_DIR.mkdir(exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)

print("Updated pre-delivery state created.")
print("Total deliveries:", len(df))
print("Missing score states:", df["score_before"].isna().sum())
print("Missing wicket states:", df["wickets_lost_before"].isna().sum())
print("Negative runs remaining:", (df["runs_remaining"] < 0).sum())
print("Invalid second-innings targets:", 
      df.loc[second_mask, "target"].isna().sum())
print("Output:", OUTPUT_FILE)
