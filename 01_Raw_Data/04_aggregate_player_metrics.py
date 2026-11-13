
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
INPUT = BASE / "02_Cleaned_Data" / "03_pressure_flagged_deliveries.csv"
OUTPUT = BASE / "02_Cleaned_Data" / "04_player_raw_metrics.csv"

df = pd.read_csv(INPUT, low_memory=False)

keys = ["player_id", "player_name", "match_format", "season", "phase"]

# ---------------- BATTER METRICS ----------------

bat = df.copy()

bat["player_id"] = bat["striker_id"]
bat["player_name"] = bat["striker_name"]
bat["pressure"] = bat["batting_pressure"].fillna(False)

bat["counted_balls"] = bat["is_batter_ball"].astype(int)
bat["scored_runs"] = bat["runs"]

# Dismissals of the striker
bat["dismissal"] = (
    bat["wicket_fall"].fillna(0).astype(int)
    & bat["dismissed_player_id"].eq(bat["striker_id"])
).astype(int)

# Create player ID-to-name mapping using both batting positions
player_names = pd.concat([
    df[["striker_id", "striker_name"]].rename(
        columns={
            "striker_id": "player_id",
            "striker_name": "player_name"
        }
    ),
    df[["non_striker_id", "non_striker_name"]].rename(
        columns={
            "non_striker_id": "player_id",
            "non_striker_name": "player_name"
        }
    )
]).dropna(subset=["player_id"]).drop_duplicates("player_id")

name_map = player_names.set_index("player_id")["player_name"]

# Identify dismissals of players other than the striker
extra_dismissals = bat[
    (bat["wicket_fall"] == 1)
    & bat["dismissed_player_id"].notna()
    & (bat["dismissed_player_id"] != bat["striker_id"])
].copy()

# Attribute dismissal to the actual dismissed player
extra_dismissals["player_id"] = extra_dismissals["dismissed_player_id"]
extra_dismissals["player_name"] = extra_dismissals["player_id"].map(name_map)

# Do not assign extra runs or balls to the dismissed non-striker
extra_dismissals["scored_runs"] = 0
extra_dismissals["counted_balls"] = 0
extra_dismissals["dismissal"] = 1

# Add only the additional dismissal records
bat = pd.concat([bat, extra_dismissals], ignore_index=True)

# Aggregate batter scoring and dismissals
batting = bat.groupby(keys).apply(
    lambda g: pd.Series({
        "pressure_runs": g.loc[g.pressure, "scored_runs"].sum(),
        "pressure_balls": g.loc[g.pressure, "counted_balls"].sum(),
        "pressure_outs": g.loc[g.pressure, "dismissal"].sum(),
        "normal_runs": g.loc[~g.pressure, "scored_runs"].sum(),
        "normal_balls": g.loc[~g.pressure, "counted_balls"].sum(),
        "normal_outs": g.loc[~g.pressure, "dismissal"].sum()
    }),
    include_groups=False
).reset_index()

batting["role"] = "Batter"

# ---------------- BOWLER METRICS ----------------

bowl = df.copy()

bowl["player_id"] = bowl["bowler_id"]
bowl["player_name"] = bowl["bowler_name"]
bowl["pressure"] = bowl["bowling_pressure"].fillna(False)

# Chargeable runs: batter runs + recorded wides/no-ball extras
extras = bowl["extras_type"].fillna("")

bowl["bowler_runs"] = (
    bowl["runs"]
    + bowl["extras1_runs"].where(
        extras.isin(["wide", "Overstep", "other"]), 0
    )
)

bowl["legal_balls"] = bowl["is_legal_ball"].astype(int)

# Correct dismissal labels from the actual dataset
bowler_wicket_types = [
    "Bowled",
    "Caught",
    "LBW",
    "Stump",
    "Hit Wicket"
]

bowl["bowler_wicket"] = (
    bowl["is_wicket"].fillna(False)
    & bowl["dismissal_type"].isin(bowler_wicket_types)
).astype(int)

# Aggregate bowler performance
bowling = bowl.groupby(keys).apply(
    lambda g: pd.Series({
        "pressure_runs": g.loc[g.pressure, "bowler_runs"].sum(),
        "pressure_balls": g.loc[g.pressure, "legal_balls"].sum(),
        "pressure_outs": g.loc[g.pressure, "bowler_wicket"].sum(),
        "normal_runs": g.loc[~g.pressure, "bowler_runs"].sum(),
        "normal_balls": g.loc[~g.pressure, "legal_balls"].sum(),
        "normal_outs": g.loc[~g.pressure, "bowler_wicket"].sum()
    }),
    include_groups=False
).reset_index()

bowling["role"] = "Bowler"

# ---------------- COMBINE ----------------

result = pd.concat([batting, bowling], ignore_index=True)

result.to_csv(OUTPUT, index=False)

print("Player aggregation completed.")
print("Batter records:", len(batting))
print("Bowler records:", len(bowling))
print("Total player records:", len(result))
print("Unique batters:", batting.player_id.nunique())
print("Unique bowlers:", bowling.player_id.nunique())

print("\nBatter dismissal totals:")
print("Total batter dismissals:", batting["pressure_outs"].sum() + batting["normal_outs"].sum())

print("\nBowler wicket total:")
print("Total bowler wickets:", bowling["pressure_outs"].sum() + bowling["normal_outs"].sum())

print("\nOutput:", OUTPUT)
