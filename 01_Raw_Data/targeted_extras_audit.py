
import pandas as pd

CSV_PATH = (
    r"C:\Users\GURU\Desktop\RR_Pressure_Analytics"
    r"\01_Raw_Data\Data Pack\ball_by_ball.csv"
)

df = pd.read_csv(CSV_PATH)

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 2000)

columns = [
    "match_id", "innings", "overid",
    "over_ball_num", "innings_ball_num",
    "striker_id", "striker_name",
    "bowler_id", "bowler_name",
    "runs", "is_wicket", "dismissal_type",
    "dismissed_player_id",
    "extras_type", "extras1_runs", "extras_2_runs"
]

# OTHER records with non-zero extras_2_runs
other = df[
    (df["extras_type"] == "other") &
    (df["extras_2_runs"] != 0)
]

print("\n========== OTHER: NON-ZERO EXTRAS_2 ==========")
print("Count:", len(other))
print(other[columns].to_string(index=False))

# OVERSTEP records with non-zero extras_2_runs
overstep = df[
    (df["extras_type"] == "Overstep") &
    (df["extras_2_runs"] != 0)
]

print("\n========== OVERSTEP: NON-ZERO EXTRAS_2 ==========")
print("Count:", len(overstep))
print(overstep[columns].to_string(index=False))

# Complete Penalty record
penalty = df[df["extras_type"] == "Penalty"]

print("\n========== COMPLETE PENALTY RECORD ==========")
print(penalty.to_string(index=False))

# Run and extras relationship
print("\n========== RUNS + EXTRAS SUMMARY ==========")

for category in ["other", "Overstep", "Penalty"]:
    subset = df[df["extras_type"] == category]

    print(f"\n{category}")
    print("Records:", len(subset))
    print("runs total:", subset["runs"].sum())
    print("extras1 total:", subset["extras1_runs"].sum())
    print("extras_2 total:", subset["extras_2_runs"].sum())

    print("\nCombinations:")
    print(
        subset.groupby(
            ["runs", "extras1_runs", "extras_2_runs"]
        ).size().to_string()
    )

# Null extras type with non-zero extras
null_extras = df[
    df["extras_type"].isna() &
    (
        (df["extras1_runs"] != 0) |
        (df["extras_2_runs"] != 0)
    )
]

print("\n========== NULL EXTRAS TYPE ANOMALIES ==========")
print("Count:", len(null_extras))
print(null_extras[columns].to_string(index=False))

print("\n========== AUDIT COMPLETE ==========")
