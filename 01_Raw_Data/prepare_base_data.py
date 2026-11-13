
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "01_Raw_Data" / "Data Pack"
OUTPUT_DIR = BASE_DIR / "02_Cleaned_Data"

balls = pd.read_csv(DATA_DIR / "ball_by_ball.csv")
matches = pd.read_csv(DATA_DIR / "matches.csv")
competitions = pd.read_csv(DATA_DIR / "competitions.csv")

# Include IPL and T20I only
valid_competitions = competitions[
    competitions["match_format"].isin(["T20", "T20I"])
   & competitions["season"].astype(str).str.match(r"^202[2-6]($|/)")
    & ~competitions["comp_id"].isin([2670, 3289])
    & (
        (competitions["match_format"] == "T20I")
        | competitions["comp_name"].str.match(r"^IPL 202[2-6]$")
    )
]

# Match IDs belonging to selected competitions
valid_match_ids = matches[
    matches["comp_id"].isin(valid_competitions["comp_id"])
]["match_id"]

filtered = balls[balls["match_id"].isin(valid_match_ids)].copy()

# Exclude Super Overs (innings 3 and 4)
filtered = filtered[filtered["innings"].isin([1, 2])].copy()

OUTPUT_DIR.mkdir(exist_ok=True)

output_file = OUTPUT_DIR / "01_filtered_base_deliveries.csv"
filtered.to_csv(output_file, index=False)

print("Selected competitions:", len(valid_competitions))
print("Selected matches:", len(valid_match_ids))
print("Filtered deliveries:", len(filtered))
print("Output saved:", output_file)
