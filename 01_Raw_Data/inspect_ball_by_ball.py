import pandas as pd
import numpy as np

def run_ball_by_ball_inspection():
    file_path = r"C:\Users\GURU\Desktop\RR_Pressure_Analytics\01_Raw_Data\Data Pack\ball_by_ball.csv"
    print(f"Loading {file_path} in strict read-only mode...")
    
    try:
        df = pd.read_csv(file_path, low_memory=False)
    except FileNotFoundError:
        print(f"ERROR: {file_path} not found in current directory.")
        return

    print("\n" + "="*80)
    print("SECTION 1: COMPLETE COLUMNS & DATA TYPES")
    print("="*80)
    print(df.dtypes.to_string())

    print("\n" + "="*80)
    print("SECTION 2: EXTRAS_TYPE FREQUENCIES (INCLUDING NULLS)")
    print("="*80)
    if 'extras_type' in df.columns:
        print(df['extras_type'].value_counts(dropna=False).to_string())
    else:
        print("CRITICAL WARNING: 'extras_type' column not found!")
        candidate_cols = [c for c in df.columns if 'extra' in c.lower() or 'type' in c.lower()]
        print(f"Candidate alternative columns: {candidate_cols}")

    print("\n" + "="*80)
    print("SECTION 3: DEEP DIVE ON EXTRAS & RUN COLUMNS")
    print("="*80)
    keywords = ['run', 'extra', 'wide', 'noball', 'overstep', 'bye', 'legbye', 'penalty', 'other', 'type']
    extras_cols = [c for c in df.columns if any(k in c.lower() for k in keywords)]
    
    print(f"Identified {len(extras_cols)} relevant columns: {extras_cols}\n")
    for col in extras_cols:
        null_cnt = df[col].isna().sum()
        dtype_str = str(df[col].dtype)
        unique_vals = df[col].dropna().unique()
        print(f"Column: {col:<24} | Type: {dtype_str:<8} | Nulls: {null_cnt:<7} | Sample Unique ({len(unique_vals)}): {unique_vals[:8]}")

    print("\n" + "="*80)
    print("SECTION 4: REPRESENTATIVE RECORDS FOR EVERY EXTRAS CLASSIFICATION")
    print("="*80)
    if 'extras_type' in df.columns:
        all_types = df['extras_type'].unique()
        id_cols = [c for c in ['match_id', 'innings', 'overid', 'over_ball_num', 'innings_ball_num'] if c in df.columns]
        display_cols = list(dict.fromkeys(id_cols + ['extras_type'] + extras_cols))
        
        for et in all_types:
            label = "NaN / Null" if pd.isna(et) else repr(et)
            print(f"\n--- SAMPLE RECORDS: extras_type = {label} ---")
            if pd.isna(et):
                sample_rows = df[df['extras_type'].isna()].head(3)
            else:
                sample_rows = df[df['extras_type'] == et].head(3)
            print(sample_rows[display_cols].to_string(index=False))

    print("\n" + "="*80)
    print("SECTION 5: SPECIFIC AUDIT OF 264 'other' RECORDS")
    print("="*80)
    if 'extras_type' in df.columns:
        other_mask = df['extras_type'].astype(str).str.strip().str.lower() == 'other'
        other_df = df[other_mask]
        print(f"Total records matching extras_type = 'other': {len(other_df)}")
        
        if len(other_df) > 0:
            print("\nRun / Extras Column Value Counts for 'other' Records:")
            for rc in extras_cols:
                print(f"\nDistribution for column '{rc}':")
                print(other_df[rc].value_counts(dropna=False).to_string())

            print("\nFirst 10 Detailed Records for extras_type = 'other':")
            display_cols = list(dict.fromkeys(id_cols + ['extras_type'] + extras_cols))
            print(other_df[display_cols].head(10).to_string(index=False))

    print("\n" + "="*80)
    print("SECTION 6: SPECIFIC AUDIT OF 'Overstep' RECORDS & OVERLAP WITH 'other'")
    print("="*80)
    if 'extras_type' in df.columns:
        overstep_mask = df['extras_type'].astype(str).str.strip() == 'Overstep'
        overstep_df = df[overstep_mask]
        print(f"Total records matching extras_type = 'Overstep': {len(overstep_df)}")
        
        if len(overstep_df) > 0:
            print("\nRun / Extras Column Value Counts for 'Overstep' Records:")
            for rc in extras_cols:
                print(f"\nDistribution for column '{rc}':")
                print(overstep_df[rc].value_counts(dropna=False).to_string())

            print("\nFirst 5 Detailed Records for extras_type = 'Overstep':")
            display_cols = list(dict.fromkeys(id_cols + ['extras_type'] + extras_cols))
            print(overstep_df[display_cols].head(5).to_string(index=False))

    print("\n" + "="*80)
    print("SECTION 7: RUN RELATIONSHIPS (runs vs. extras1_runs vs. extras_2_runs)")
    print("="*80)
    target_run_cols = [c for c in df.columns if any(k in c.lower() for k in ['runs', 'extras1', 'extras_2', 'extras_runs', 'batter_runs'])]
    print(f"Target run columns found: {target_run_cols}")
    if target_run_cols and 'extras_type' in df.columns:
        print("\nAggregated sums and means by extras_type:")
        print(df.groupby('extras_type', dropna=False)[target_run_cols].agg(['count', 'sum', 'mean']).to_string())

    print("\n" + "="*80)
    print("SECTION 8: CHRONOLOGICAL ORDERING & SEQUENCE AUDIT")
    print("="*80)
    seq_cols = [c for c in ['match_id', 'innings', 'innings_ball_num', 'overid', 'over_ball_num', 'extras_type'] if c in df.columns]
    print(f"Sequence columns identified: {seq_cols}")
    
    if 'match_id' in df.columns and 'innings' in df.columns:
        # Filter a sample match containing extras for sequence inspection
        sample_matches = df[df['extras_type'].notna()]['match_id'].unique()[:2]
        for m_id in sample_matches:
            print(f"\n--- Sequence Audit for Sample Match ID: {m_id} (Innings 1) ---")
            m_df = df[(df['match_id'] == m_id) & (df['innings'] == 1)]
            
            # Sort by present sequence columns
            sort_keys = [c for c in ['innings_ball_num', 'overid', 'over_ball_num'] if c in m_df.columns]
            if sort_keys:
                m_df = m_df.sort_values(by=sort_keys)
            
            print(m_df[seq_cols].head(20).to_string(index=False))
            
            # Check duplicate sequence values
            if 'innings_ball_num' in df.columns:
                dups = m_df[m_df.duplicated(subset=['innings_ball_num'], keep=False)]
                print(f"Duplicate innings_ball_num count: {len(dups)}")
                if len(dups) > 0:
                    print(dups[seq_cols].to_string(index=False))
                    
            if 'overid' in df.columns and 'over_ball_num' in df.columns:
                over_dups = m_df[m_df.duplicated(subset=['overid', 'over_ball_num'], keep=False)]
                print(f"Duplicate (overid, over_ball_num) count: {len(over_dups)}")
                if len(over_dups) > 0:
                    print(over_dups[seq_cols].to_string(index=False))

if __name__ == "__main__":
    run_ball_by_ball_inspection()