# RR Pressure Performance Analytics

## SupeRR Selector Season 2 — Rajasthan Royals

A professional cricket analytics project focused on analysing **player performance under pressure** using historical IPL and T20 International data from 2022 onwards, as part of the Rajasthan Royals SupeRR Selector competition.

---

## Project Structure

| Folder | Purpose |
|--------|---------|
| `01_Raw_Data/` | Unprocessed source datasets (ball-by-ball, match-level, player metadata). Data files are stored here exactly as received — **never modify originals**. |
| `02_Cleaned_Data/` | Cleaned, validated, and transformed datasets ready for analysis. Includes de-duplicated records, standardised column names, and type-corrected fields. |
| `03_Exploratory_Analysis/` | Jupyter notebooks and scripts for initial data exploration, summary statistics, distribution plots, and correlation checks. |
| `04_Pressure_Metrics/` | Derived pressure indices, situational difficulty scores, and custom metrics that quantify match context (e.g., required run-rate gaps, wickets-in-hand, overs remaining). |
| `05_Player_Rankings/` | Aggregated player rankings and composite scores based on pressure-adjusted performance metrics. |
| `06_Visualizations/` | Publication-ready charts, dashboards, and visual assets for the final submission and methodology report. |
| `07_Methodology_Report/` | Written documentation of the analytical approach, metric definitions, assumptions, limitations, and statistical methods used. |
| `08_Final_Submission/` | Polished deliverables formatted per competition requirements (final report, data files, and supporting materials). |
| `09_Python_Scripts/` | Reusable Python modules, utility functions, and pipeline scripts that power the analysis end-to-end. |

---

## Key Principles

- **Reproducibility** — Every transformation is scripted; no manual edits to data files.
- **Data Integrity** — Raw data is never overwritten; all changes are saved to `02_Cleaned_Data/`.
- **Version Control** — Track changes with Git; commit early and often.
- **Documentation** — Every metric and decision is documented in `07_Methodology_Report/`.

---

## Data Scope

- **Formats:** IPL (Indian Premier League) and T20 International matches.
- **Period:** 2022 season onwards.
- **Granularity:** Ball-by-ball and match-level records.

---

## Status

> **Workspace initialised.** Awaiting data ingestion and competition ZIP extraction.

---

*Project created: 2026-11-11*
