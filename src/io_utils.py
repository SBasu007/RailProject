# Purpose: Single place for reading/writing CSVs and constants; avoids duplicated parsing logic.

# Core contents:

# Loaders: read ntes_train_positions.csv, tms_track_status.csv, fois_freight_info.csv, datalogger_events.csv, mvis_asset_health.csv, weather.csv with strict column checks and timestamp parsing.

# Writers: save timetables, features, metrics, and suggestions in consistent formats.

# Constants: load constants.json, expose typed accessors (headway_sec, station_ids, sections, priorities).

# Utilities: time-zone helpers, ID normalization, safe merges with keys and suffix handling.

# Outputs: clean DataFrames ready for features/simulator and dicts for constants.