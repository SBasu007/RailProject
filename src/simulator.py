# Purpose: Build the “digital twin” for a short horizon; detect conflicts, enforce hard constraints, step schedules, and compute baseline/after metrics.

# Core contents:

# Data loaders for schedule/events into in-memory structures (or accept DataFrames passed from io_utils).

# Conflict detection: section and platform exclusivity with headway 
# H
# H; functions like detect_conflicts(events, headway_sec) → list of conflicts.

# Propagation logic: simple rules for how delays shift downstream (e.g., if a section is occupied, push following train’s entry time).

# Metrics: total/average/max delay, number of conflicts, compute per-train delay vectors, and produce “timeline” rows for Gantt.

# Orchestrators: run_baseline(schedule, rules) → results; apply_offsets(schedule, offsets) → new_schedule + results.

# Outputs: metrics dict, conflict list, and two timetables (before.csv, after.csv) that the UI or CLI can render.
