# If we use streamlit
import os
import io
import json
import shutil
import tempfile
from contextlib import redirect_stdout, redirect_stderr
from typing import Dict, Tuple, Any

import pandas as pd
import streamlit as st

# IMPORTANT: We keep backend solver code as-is and import it from the existing module
# Ensure train_dispatch_optimizer1.py is present in the same directory.
from train_dispatch_optimizer1 import TrainDispatchOptimizer


# ---------------------------
# UI Helpers
# ---------------------------

def write_uploaded_file(uploaded_file, dest_path: str) -> None:
    """Write an uploaded file (Streamlit UploadedFile) to the given destination path."""
    with open(dest_path, "wb") as f:
        f.write(uploaded_file.getbuffer())


def build_results_tables(solution: Dict[str, Any]) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Build pandas DataFrames for each train:
    - Timetable (Station, Planned, Actual Arrival, Actual Departure, Delay)
    - Block Assignments (Block, Entry, Exit)
    """
    result = {}
    for train_id, sol in solution.items():
        # Timetable
        rows = []
        for i, station in enumerate(sol["route"]):
            rows.append(
                {
                    "Station": station,
                    "Planned (min)": sol["planned"][i],
                    "Actual Arrival (min)": sol["actual_arrival"][station],
                    "Actual Departure (min)": sol["actual_departure"][station],
                    "Delay (min)": sol["delays"][station],
                }
            )
        timetable_df = pd.DataFrame(rows)

        # Block assignments
        block_rows = []
        for block_id, (entry, exit_) in sorted(sol["blocks"].items(), key=lambda x: (x[1][0], x[0])):
            block_rows.append({"Block": block_id, "Entry (min)": entry, "Exit (min)": exit_})
        blocks_df = pd.DataFrame(block_rows)

        result[train_id] = (timetable_df, blocks_df)
    return result


def inject_css() -> None:
    """Inject minimal CSS to give a clean, railway-control-panel feel."""
    st.markdown(
        """
        <style>
        /* Global font and background */
        body, .stApp {
            background-color: #0e1117;
            color: #e8e8e8;
        }

        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background-color: #141a22;
            border-right: 1px solid #2b3137;
        }

        /* Headers */
        h1, h2, h3 {
            color: #E0E6ED;
        }

        /* Console-like panel */
        .console-panel {
            background-color: #0b0f14;
            border: 1px solid #2b3137;
            border-radius: 6px;
            padding: 12px;
            font-family: "SFMono-Regular", Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
            color: #c9d1d9;
            white-space: pre-wrap;
        }

        /* Tables */
        .dataframe td, .dataframe th {
            border-color: #2b3137 !important;
        }

        /* Card-like containers */
        .panel {
            background-color: #121821;
            padding: 16px;
            border-radius: 8px;
            border: 1px solid #2b3137;
        }

        /* Buttons */
        .stButton>button {
            background-color: #1f6feb;
            color: white;
            border: none;
            border-radius: 6px;
        }
        .stButton>button:hover {
            background-color: #388bfd;
        }

        /* Subtle separators */
        hr {
            border: 0;
            border-top: 1px solid #2b3137;
            margin: 1rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def run_optimization(trains_file_path: str, maintenance_file_path: str) -> Tuple[bool, str, Dict[str, Any], float, bytes]:
    """
    Run the optimizer end-to-end, capturing logs and returning:
    - success (bool)
    - logs (str)
    - solution (dict)
    - objective_value (float)
    - schedule_image_bytes (bytes)
    """
    optimizer = TrainDispatchOptimizer()

    log_stream = io.StringIO()
    img_bytes = b""

    with redirect_stdout(log_stream), redirect_stderr(log_stream):
        print("=" * 80)
        print("TRAIN DISPATCH OPTIMIZATION - STREAMLIT RUN")
        print("=" * 80)

        # Load inputs
        optimizer.load_data(trains_file_path)
        optimizer.load_maintenance_data(maintenance_file_path)

        # Build model
        optimizer.build_model()

        # Optional: add additional solver logging without changing backend code
        optimizer.solver.parameters.log_search_progress = True

        # Solve
        success = optimizer.solve_model()

        if success:
            # Visualization will save 'schedule.png' in CWD
            optimizer.visualize_schedule()
        else:
            print("No feasible solution. Try adjusting inputs or relaxing constraints.")

    # After capturing logs
    logs = log_stream.getvalue()

    # Prepare return objects
    solution = optimizer.solution if success else {}
    objective_value = optimizer.solver.ObjectiveValue() if success else float("nan")

    # Read the generated schedule.png if present
    schedule_path_candidates = ["schedule.png"]
    for candidate in schedule_path_candidates:
        if os.path.exists(candidate):
            with open(candidate, "rb") as f:
                img_bytes = f.read()
            break

    return success, logs, solution, objective_value, img_bytes


# ---------------------------
# Streamlit App
# ---------------------------

st.set_page_config(page_title="Train Dispatch Optimizer", layout="wide")
inject_css()

st.title("🚆 Train Dispatch Optimization UI")
st.caption("Powered by Google OR-Tools CP-SAT. Upload inputs, run optimization, inspect logs, results, and visualization.")

# Sidebar: File uploads
with st.sidebar:
    st.header("Inputs")
    st.markdown("Upload your JSON files. Leave empty to use repository defaults.")

    trains_upload = st.file_uploader(
        "trains.json",
        type=["json"],
        accept_multiple_files=False,
        help="Upload train schedule JSON",
        key="trains_uploader",
    )
    maintenance_upload = st.file_uploader(
        "maintenance.json",
        type=["json"],
        accept_multiple_files=False,
        help="Upload maintenance windows JSON",
        key="maintenance_uploader",
    )

    st.markdown("---")
    run_clicked = st.button("Run Optimization", use_container_width=True)

# Main Panel Sections
logs_container = st.container()
results_container = st.container()
viz_container = st.container()

# Session state keys to persist results across reruns
if "last_logs" not in st.session_state:
    st.session_state.last_logs = ""
if "last_solution" not in st.session_state:
    st.session_state.last_solution = {}
if "last_objective" not in st.session_state:
    st.session_state.last_objective = None
if "last_image" not in st.session_state:
    st.session_state.last_image = b""
if "last_success" not in st.session_state:
    st.session_state.last_success = False

if run_clicked:
    # Create a temporary working directory for this run
    workdir = tempfile.mkdtemp(prefix="tdo_run_")
    trains_path = os.path.join(workdir, "trains.json")
    maintenance_path = os.path.join(workdir, "maintenance.json")

    # If uploads provided, write them; otherwise, fallback to repo defaults
    if trains_upload is not None:
        write_uploaded_file(trains_upload, trains_path)
    else:
        # Fallback to local default trains.json (must exist in repository)
        shutil.copyfile("trains.json", trains_path)

    if maintenance_upload is not None:
        write_uploaded_file(maintenance_upload, maintenance_path)
    else:
        # Fallback to local default maintenance.json (must exist in repository)
        shutil.copyfile("maintenance.json", maintenance_path)

    with st.spinner("Running optimization..."):
        success, logs, solution, objective_value, img_bytes = run_optimization(
            trains_file_path=trains_path, maintenance_file_path=maintenance_path
        )

    # Persist results in session
    st.session_state.last_logs = logs
    st.session_state.last_solution = solution
    st.session_state.last_objective = objective_value
    st.session_state.last_image = img_bytes
    st.session_state.last_success = success

# Section 1: Optimization Logs
with logs_container:
    st.subheader("Optimization Logs")
    if st.session_state.last_logs:
        st.markdown(f"<div class='console-panel'>{st.session_state.last_logs}</div>", unsafe_allow_html=True)
    else:
        st.info("No logs yet. Upload inputs and click 'Run Optimization' in the sidebar.")

# Section 2: Train Timetable Results
with results_container:
    st.subheader("Train Timetable Results")
    if st.session_state.last_success and st.session_state.last_solution:
        if st.session_state.last_objective is not None:
            st.markdown(f"Objective Value: `{st.session_state.last_objective}`")

        tables = build_results_tables(st.session_state.last_solution)
        # Display tables per train
        for train_id in sorted(tables.keys()):
            timetable_df, blocks_df = tables[train_id]
            st.markdown(f"### Train {train_id}")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Timetable (Planned vs Actual)**")
                st.dataframe(timetable_df, use_container_width=True)
            with col2:
                st.markdown("**Block Assignments**")
                st.dataframe(blocks_df, use_container_width=True)
            st.markdown("---")
    else:
        st.info("Results will appear here after a successful optimization run.")

# Section 3: Visualization
with viz_container:
    st.subheader("Visualization")
    if st.session_state.last_success and st.session_state.last_image:
        st.image(st.session_state.last_image, caption="Gantt Chart - Schedule", use_container_width=True)
    elif st.session_state.last_success and not st.session_state.last_image:
        st.warning("Visualization image not found. Ensure the backend saved 'schedule.png'.")
    else:
        st.info("The Gantt chart will be displayed here after a successful run.")


# Footer note
st.markdown("---")
st.caption("Tip: You can re-run optimization with different inputs at any time without restarting the app.")
