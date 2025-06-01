# bo_optimize.py

import os
import json
import subprocess
import shutil
import time

import numpy as np
import pandas as pd
from skopt import gp_minimize
from skopt.space import Integer, Real
from skopt.utils import use_named_args

# ─────────────────────────────────────────────────────────────────────────────
# FIXED FIELDS: These fields remain constant across all BO trials. Adjust
# them to match your baseline simulation requirements.
# ─────────────────────────────────────────────────────────────────────────────
FIXED_PARAMS = {
    "f1_ghz": 3.0,
    "f2_ghz": 9.0,
    "eps_r_substrate": 4.4,
    "h_substrate_um": 200.0,
    "metal_thickness_um": 2.0,
    "substrate_tand": 0.02,

    # ───── FAST EM SETTINGS ─────
    "sim_box_padding_factor": 1.0,  # minimal padding
    "mesh_resolution_factor": 4,  # coarser mesh (much faster)
    "fsteps_output": 41,  # fewer frequency points
    "EndCriteria": 1e-5,
    "NrTS": 500000,  # fewer timesteps (speeds up time‐domain)
    "numThreads": 4,  # use fewer threads if your PC has limited cores
    # ───────────────────────────────

    "add_common_ground_via": True,
    "common_via_radius_um": 30.0
}
# ─────────────────────────────────────────────────────────────────────────────
space = [
    Integer(2, 10, name="N1"),  # primary turns: 2–10
    Integer(2, 10, name="N2"),  # secondary turns: 2–10
    Real(10.0, 40.0, name="w_um"),  # trace width (µm): 10–40
    Real(10.0, 40.0, name="s_um"),  # turn spacing (µm): 10–40
]


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY: Given a path to the JSON config (after simulate_transformer.py
# runs), return the path to the S‐parameter CSV that was produced.
# simulate_transformer.py saves a CSV under outputs/<config_name>/<config_name>_sparameters.csv
# ─────────────────────────────────────────────────────────────────────────────
def find_sparam_csv(config_path: str) -> str:
    """
    Input:
        config_path = 'configs/bo_trial_<i>.json'
    Returns:
        path_to_csv = 'outputs/bo_trial_<i>/bo_trial_<i>_sparameters.csv'
    """
    cfg_name = os.path.splitext(os.path.basename(config_path))[0]
    out_dir = os.path.join("outputs", cfg_name)
    candidate = os.path.join(out_dir, f"{cfg_name}_sparameters.csv")
    if os.path.exists(candidate):
        return candidate
    else:
        raise FileNotFoundError(f"Could not find S-parameters CSV for {cfg_name} at {candidate}")


# ─────────────────────────────────────────────────────────────────────────────
# OBJECTIVE FUNCTION: runs one BO trial
# ─────────────────────────────────────────────────────────────────────────────
@use_named_args(space)
def bo_objective(**params):
    # step 1) merge with fixed fields
    combined = { **FIXED_PARAMS, **params }

    # step 2) convert any numpy types to native Python
    for k, v in combined.items():
        if isinstance(v, (np.integer, np.int64, np.int32)):
            combined[k] = int(v)
        elif isinstance(v, (np.floating, np.float64, np.float32)):
            combined[k] = float(v)

    # now combined is safe to JSON‐serialize:
    cfg_name = f"bo_trial_{int(time.time())}"
    config_path = os.path.join("configs", f"{cfg_name}.json")
    with open(config_path, "w") as fp:
        json.dump(combined, fp, indent=2)
    # 3) Remove any leftover output folder if it exists (clean slate)
    out_dir = os.path.join("outputs", cfg_name)
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)

    # 4) Call simulate_transformer.py via subprocess
    #    This will generate the GDS, run OpenEMS, save CSV in outputs/bo_trial_<timestamp>/...
    cmd = ["python", "simulate_transformer.py", config_path]
    print(f"\n[BO] Running trial {cfg_name} with params: {params}")
    ret = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if ret.returncode != 0:
        print("---- STDOUT ----")
        print(ret.stdout)
        print("---- STDERR ----")
        print(ret.stderr)
        raise RuntimeError(f"simulate_transformer.py failed for {cfg_name}")

    # 5) Locate and read the generated CSV
    try:
        csv_path = find_sparam_csv(config_path)
    except FileNotFoundError as e:
        print("ERROR: " + str(e))
        # Return a large penalty so BO will avoid this region
        return +1000.0

    df = pd.read_csv(csv_path, comment="#")
    # CSV format is: [ 'Frequency (Hz)', '|S11|', 'Phase(S11)', '|S21|', 'Phase(S21)', ... ]
    # So column index 3 is '|S21|', in linear magnitude. We want dB.
    if "|S21|" not in df.columns:
        print("ERROR: '|S21|' column not found in CSV. Full columns:", df.columns.tolist())
        return +1000.0

    # Compute S21_dB
    s21_lin = df["|S21|"].to_numpy()
    # Avoid log(0) by adding a tiny floor
    s21_db = 20.0 * np.log10(np.maximum(s21_lin, 1e-12))

    # We choose the single *maximum* coupling (peak |S21|_dB) over the band:
    best_s21_db = np.max(s21_db)

    # We want to MAXIMIZE best_s21_db, but skopt does minimization, so we return –best_s21_db:
    objective_value = -1.0 * best_s21_db
    print(f"[BO] ==> best |S21|_dB = {best_s21_db:.2f} dB   → returning {objective_value:.2f}\n")
    return objective_value


# ─────────────────────────────────────────────────────────────────────────────
# MAIN: Run gp_minimize with the objective above
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1) Number of BO calls (you can reduce to 10–15 for a quick test, but expect low accuracy)
    N_CALLS = 20

    # 2) Run BO
    res = gp_minimize(
        func=bo_objective,
        dimensions=space,
        n_calls=N_CALLS,
        n_random_starts=5,  # first 5 are random, then GP-based sampling
        random_state=42,
        verbose=True
    )

    # 3) Print results
    print("\n=== BO COMPLETE ===")
    print(f"Best parameters found (min. objective → max |S21|):")
    best_params = {dim.name: val for dim, val in zip(space, res.x)}
    print(best_params)
    print(f"Corresponding (max |S21|_dB) = {-res.fun:.2f} dB")

    # 4) Save the best JSON out for posterity
    best_name = "bo_best_config"
    best_config = {**FIXED_PARAMS, **best_params}
    os.makedirs("configs", exist_ok=True)
    best_path = os.path.join("configs", f"{best_name}.json")
    with open(best_path, "w") as f:
        json.dump(best_config, f, indent=2)
    print(f"Best config written to: {best_path}")
