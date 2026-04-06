import numpy as np
import pandas as pd
import os
import random
import opensim as osim
import re

BASE_DIR = os.getcwd()

MODEL_PATH = os.path.join(BASE_DIR, "constant/Rajagopal2016.osim")
SETUP_PATH = os.path.join(BASE_DIR, "constant/setup_id.xml")
EXTERNAL_LOADS = os.path.join(BASE_DIR, "constant/external_loads.xml")

BASE_JK = os.path.join(BASE_DIR, "base_inputs/running_JK_base.mot")
BASE_GRF = os.path.join(BASE_DIR, "base_inputs/running_GRF_base.mot")

OUT_JK = os.path.join(BASE_DIR, "generated/jk")
OUT_GRF = os.path.join(BASE_DIR, "generated/grf")
OUT_STO = os.path.join(BASE_DIR, "generated/sto")

DATASET_DIR = os.path.join(BASE_DIR, "generated")

os.makedirs(OUT_JK, exist_ok=True)
os.makedirs(OUT_GRF, exist_ok=True)
os.makedirs(OUT_STO, exist_ok=True)

# ---- Safety Checks ----
for path in [MODEL_PATH, SETUP_PATH, EXTERNAL_LOADS, BASE_JK, BASE_GRF]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required file: {path}")

N_PER_SEVERITY = 200

SEVERITY_CONFIG = {
    0: {"valgus": (0, 0.5),  "grf_scale": 1.00},
    1: {"valgus": (3, 4),    "grf_scale": 1.10},
    2: {"valgus": (6, 7),    "grf_scale": 1.25},
    3: {"valgus": (10, 12),  "grf_scale": 1.50}
}

TARGET_FRAMES = 72  # enforce consistent time length

# -------------------------
# File Utilities
# -------------------------

def load_mot(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    header_end = None
    for i, line in enumerate(lines):
        if "endheader" in line.lower():
            header_end = i
            break

    header = lines[:header_end+1]
    data = pd.read_csv(filepath, sep="\t", skiprows=header_end+1)
    return header, data

def save_mot(filepath, header, df):
    with open(filepath, 'w') as f:
        f.writelines(header)
        df.to_csv(f, sep="\t", index=False)

def load_sto_trimmed(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    header_end = None
    for i, line in enumerate(lines):
        if "endheader" in line.lower():
            header_end = i
            break

    df = pd.read_csv(filepath, sep="\t", skiprows=header_end+1)

    # Drop time column
    if "time" in df.columns:
        df = df.drop(columns=["time"])

    # Enforce fixed length
    if len(df) > TARGET_FRAMES:
        df = df.iloc[:TARGET_FRAMES]
    elif len(df) < TARGET_FRAMES:
        raise ValueError(f"File {filepath} has fewer than {TARGET_FRAMES} frames")

    return df.values  # return numpy array

# -------------------------
# Perturbations
# -------------------------

def perturb_jk(df, severity):
    df_new = df.copy()
    n = len(df_new)
    t = np.linspace(0, 1, n)

    valgus_min, valgus_max = SEVERITY_CONFIG[severity]["valgus"]
    valgus_deg = random.uniform(valgus_min, valgus_max)
    valgus_rad = np.deg2rad(valgus_deg)

    center = random.uniform(0.2, 0.8)
    width = random.uniform(0.05, 0.15)
    window = np.exp(-((t - center) ** 2) / (2 * width ** 2))

    hip_scale = random.uniform(0.3, 0.6)
    pelvis_scale = random.uniform(0.2, 0.4)

    if "knee_adduction_r" in df_new.columns:
        df_new["knee_adduction_r"] += valgus_rad * window

    if "hip_adduction_r" in df_new.columns:
        df_new["hip_adduction_r"] += hip_scale * valgus_rad * window

    if "pelvis_list" in df_new.columns:
        df_new["pelvis_list"] += pelvis_scale * valgus_rad * window

    noise_amp = 0.001 * severity
    smooth_noise = np.convolve(np.random.randn(n), np.ones(25)/25, mode="same")
    df_new += noise_amp * smooth_noise[:, None]

    return df_new, valgus_deg


def perturb_grf(df, severity):
    df_new = df.copy()
    n = len(df_new)
    t = np.linspace(0, 1, n)

    base_scale = SEVERITY_CONFIG[severity]["grf_scale"]

    if "ground_force_vy" in df_new.columns:
        df_new["ground_force_vy"] *= base_scale
        stance_window = np.exp(-((t - 0.5) ** 2) / (2 * 0.12 ** 2))
        overload_factor = 1 + (severity * 0.10) * stance_window
        df_new["ground_force_vy"] *= overload_factor

    if "ground_force_vx" in df_new.columns:
        df_new["ground_force_vx"] *= (1 + severity * 0.05)

    noise_amp = 0.002 * severity
    smooth_noise = np.convolve(np.random.randn(n), np.ones(25)/25, mode="same")

    for col in df_new.columns:
        if "ground_force" in col:
            df_new[col] += noise_amp * smooth_noise

    return df_new, base_scale


# -------------------------
# Inverse Dynamics
# -------------------------

def run_id(jk_path, grf_path, sto_out):
    temp_external = os.path.join(BASE_DIR, "temp_external_loads.xml")

    with open(EXTERNAL_LOADS, "r") as f:
        xml_text = f.read()

    xml_text = re.sub(
        r"<datafile>.*?</datafile>",
        f"<datafile>{grf_path}</datafile>",
        xml_text
    )

    with open(temp_external, "w") as f:
        f.write(xml_text)

    tool = osim.InverseDynamicsTool(SETUP_PATH)
    tool.setModelFileName(MODEL_PATH)
    tool.setCoordinatesFileName(jk_path)
    tool.setResultsDir(OUT_STO)
    tool.setOutputGenForceFileName(os.path.basename(sto_out))
    tool.setExternalLoadsFileName(temp_external)

    tool.run()
    os.remove(temp_external)


# -------------------------
# Load Base Inputs
# -------------------------

jk_header, jk_base = load_mot(BASE_JK)
grf_header, grf_base = load_mot(BASE_GRF)

# -------------------------
# Dataset Generation
# -------------------------

X_data = []
y_labels = []
metadata_rows = []

for severity in SEVERITY_CONFIG.keys():
    for i in range(N_PER_SEVERITY):

        sample_id = f"run_S{severity}_{i:03d}"

        jk_out = os.path.join(OUT_JK, f"{sample_id}_JK.mot")
        grf_out = os.path.join(OUT_GRF, f"{sample_id}_GRF.mot")
        sto_out = os.path.join(OUT_STO, f"{sample_id}.sto")

        jk_new, valgus_deg = perturb_jk(jk_base, severity)
        grf_new, grf_scale = perturb_grf(grf_base, severity)

        save_mot(jk_out, jk_header, jk_new)
        save_mot(grf_out, grf_header, grf_new)

        run_id(jk_out, grf_out, sto_out)

        # ---- Load STO and append to dataset ----
        sto_array = load_sto_trimmed(sto_out)

        X_data.append(sto_array)
        y_labels.append(severity)

        metadata_rows.append({
            "sample_id": sample_id,
            "severity": severity,
            "valgus_deg": valgus_deg,
            "grf_scale": grf_scale
        })

        print(f"Completed: {sample_id}")

# Convert to NumPy
X_data = np.array(X_data)   # shape: (samples, 72, 39)
y_labels = np.array(y_labels)

# Save
np.save(os.path.join(DATASET_DIR, "X.npy"), X_data)
np.save(os.path.join(DATASET_DIR, "y.npy"), y_labels)

pd.DataFrame(metadata_rows).to_csv(
    os.path.join(DATASET_DIR, "dataset_metadata.csv"),
    index=False
)

print("Dataset generation complete.")
print(f"Final dataset shape: {X_data.shape}")