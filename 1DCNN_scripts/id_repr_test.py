#id reproducibiluty test
import os
import numpy as np
import opensim as osim
import pandas as pd
import re

# ===============================================
# SETTINGS
# ===============================================
NUM_TRIALS = 50
TARGET_FRAMES = 72
VARIATION_THRESHOLD = 0.05  # 5%
PASS_RATIO_REQUIRED = 0.90  # 90%

BASE_DIR = os.getcwd()

MODEL_PATH = os.path.join(BASE_DIR, "constant/Rajagopal2016.osim")
SETUP_PATH = os.path.join(BASE_DIR, "constant/setup_id.xml")
EXTERNAL_LOADS = os.path.join(BASE_DIR, "constant/external_loads.xml")

BASE_JK = os.path.join(BASE_DIR, "base_inputs/running_JK_base.mot")
BASE_GRF = os.path.join(BASE_DIR, "base_inputs/running_GRF_base.mot")

OUT_STO = os.path.join(BASE_DIR, "generated/sto_repro")
os.makedirs(OUT_STO, exist_ok=True)


# ===============================================
# UTILITIES
# ===============================================

def load_sto_trimmed(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    header_end = None
    for i, line in enumerate(lines):
        if "endheader" in line.lower():
            header_end = i
            break

    df = pd.read_csv(filepath, sep="\t", skiprows=header_end+1)

    if "time" in df.columns:
        df = df.drop(columns=["time"])

    if len(df) > TARGET_FRAMES:
        df = df.iloc[:TARGET_FRAMES]

    return df.values


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


# ===============================================
# REPRODUCIBILITY TEST
# ===============================================

print("\nTRIAL,MeanPercentVariation,Pass")

baseline_output = None
all_variations = []

for trial in range(NUM_TRIALS):

    sto_out = os.path.join(OUT_STO, f"repro_{trial:03d}.sto")

    # Run identical ID
    run_id(BASE_JK, BASE_GRF, sto_out)

    current_output = load_sto_trimmed(sto_out)

    if trial == 0:
        baseline_output = current_output
        print(f"{trial+1},0.0000,PASS")
        all_variations.append(0.0)
        continue

    # Percent variation per element
    epsilon = 1e-8
    percent_diff = np.abs(current_output - baseline_output) / (np.abs(baseline_output) + epsilon)

    mean_variation = np.mean(percent_diff)
    all_variations.append(mean_variation)

    passed = mean_variation <= VARIATION_THRESHOLD

    print(f"{trial+1},{mean_variation:.6f},{'PASS' if passed else 'FAIL'}")


# ===============================================
# SUMMARY
# ===============================================

all_variations = np.array(all_variations)

pass_ratio = np.mean(all_variations <= VARIATION_THRESHOLD)

print("\nSUMMARY")
print(f"MeanVariation,{np.mean(all_variations):.6f}")
print(f"StdVariation,{np.std(all_variations):.6f}")
print(f"PassRatio,{pass_ratio:.2f}")
print("FINAL PASS" if pass_ratio >= PASS_RATIO_REQUIRED else "FINAL FAIL")