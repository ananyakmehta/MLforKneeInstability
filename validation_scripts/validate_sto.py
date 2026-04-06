import os
import numpy as np
import opensim as osim

STO_DIR = "generated/sto"

def validate_sto_file(filepath):
    print(f"\nChecking: {os.path.basename(filepath)}")

    # Try loading with OpenSim
    try:
        table = osim.TimeSeriesTable(filepath)
    except Exception as e:
        print("Cannot load file:", e)
        return False

    # Basic dimensions
    n_rows = table.getNumRows()
    n_cols = table.getNumColumns()

    print("Rows:", n_rows)
    print("Columns:", n_cols)

    if n_rows == 0:
        print("No data rows")
        return False

    # Time vector check
    time = np.array(table.getIndependentColumn())

    if not np.all(np.diff(time) > 0):
        print("Time not strictly increasing")
        return False

    # Data matrix
    data = table.getMatrix().to_numpy()

    if np.isnan(data).any():
        print("NaNs detected")
        return False

    max_val = np.max(np.abs(data))
    print("Max abs value:", max_val)

    if max_val > 1e6:
        print("Suspiciously large values")

    print("File looks valid")
    return True


def main():
    files = [f for f in os.listdir(STO_DIR) if f.endswith(".sto")]

    if not files:
        print("No STO files found.")
        return

    print("Found", len(files), "STO files")

    all_valid = True

    for f in files[:5]:  # check first 5 for quick validation
        valid = validate_sto_file(os.path.join(STO_DIR, f))
        if not valid:
            all_valid = False

    if all_valid:
        print("\n All checked files look good!")
    else:
        print("\n Some files failed validation.")


if __name__ == "__main__":
    main()