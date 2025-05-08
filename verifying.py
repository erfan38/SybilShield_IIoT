import os
import pandas as pd

DATA_DIR = "data_per_node"
honest_count = 0
sybil_count = 0
errors = []

for fname in sorted(os.listdir(DATA_DIR)):
    if not fname.endswith(".csv"):
        continue

    path = os.path.join(DATA_DIR, fname)
    try:
        df = pd.read_csv(path)

        if "is_sybil" not in df.columns:
            errors.append(f"[✗] Missing 'is_sybil' column in {fname}")
            continue

        if df["is_sybil"].nunique() > 1:
            errors.append(f"[✗] Mixed labels in {fname}")
            continue

        if df.isnull().values.any():
            errors.append(f"[✗] NaN values in {fname}")

        if (df.select_dtypes(include='number') < 0).any().any():
            errors.append(f"[✗] Negative values in numeric features in {fname}")

        if df["is_sybil"].iloc[0]:
            sybil_count += 1
        else:
            honest_count += 1

    except Exception as e:
        errors.append(f"[✗] Failed to read {fname}: {str(e)}")

print(f"\n[✓] Total honest nodes: {honest_count}")
print(f"[✓] Total Sybil clones: {sybil_count}")
print(f"[✓] Verified {honest_count + sybil_count} files")

if errors:
    print("\n[⚠] Issues found:")
    for e in errors:
        print(e)
else:
    print("\n[✓] All files passed integrity checks.")
