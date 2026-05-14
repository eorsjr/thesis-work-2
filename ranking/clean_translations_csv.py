import pandas as pd

# Load CSV without limiting columns
df_raw = pd.read_csv("./data/input/translation_versions.csv", header=None)

# Keep only the first 12 columns
df_trimmed = df_raw.iloc[:, :12]

# Drop rows that contain any NaN within these 12 required columns
valid_rows = df_trimmed.dropna(axis=0, how="any")

# Reset index
valid_rows = valid_rows.reset_index(drop=True)

# Save cleaned CSV
valid_rows.to_csv("./data/input/translation_versions_cleaned.csv", index=False, header=False)

print(f"Original rows: {len(df_raw)}")
print(f"Cleaned rows: {len(valid_rows)}")
print("Saved cleaned CSV.")