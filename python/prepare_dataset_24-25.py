import pandas as pd

def prepare_dataset(excel_path: str):
    # --- Load all sheets ---
    xls = pd.ExcelFile(excel_path)
    print("Sheets found:", xls.sheet_names)

    all_data = []
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name)
        df["WeekName"] = sheet_name  # Keep original sheet name
        all_data.append(df)

    # --- Combine all weeks ---
    full_df = pd.concat(all_data, ignore_index=True)
    print(f"Loaded {len(full_df)} total rows from {len(xls.sheet_names)} weeks")

    # --- Normalize column names ---
    full_df.columns = [c.strip().replace(" ", "_") for c in full_df.columns]

    # Ensure required columns exist
    required_cols = ["Player", "Pos", "Team", "FPT", "CR", "Minutes", "Week"]
    for col in required_cols:
        if col not in full_df.columns:
            raise ValueError(f"Missing required column: {col}")

    # --- Sort by player and week ---
    full_df = full_df.sort_values(by=["Player", "Week"]).reset_index(drop=True)

    # --- Compute next week's target ---
    full_df["NextFantasyPoints"] = full_df.groupby("Player")["FPT"].shift(-1)

    # --- Split datasets ---
    train_df = full_df.dropna(subset=["NextFantasyPoints"]).copy()
    latest_week_df = full_df.loc[full_df.groupby("Player")["Week"].idxmax()].copy()

    # --- Save CSVs ---
    train_path = "../TrainingModel/processed_train_dataset_24-25.csv"
    latest_path = "../TrainingModel/latest_week_dataset_24-25.csv"
    train_df.to_csv(train_path, index=False)
    latest_week_df.to_csv(latest_path, index=False)

    print(f"Training dataset saved: {train_path} ({len(train_df)} rows)")
    print(f"Latest week dataset saved: {latest_path} ({len(latest_week_df)} players)")

    return train_df, latest_week_df

if __name__ == "__main__":
    excel_file = "players_data_dunkest_24-25.xlsx"  # Adjust path if needed
    prepare_dataset(excel_file)
