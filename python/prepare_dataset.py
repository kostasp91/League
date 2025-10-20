#!/usr/bin/env python3
"""
prepare_dataset.py — leak-free features + volume-aware hot hand + participation signals

Creates:
  - processed_train_dataset.csv
  - latest_week_data.csv

Usage:
  python prepare_dataset.py --input_excel players_data_dunkest.xlsx --out_dir ../TrainingModel
"""
import argparse
import os
import re
from typing import List
import pandas as pd


def log(msg: str):
    print(f"[prepare_dataset] {msg}")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]
    syn_map = {
        "player": "Player", "name": "Player",
        "pos": "Pos", "position": "Pos",
        "team": "Team",
        "fpt": "FPT", "fp": "FPT", "fantasypoints": "FPT",
        "cr": "CR", "cost": "CR", "price": "CR",
        "min": "MIN", "minutes": "MIN",
        "fgm": "FGM", "fga": "FGA",
        "3pm": "3PM", "3pa": "3PA",
        "ftm": "FTM", "fta": "FTA",
        "reb": "REB", "ast": "AST", "stl": "STL",
        "blk": "BLK", "tov": "TOV", "pf": "PF",
        "week": "Week",
    }
    rename = {c: syn_map.get(c.lower(), c) for c in df.columns}
    return df.rename(columns=rename)


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col not in {"Player", "Pos", "Team", "SheetName"}:
            out[col] = pd.to_numeric(out[col], errors="ignore")
    num_cols = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
    out[num_cols] = out[num_cols].fillna(0)
    return out


def ensure_boxscore_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols_needed = ["FGM","FGA","3PM","3PA","FTM","FTA","REB","AST","STL","BLK","TOV","PF"]
    out = df.copy()
    for c in cols_needed:
        if c not in out.columns: out[c] = 0
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)
    return out


def leak_free_expanding_mean(series: pd.Series) -> pd.Series:
    # Average up to the previous observation (no leakage)
    return series.shift(1).expanding().mean()


def compute_features(full_df: pd.DataFrame) -> pd.DataFrame:
    required = ["Player", "Pos", "Team", "FPT", "CR", "MIN", "Week"]
    for c in required:
        if c not in full_df.columns:
            raise ValueError(f"Missing required column: {c}")

    # Sort for time-aware ops
    full_df = full_df.sort_values(["Player", "Week"]).reset_index(drop=True)

    # --- Time & form ---
    full_df["lastFPT"] = full_df.groupby("Player")["FPT"].shift(1)
    full_df["prevFPT"] = full_df.groupby("Player")["FPT"].shift(1)
    full_df["roll3FPT"] = full_df.groupby("Player")["prevFPT"].transform(
        lambda s: s.rolling(3, min_periods=1).mean()
    )

    # --- Percentages (row-level) ---
    full_df["FG_pct"] = (full_df["FGM"] / full_df["FGA"]).replace([float("inf")], 0).fillna(0)
    full_df["TP_pct"] = (full_df["3PM"] / full_df["3PA"]).replace([float("inf")], 0).fillna(0)
    full_df["FT_pct"] = (full_df["FTM"] / full_df["FTA"]).replace([float("inf")], 0).fillna(0)

    # --- Leak-free expanding averages ---
    full_df["avgFPT"] = full_df.groupby("Player")["FPT"].transform(leak_free_expanding_mean).fillna(0)
    for col in ["REB","AST","STL","BLK","TOV","PF","FG_pct","TP_pct","FT_pct"]:
        full_df[f"avg_{col}"] = full_df.groupby("Player")[col].transform(leak_free_expanding_mean).fillna(0)

    # --- Minutes & participation (leak-free) ---
    full_df["played"] = (full_df["MIN"] > 0).astype(int)
    full_df["avgMIN"] = (
        full_df.groupby("Player")["MIN"].transform(lambda s: s.shift(1).expanding().mean()).fillna(0)
    )
    full_df["prevMIN"]  = full_df.groupby("Player")["MIN"].shift(1)
    full_df["roll3MIN"] = full_df.groupby("Player")["prevMIN"].transform(
        lambda s: s.rolling(3, min_periods=1).mean()
    ).fillna(0)

    full_df["games_so_far"] = full_df.groupby("Player").cumcount()
    full_df["games_played_so_far"] = full_df.groupby("Player")["played"].cumsum().shift(1).fillna(0)
    full_df["availability"] = (
        (full_df["games_played_so_far"] / full_df["games_so_far"].replace(0, pd.NA))
        .fillna(1.0).clip(0,1)
    )

    # FPM & avgFPM (leak-free)
    full_df["fpm"] = (full_df["FPT"] / full_df["MIN"].replace(0, pd.NA)).fillna(0)
    full_df["avgFPM"] = full_df.groupby("Player")["fpm"].transform(
        lambda s: s.shift(1).expanding().mean()
    ).fillna(0)

    # --- P(play) (leak-free, recent) ---
    p_ema = (
        full_df.groupby("Player")["played"]
        .apply(lambda s: s.shift(1).ewm(alpha=0.6, adjust=False).mean())
        .reset_index(level=0, drop=True).fillna(1.0)
    )
    p_recent5 = (
        full_df.groupby("Player")["played"]
        .apply(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
        .reset_index(level=0, drop=True).fillna(1.0)
    )
    full_df["p_play"] = 0.6 * p_ema + 0.4 * p_recent5

    # --- Hot-hand (volume-aware) ---
    full_df["avg_3PA"] = full_df.groupby("Player")["3PA"].transform(leak_free_expanding_mean).fillna(0)
    full_df["HotHandAdj"] = 1.0
    mask = (full_df["TP_pct"] > 0.40) & (full_df["avg_3PA"] < 4)
    full_df.loc[mask, "HotHandAdj"] = 0.85
    full_df["adjEfficiency"] = full_df["avgFPT"] * full_df["HotHandAdj"]

    # --- Target ---
    full_df["NextFantasyPoints"] = full_df.groupby("Player")["FPT"].shift(-1)

    return full_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_excel", type=str, default="players_data_dunkest.xlsx")
    parser.add_argument("--out_dir", type=str, default="../TrainingModel")
    args = parser.parse_args()

    xls = pd.ExcelFile(args.input_excel)
    log(f"Found sheets: {xls.sheet_names}")

    frames: List[pd.DataFrame] = []
    for idx, sheet in enumerate(xls.sheet_names, start=1):
        df = pd.read_excel(xls, sheet_name=sheet)
        df["SheetName"] = sheet
        df = normalize_columns(df)
        m = re.search(r"(\d+)", sheet)
        df["Week"] = int(m.group(1)) if m else idx
        df = ensure_boxscore_columns(df)
        df = coerce_numeric(df)
        frames.append(df)

    full_df = pd.concat(frames, ignore_index=True)
    log(f"Combined total rows: {len(full_df)}")

    musts = ["Player", "Pos", "Team", "FPT", "CR", "MIN", "Week"]
    for c in musts:
        if c not in full_df.columns:
            raise ValueError(f"Missing column after normalization: {c}")
    for c in ["Player", "Pos", "Team", "SheetName"]:
        if c in full_df.columns:
            full_df[c] = full_df[c].astype(str).str.strip()

    full_df = compute_features(full_df)

    latest_week = int(full_df["Week"].max())
    log(f"Latest week detected: {latest_week}")

    train_df = full_df[(full_df["Week"] < latest_week) & (full_df["NextFantasyPoints"].notna())].copy()
    latest_df = full_df[full_df["Week"] == latest_week].copy()

    os.makedirs(args.out_dir, exist_ok=True)
    train_out = os.path.join(args.out_dir, "processed_train_dataset.csv")
    latest_out = os.path.join(args.out_dir, "latest_week_data.csv")

    train_df.to_csv(train_out, index=False)
    latest_df.to_csv(latest_out, index=False)

    log(f"Training dataset saved: {train_out} ({len(train_df)} rows)")
    log(f"Latest-week dataset saved: {latest_out} ({len(latest_df)} rows)")
    log("Done.")


if __name__ == "__main__":
    main()
