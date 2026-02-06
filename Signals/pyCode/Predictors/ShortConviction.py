# ABOUTME: Short Interest Conviction Score - Novel factor measuring short seller conviction
# ABOUTME: Identifies shorts who increase positions despite positive price movements (high conviction)

"""
ShortConviction.py

Usage:
    Run from [Repo-Root]/Signals/pyCode/
    python3 Predictors/ShortConviction.py

Inputs:
    - SignalMasterTable.parquet: Monthly master table with columns [permno, gvkey, time_avail_m, ret]
    - monthlyCRSP.parquet: Monthly CRSP data with columns [permno, time_avail_m, shrout]
    - monthlyShortInterest.parquet: Short interest data with columns [gvkey, time_avail_m, shortint]

Outputs:
    - ShortConviction.csv: CSV file with columns [permno, yyyymm, ShortConviction]

Signal Definition:
    Measures short seller conviction by identifying when short interest changes move
    against recent price movements. High positive values indicate short interest
    increased despite positive returns (bearish conviction). Standardized cross-sectionally.

    ShortConviction = -1 * SI_pct_change * cumret_2m

    Where:
    - SI_pct_change = (SI_ratio_t - SI_ratio_t-2) / SI_ratio_t-2
    - cumret_2m = cumulative return over months t-1 and t-2
    - Standardized by month (mean 0, std 1)
    - Winsorized at [-3, 3] to handle outliers
"""

import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.save_standardized import save_predictor
from utils.stata_replication import stata_multi_lag

print("Starting ShortConviction.py...")

# DATA LOAD
print("Loading SignalMasterTable...")
signal_master = pd.read_parquet(
    "../pyData/Intermediate/SignalMasterTable.parquet",
    columns=["permno", "gvkey", "time_avail_m", "ret"],
)
print(f"Loaded {len(signal_master)} rows from SignalMasterTable")

# Load shares outstanding from CRSP
print("Loading monthlyCRSP...")
monthly_crsp = pd.read_parquet(
    "../pyData/Intermediate/monthlyCRSP.parquet",
    columns=["permno", "time_avail_m", "shrout"],
)
df = pd.merge(
    signal_master,
    monthly_crsp,
    on=["permno", "time_avail_m"],
    how="inner",
    validate="1:1",
)
print(f"After merge with monthlyCRSP: {len(df)} rows")

# Preserve observations with missing gvkey (will add back later with missing signal)
df_missing_gvkey = df[df["gvkey"].isna()].copy()

# Keep only observations with valid gvkey for short interest merge
df = df[df["gvkey"].notna()]
print(f"After filtering to valid gvkey: {len(df)} rows")

# Load short interest data
print("Loading monthlyShortInterest...")
monthly_short = pd.read_parquet(
    "../pyData/Intermediate/monthlyShortInterest.parquet",
    columns=["gvkey", "time_avail_m", "shortint"],
)
df = pd.merge(
    df, monthly_short, on=["gvkey", "time_avail_m"], how="left", validate="1:1"
)
print(f"After merge with monthlyShortInterest: {len(df)} rows")

# SIGNAL CONSTRUCTION
print("Calculating ShortConviction signal...")

# Step 1: Calculate short interest ratio
df["SI_ratio"] = df["shortint"] / df["shrout"]

# Step 2: Sort by permno and time for lag calculations
df = df.sort_values(["permno", "time_avail_m"])

# Step 3: Create lagged variables using stata_multi_lag
# Lag returns by 1 and 2 months
df = stata_multi_lag(df, "permno", "time_avail_m", "ret", [1, 2])

# Lag SI_ratio by 2 months
df = stata_multi_lag(df, "permno", "time_avail_m", "SI_ratio", [2])

# Step 4: Calculate cumulative 2-month return
# cumret_2m = (1 + ret_t-1) * (1 + ret_t-2) - 1
df["cumret_2m"] = (1 + df["ret_lag1"]) * (1 + df["ret_lag2"]) - 1

# Step 5: Calculate percentage change in short interest ratio
# SI_pct_change = (SI_ratio_t - SI_ratio_t-2) / SI_ratio_t-2
df["SI_pct_change"] = (df["SI_ratio"] - df["SI_ratio_lag2"]) / df["SI_ratio_lag2"]

# Step 6: Calculate raw conviction score
# High positive values = SI increased while price rose (bearish conviction)
# Negative value = SI decreased while price fell (capitulation/bullish)
df["ShortConviction_raw"] = -df["SI_pct_change"] * df["cumret_2m"]

# Step 7: Winsorize at [-3, 3] to handle extreme outliers
df["ShortConviction_raw"] = df["ShortConviction_raw"].clip(-3, 3)

# Step 8: Standardize cross-sectionally by month
# Calculate mean and std by month
monthly_stats = df.groupby("time_avail_m")["ShortConviction_raw"].agg(["mean", "std"])
df = df.merge(
    monthly_stats, left_on="time_avail_m", right_index=True, how="left", suffixes=("", "_monthly")
)

# Standardize: (x - mean) / std
# Handle cases where std is 0 or NaN
df["ShortConviction"] = np.where(
    (df["std"] > 0) & df["ShortConviction_raw"].notna(),
    (df["ShortConviction_raw"] - df["mean"]) / df["std"],
    np.nan
)

# Step 9: Clean up temporary columns
df = df[["permno", "time_avail_m", "ShortConviction"]].copy()

# Step 10: Append back observations with missing gvkey (will have missing ShortConviction)
df_missing_gvkey["ShortConviction"] = np.nan
df = pd.concat([df, df_missing_gvkey[["permno", "time_avail_m", "ShortConviction"]]], ignore_index=True)

print(f"Calculated ShortConviction for {df['ShortConviction'].notna().sum()} observations")

# SAVE
save_predictor(df, "ShortConviction")
print("ShortConviction.py completed successfully")
