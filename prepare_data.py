"""
Task 2 - Data preparation for GEPIII electricity data (2016).

Usage:
    python prepare_data.py path/to/raw_folder

The raw folder must contain the Kaggle ASHRAE files:
    train.csv, weather_train.csv, building_metadata.csv

Outputs (in data/processed/):
    electricity_2016_features.parquet (or .csv.gz if pyarrow is missing)
    prep_report.txt   <- paste these numbers into the report placeholders
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

# ----------------------------- settings ---------------------------------
MIN_ZERO_RUN = 24            # remove zero runs of at least this many hours
APPLY_SITE0_KBTU_FIX = True  # site 0 electricity is widely reported to be in kBTU
KBTU_TO_KWH = 0.2931         # verify against [1], [2] and the Kaggle discussion
TRAIN_END = "2016-09-13"     # train: before this date  (~70%)
VAL_END = "2016-11-07"       # val: before this date    (~15%); test: the rest
WEATHER_FFILL_LIMIT = 6      # hours of forward-fill allowed for weather gaps
LAGS = (24, 48, 168)         # all >= 24 h so a 24 h-ahead forecast does not leak

RAW = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

lines = []


def note(msg):
    print(msg)
    lines.append(msg)


# ----------------------------- load -------------------------------------
train = pd.read_csv(RAW / "train.csv", parse_dates=["timestamp"])
weather = pd.read_csv(RAW / "weather_train.csv", parse_dates=["timestamp"])
meta = pd.read_csv(RAW / "building_metadata.csv")

note("== RAW DATA ==")
note(f"train.csv rows: {len(train):,} | buildings: {train.building_id.nunique():,}")
note(f"weather_train.csv rows: {len(weather):,} | sites: {weather.site_id.nunique()}")
note(f"building_metadata.csv rows: {len(meta):,} | sites: {meta.site_id.nunique()}")
note("Meter types (0=electricity): "
     + str(train.meter.value_counts().sort_index().to_dict()))

# ----------------------------- filter -----------------------------------
df = train[train.meter == 0].drop(columns="meter").copy()
df = df[df.timestamp.dt.year == 2016]
note("\n== ELECTRICITY, 2016 ==")
note(f"rows: {len(df):,} | buildings: {df.building_id.nunique():,}")

# ----------------------------- merge ------------------------------------
df = df.merge(meta, on="building_id", how="left")
df = df.merge(weather, on=["site_id", "timestamp"], how="left")
note(f"rows after merge: {len(df):,} | sites: {df.site_id.nunique()}")

note("\n== MISSING VALUES AFTER MERGE (% of rows) ==")
miss = (df.isna().mean() * 100).round(1)
for col, pct in miss[miss > 0].sort_values(ascending=False).items():
    note(f"{col}: {pct}%")

# ----------------------------- clean ------------------------------------
note("\n== CLEANING ==")
neg = (df.meter_reading < 0).sum()
df = df[df.meter_reading >= 0]
note(f"negative readings removed: {neg:,}")

if APPLY_SITE0_KBTU_FIX:
    m = df.site_id == 0
    df.loc[m, "meter_reading"] *= KBTU_TO_KWH
    note(f"site 0 readings converted kBTU -> kWh (x{KBTU_TO_KWH}): {m.sum():,} rows")

df = df.sort_values(["building_id", "timestamp"]).reset_index(drop=True)
is_zero = df.meter_reading.eq(0)
new_run = (is_zero != is_zero.shift()) | (df.building_id != df.building_id.shift())
run_id = new_run.cumsum()
run_len = is_zero.groupby(run_id).transform("size")
bad = is_zero & (run_len >= MIN_ZERO_RUN)
note(f"zero readings total: {is_zero.sum():,}")
note(f"zero runs >= {MIN_ZERO_RUN} h removed: {bad.sum():,} rows "
     f"from {df.loc[bad, 'building_id'].nunique()} buildings")
df = df[~bad].reset_index(drop=True)
note(f"isolated/short zero readings kept: {df.meter_reading.eq(0).sum():,}")

# Outliers are flagged, not removed (report how many)
g = df.groupby("building_id").meter_reading
z = (df.meter_reading - g.transform("mean")) / g.transform("std").replace(0, np.nan)
df["outlier_flag"] = (z.abs() > 5).astype("int8")
note(f"readings > 5 std from building mean (flagged, kept): {df.outlier_flag.sum():,}")

# Weather gaps: short forward-fill within site (no use of future values)
wcols = ["air_temperature", "cloud_coverage", "dew_temperature",
         "precip_depth_1_hr", "sea_level_pressure", "wind_direction", "wind_speed"]
df = df.sort_values(["site_id", "building_id", "timestamp"])
for c in ("cloud_coverage", "precip_depth_1_hr", "year_built", "floor_count"):
    df[c + "_missing"] = df[c].isna().astype("int8")
df[wcols] = df.groupby(["site_id", "building_id"])[wcols].ffill(limit=WEATHER_FFILL_LIMIT)
df = df.sort_values(["building_id", "timestamp"]).reset_index(drop=True)

# ----------------------------- split ------------------------------------
df["split"] = np.where(df.timestamp < TRAIN_END, "train",
                       np.where(df.timestamp < VAL_END, "val", "test"))
note("\n== CHRONOLOGICAL SPLIT ==")
for s in ("train", "val", "test"):
    sub = df[df.split == s]
    note(f"{s}: {len(sub):,} rows ({len(sub) / len(df):.1%}) | "
         f"{sub.timestamp.min().date()} to {sub.timestamp.max().date()}")

# Remaining gaps: fill with TRAIN-ONLY medians (no leakage)
fill_cols = wcols + ["year_built", "floor_count"]
medians = df.loc[df.split == "train", fill_cols].median()
df[fill_cols] = df[fill_cols].fillna(medians)

# ----------------------------- features ---------------------------------
ts = df.timestamp
df["hour"] = ts.dt.hour
df["dayofweek"] = ts.dt.dayofweek
df["month"] = ts.dt.month
df["is_weekend"] = (df.dayofweek >= 5).astype("int8")
df["hour_sin"] = np.sin(2 * np.pi * df.hour / 24)
df["hour_cos"] = np.cos(2 * np.pi * df.hour / 24)
hol = USFederalHolidayCalendar().holidays("2016-01-01", "2016-12-31")
df["is_holiday"] = ts.dt.normalize().isin(hol).astype("int8")  # approximation (US calendar)

df["log_sqft"] = np.log1p(df.square_feet)
df["building_age"] = 2016 - df.year_built
df["primary_use"] = df.primary_use.astype("category")

base = df[["building_id", "timestamp", "meter_reading"]]
for h in LAGS:
    lag = base.copy()
    lag["timestamp"] = lag.timestamp + pd.Timedelta(hours=h)
    df = df.merge(lag.rename(columns={"meter_reading": f"lag_{h}h"}),
                  on=["building_id", "timestamp"], how="left")
df["lag_mean_24_48"] = df[["lag_24h", "lag_48h"]].mean(axis=1)

note("\n== FEATURES ==")
note("time: hour, dayofweek, month, is_weekend, hour_sin, hour_cos, is_holiday")
note("weather: " + ", ".join(wcols))
note("building: primary_use, square_feet, log_sqft, year_built, building_age, floor_count")
note("lag: " + ", ".join(f"lag_{h}h" for h in LAGS) + ", lag_mean_24_48")
note(f"rows with missing lag_24h (first day of each series or gaps): "
     f"{df.lag_24h.isna().sum():,}")

note("\n== FINAL TABLE ==")
note(f"rows: {len(df):,} | columns: {df.shape[1]} | buildings: {df.building_id.nunique():,}"
     f" | sites: {df.site_id.nunique()}")
note("Target summary (kWh): " + df.meter_reading.describe().round(2).to_dict().__str__())

# ----------------------------- save -------------------------------------
try:
    df.to_parquet(OUT / "electricity_2016_features.parquet", index=False)
    note("saved: data/processed/electricity_2016_features.parquet")
except ImportError:
    df.to_csv(OUT / "electricity_2016_features.csv.gz", index=False)
    note("saved: data/processed/electricity_2016_features.csv.gz")

(OUT / "prep_report.txt").write_text("\n".join(lines), encoding="utf-8")
print("\nDone. Paste prep_report.txt numbers into the report.")
