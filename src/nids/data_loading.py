"""Stage 1 (Data Acquisition & Partition) + Stage 2 cleaning half of the pipeline.

CICFlowMeter CSV headers vary slightly in spacing/casing between dataset
releases (" Destination Port" vs "Destination Port", "Flow Bytes/s" vs
"Flow Bytesss/s" in some corrupted mirrors, etc). Everything here works off
a *normalized* column-name space (lower snake_case) so the rest of the
package never has to special-case a raw CSV header again.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import config

logger = logging.getLogger(__name__)

_WS_RE = re.compile(r"[^0-9a-zA-Z]+")


def normalize_col(name: str) -> str:
    """'  Destination Port ' -> 'destination_port'; 'Flow Bytes/s' -> 'flow_bytes_s'."""
    name = name.strip().lower()
    name = _WS_RE.sub("_", name)
    return name.strip("_")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: normalize_col(c) for c in df.columns})
    # CICFlowMeter sometimes duplicates 'Fwd Header Length' as
    # 'Fwd Header Length.1' -> normalizes to 'fwd_header_length_1'; keep as-is,
    # it is harmless duplication and MI-based ranking (Stage 3) will simply
    # rank it low.
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]
    return df


# Canonical label lookup: maps many raw-label spellings (encoding artifacts,
# inconsistent hyphenation/whitespace/casing across the 7-8 daily files) to
# exactly one of the 15 config.CLASSES. Necessary per Review §3 ("Label
# normalization: Necessary -- CIC-IDS2017 labels have inconsistent
# capitalisation and spacing across daily files").
_LABEL_ALIASES: Dict[str, str] = {}
for _c in config.CLASSES:
    _LABEL_ALIASES[_c.lower()] = _c
_LABEL_ALIASES.update({
    "benign": "BENIGN",
    "web attack \x96 brute force": "Web Attack - Brute Force",
    "web attack – brute force": "Web Attack - Brute Force",
    "web attack  brute force": "Web Attack - Brute Force",
    "web attack \x96 xss": "Web Attack - XSS",
    "web attack – xss": "Web Attack - XSS",
    "web attack \x96 sql injection": "Web Attack - Sql Injection",
    "web attack – sql injection": "Web Attack - Sql Injection",
    "web attack sql injection": "Web Attack - Sql Injection",
    "ftp-patator": "FTP-Patator",
    "ssh-patator": "SSH-Patator",
    "dos hulk": "DoS Hulk",
    "dos goldeneye": "DoS GoldenEye",
    "dos slowloris": "DoS slowloris",
    "dos slowhttptest": "DoS Slowhttptest",
    "portscan": "PortScan",
    "ddos": "DDoS",
    "infiltration": "Infiltration",
    "heartbleed": "Heartbleed",
    "bot": "Bot",
})


def normalize_label(raw: pd.Series) -> pd.Series:
    cleaned = (
        raw.astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.replace("–", "-", regex=False)
        .str.replace("\x96", "-", regex=False)
        .str.strip()
    )
    key = cleaned.str.lower().str.replace(r"\s*-\s*", " - ", regex=True).str.strip()
    mapped = key.map(_LABEL_ALIASES)
    unmapped = mapped.isna()
    if unmapped.any():
        # Fall back to fuzzy contains-match, then leave truly unknown labels
        # as-is (surfaced loudly rather than silently dropped).
        for raw_val in cleaned[unmapped].unique():
            low = raw_val.lower()
            hit = next((c for c in config.CLASSES if c.lower() in low or low in c.lower()), None)
            if hit:
                mapped[cleaned == raw_val] = hit
        still_unmapped = mapped.isna()
        if still_unmapped.any():
            bad = cleaned[still_unmapped].unique().tolist()
            logger.warning("Unrecognised label values kept as-is: %s", bad)
            mapped[still_unmapped] = cleaned[still_unmapped]
    return mapped


def discover_day_files(data_dir: Path) -> Dict[str, List[Path]]:
    """Group every *.csv in data_dir by which weekday name appears in its filename."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {data_dir}. Update nids.config.DATA_DIR."
        )
    csvs = sorted(data_dir.glob("*.csv")) + sorted(data_dir.glob("*.CSV"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found under {data_dir}")

    day_files: Dict[str, List[Path]] = {d: [] for d in config.DAY_NAMES}
    for f in csvs:
        low = f.name.lower()
        matched = [d for d in config.DAY_NAMES if d.lower() in low]
        if not matched:
            logger.warning("Could not infer weekday for file %s -- skipped", f.name)
            continue
        day_files[matched[0]].append(f)
    missing = [d for d, files in day_files.items() if not files]
    if missing:
        logger.warning("No CSV files found for day(s): %s", missing)
    return day_files


_TS_FORMATS = ["%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M", "%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M"]


def _parse_timestamps(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", dayfirst=False)
    if parsed.isna().mean() > 0.5:
        for fmt in _TS_FORMATS:
            trial = pd.to_datetime(series, format=fmt, errors="coerce")
            if trial.isna().mean() < parsed.isna().mean():
                parsed = trial
    return parsed


def load_raw_data(
    data_dir: Optional[Path] = None,
    nrows_per_file: Optional[int] = None,
) -> pd.DataFrame:
    """Stage 1: load every daily CSV, tag provenance, sort chronologically.

    Adds metadata columns: day, day_order, source_file, partition, timestamp.
    IP/port identifier columns are RETAINED at this stage (Stage 4 session
    reconstruction needs them) -- they are excluded from the model feature
    matrix later, in features.py, never here.
    """
    data_dir = Path(data_dir or config.DATA_DIR)
    day_files = discover_day_files(data_dir)

    frames = []
    for day, files in day_files.items():
        for f in files:
            logger.info("Loading %s (day=%s)", f.name, day)
            df = pd.read_csv(f, low_memory=False, nrows=nrows_per_file)
            df = normalize_columns(df)
            df["day"] = day
            df["day_order"] = config.DAY_ORDER[day]
            df["source_file"] = f.name
            frames.append(df)

    if not frames:
        raise RuntimeError("No data loaded -- check config.DATA_DIR and file naming.")

    data = pd.concat(frames, axis=0, ignore_index=True, sort=False)

    if "timestamp" in data.columns:
        data["timestamp"] = _parse_timestamps(data["timestamp"])
    if "timestamp" not in data.columns or data["timestamp"].isna().all():
        logger.warning(
            "No usable 'timestamp' column found -- falling back to a synthetic "
            "monotonic clock derived from (day_order, row order). Session "
            "timing (tau, TC_t) will use 1-second synthetic spacing; replace "
            "with the real CICFlowMeter Timestamp column if available."
        )
        synthetic_seconds = data.groupby("day_order").cumcount().astype(float)
        data["timestamp"] = pd.to_datetime(data["day_order"] * 86400 + synthetic_seconds, unit="s")
    else:
        # Rows with unparsable timestamps still need an ordering; fall back
        # to file row-order jittered by 1ms so they do not all collide.
        na_mask = data["timestamp"].isna()
        if na_mask.any():
            logger.warning("%d rows had unparsable timestamps; imputing via row order.", na_mask.sum())
            fallback = pd.to_datetime(data["day_order"] * 86400 + np.arange(len(data)) * 1e-3, unit="s")
            data.loc[na_mask, "timestamp"] = fallback[na_mask]

    data = data.sort_values(["day_order", "timestamp"]).reset_index(drop=True)

    if config.LABEL_COLUMN not in data.columns:
        # some releases spell it 'Label ' with trailing space -> already
        # normalized to 'label'; if truly absent this is a hard error.
        raise KeyError("No 'label' column found after normalization; check source CSVs.")
    data[config.LABEL_COLUMN] = normalize_label(data[config.LABEL_COLUMN])

    data["partition"] = data["day"].map(config.SPLIT_MAP)

    logger.info(
        "Loaded %d flows across %d files. Partition sizes: %s",
        len(data), sum(len(v) for v in day_files.values()),
        data["partition"].value_counts().to_dict(),
    )
    return data


def clean_data(df: pd.DataFrame, numeric_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """Stage 2 cleaning: Inf -> NaN -> drop (CIC-IDS2017's known ~1.8% corrupt rows).

    Only numeric feature columns are inspected; metadata / identifier /
    label columns are left untouched.
    """
    df = df.copy()
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in ("day_order",)]

    before = len(df)
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    bad_mask = df[numeric_cols].isna().any(axis=1)
    dropped_pct = 100.0 * bad_mask.mean()
    df = df.loc[~bad_mask].reset_index(drop=True)
    logger.info(
        "Cleaning: dropped %d/%d rows (%.2f%%) with Inf/NaN in numeric features.",
        before - len(df), before, dropped_pct,
    )
    return df


def split_partitions(df: pd.DataFrame):
    """Returns (train_df, val_df, test_df) -- chronological, session-safe."""
    train = df.loc[df["partition"] == "train"].reset_index(drop=True)
    val = df.loc[df["partition"] == "val"].reset_index(drop=True)
    test = df.loc[df["partition"] == "test"].reset_index(drop=True)
    return train, val, test
