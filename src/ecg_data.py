"""Shared data access for the PTB-XL project: labels, rows, torch Dataset."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
SPLITS = ROOT / "splits"


def load_split() -> dict:
    return json.loads((SPLITS / "main_v1.json").read_text(encoding="utf-8"))


def load_meta() -> pd.DataFrame:
    return pd.read_parquet(PROC / "meta.parquet")


def label_columns(meta: pd.DataFrame):
    sup = sorted(c[4:] for c in meta.columns if c.startswith("sup_"))
    rhy = sorted(c[4:] for c in meta.columns if c.startswith("rhy_"))
    return sup, rhy


def build_labels(meta: pd.DataFrame, sup, rhy):
    ys = meta[[f"sup_{c}" for c in sup]].to_numpy(np.float32)
    yr = meta[[f"rhy_{c}" for c in rhy]].to_numpy(np.float32)
    mask_s = meta["has_diag_statement"].to_numpy(bool)
    mask_r = yr.sum(axis=1) > 0          # a record with no rhythm statement is unlabelled
    return ys, yr, mask_s, mask_r


def row_map(meta: pd.DataFrame) -> dict:
    return {int(e): i for i, e in enumerate(meta["ecg_id"].tolist())}


def split_rows(meta: pd.DataFrame, split: dict) -> dict:
    rm = row_map(meta)
    return {k: np.array(sorted(rm[int(e)] for e in split[f"{k}_ecg_ids"]), dtype=np.int64)
            for k in ("train", "val", "test")}


class ECGDataset(Dataset):
    """Reads rows from a (N, 12, 1000) float32 memmap; opens it lazily per worker."""

    def __init__(self, cache: Path, rows, ys, yr, mask_s, mask_r):
        self.cache = str(cache)
        self.rows = rows
        self.ys, self.yr = ys, yr
        self.mask_s, self.mask_r = mask_s, mask_r
        self.mm = None

    def _open(self):
        if self.mm is None:
            self.mm = np.load(self.cache, mmap_mode="r")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        self._open()
        r = int(self.rows[i])
        x = np.array(self.mm[r], dtype=np.float32, copy=True)
        return (torch.from_numpy(x), torch.from_numpy(self.ys[r]),
                torch.from_numpy(self.yr[r]),
                torch.tensor(self.mask_s[r]), torch.tensor(self.mask_r[r]))
