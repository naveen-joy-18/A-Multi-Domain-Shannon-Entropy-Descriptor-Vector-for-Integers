"""
figures/data_loader.py
=======================
The whole point of this module: nobody should have to hand-type a file
path before a figure gets made. Give it the project root (or nothing --
it searches from cwd upward) and it finds the fingerprint data itself,
the same way a good crew finds the vault without needing a hand-drawn
map for every single hallway.

Search strategy, in order:
1. An explicit `data_path` argument, if given (still auto-detects
   whether it's a single CSV or a directory of part-files).
2. A consolidated file named `ueft_fingerprints.csv` anywhere under the
   search root (recursive, shallow-first).
3. A directory named `ueft_output` (or containing files matching
   `part_########_########.csv`) anywhere under the search root --
   these are concatenated on the fly.

Raises a clear, actionable error if nothing is found -- never silently
returns an empty DataFrame, because a figure silently plotted from
zero rows is far more dangerous than a crash.
"""

from __future__ import annotations
import os
import re
import glob
import pandas as pd
from tqdm import tqdm

CONSOLIDATED_NAME = "ueft_fingerprints.csv"
PART_FILE_PATTERN = re.compile(r"^part_\d{8}_\d{8}\.csv$")


def _find_consolidated_file(root: str) -> str | None:
    for dirpath, _, filenames in os.walk(root):
        if CONSOLIDATED_NAME in filenames:
            return os.path.join(dirpath, CONSOLIDATED_NAME)
    return None


def _find_part_file_dir(root: str) -> str | None:
    for dirpath, _, filenames in os.walk(root):
        if any(PART_FILE_PATTERN.match(f) for f in filenames):
            return dirpath
    return None


def _load_from_part_dir(part_dir: str) -> pd.DataFrame:
    part_files = sorted(
        f for f in os.listdir(part_dir) if PART_FILE_PATTERN.match(f)
    )
    if not part_files:
        raise FileNotFoundError(f"No part_########_########.csv files found in {part_dir}")
    frames = []
    for f in tqdm(part_files, desc="[data_loader] reading part-files", unit="file"):
        frames.append(pd.read_csv(os.path.join(part_dir, f)))
    df = pd.concat(frames, ignore_index=True).sort_values("n").reset_index(drop=True)
    return df


def load_fingerprints(data_path: str = None, search_root: str = ".") -> pd.DataFrame:
    """
    Auto-locate and load the UEFT fingerprint table.

    Parameters
    ----------
    data_path : str, optional
        Explicit path to either a consolidated CSV or a directory of
        part-files. If given, this takes priority and no search is
        performed.
    search_root : str
        Directory to search from if `data_path` is not given. Defaults
        to the current working directory; searches recursively.

    Returns
    -------
    pd.DataFrame with an 'n' column plus all entropy coordinate columns.
    """
    if data_path is not None:
        if os.path.isdir(data_path):
            print(f"[data_loader] loading part-files from explicit dir: {data_path}")
            return _load_from_part_dir(data_path)
        elif os.path.isfile(data_path):
            print(f"[data_loader] loading consolidated file: {data_path}")
            return pd.read_csv(data_path)
        else:
            raise FileNotFoundError(f"data_path does not exist: {data_path}")

    print(f"[data_loader] no explicit path given -- searching under: {os.path.abspath(search_root)}")

    consolidated = _find_consolidated_file(search_root)
    if consolidated:
        print(f"[data_loader] found consolidated file: {consolidated}")
        return pd.read_csv(consolidated)

    part_dir = _find_part_file_dir(search_root)
    if part_dir:
        print(f"[data_loader] no consolidated file found; found part-files in: {part_dir}")
        return _load_from_part_dir(part_dir)

    raise FileNotFoundError(
        f"Could not find '{CONSOLIDATED_NAME}' or any part_########_########.csv "
        f"files anywhere under '{os.path.abspath(search_root)}'. "
        f"Run the pipeline first (python -m ueft.main --N ...), or pass an "
        f"explicit data_path."
    )


def entropy_columns(df: pd.DataFrame) -> list:
    """
    Auto-detect the entropy coordinate columns (everything starting
    with 'H_') rather than hand-listing them -- so if positional bases
    or new domains are added later, every figure script picks them up
    automatically with zero edits.
    """
    return [c for c in df.columns if c.startswith("H_")]


def find_exemplars(df: pd.DataFrame) -> dict:
    """
    Auto-select exemplar integers for comparison figures, by
    mathematical role rather than a hardcoded list -- so this keeps
    working even if the fingerprint is later regenerated for a
    different N.

    Returns dict: {label: n}
    """
    exemplars = {}

    # A small prime
    small_primes = [p for p in [7, 11, 13, 17, 19, 23] if (df["n"] == p).any()]
    if small_primes:
        exemplars["Prime (n=%d)" % small_primes[0]] = small_primes[0]

    # A prime power (2^k)
    for k in [6, 7, 8, 9, 10]:
        val = 2 ** k
        if (df["n"] == val).any():
            exemplars[f"Prime power (n={val})"] = val
            break

    # A highly composite number within range (pick the largest divisor
    # count available in the dataset)
    if "H_arithmetic" in df.columns:
        # Highly composite numbers have LOW algebraic entropy spread but
        # HIGH divisor count; approximate by minimal H_algebraic among
        # numbers with many distinct prime factors -- simpler: just use
        # a well-known highly composite number if present.
        for hcn in [720720, 720, 360, 60, 12]:
            if (df["n"] == hcn).any():
                exemplars[f"Highly composite (n={hcn})"] = hcn
                break

    # A perfect number (6, 28, 496, 8128)
    for pn in [6, 28, 496, 8128]:
        if (df["n"] == pn).any():
            exemplars[f"Perfect number (n={pn})"] = pn
            break

    return exemplars
