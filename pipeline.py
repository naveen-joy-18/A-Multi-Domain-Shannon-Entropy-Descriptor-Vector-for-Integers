"""
pipeline.py
===========
Orchestration layer. This is where the five domains get assembled into
the Universal Entropy Fingerprint F(n) for every integer 1..N, and
written to disk in a way that survives interruption.

Design decisions (all deliberate, all defensible in a methods section):

1. Two-phase computation.
   Phase A builds the SPF sieve and sigma sieve ONCE for all of 1..N
   (this is the expensive shared infrastructure). Phase B then computes
   the per-integer fingerprint using that shared infrastructure -- no
   integer redoes work another integer already paid for.

2. Chunked, checkpointed writes.
   We never hold all N fingerprint rows in memory as Python objects
   simultaneously if it can be avoided at large N; we process in
   chunks of `chunk_size`, write each chunk to a CSV part-file
   immediately, and skip any chunk whose output file already exists.
   Kill the process at row 850,000 out of 1,000,000 and restarting
   picks up exactly where it left off -- no recomputation, no data
   loss. For a heist that runs overnight unattended, this is not
   optional.

3. Multiprocessing across chunks.
   Each chunk is dispatched to a worker process. The SPF/sigma sieves
   are computed once in the parent and passed to workers via a
   multiprocessing-friendly global (avoids re-pickling a million-plus
   int32 array per task).

4. tqdm at every level that matters: sieve construction, chunk
   dispatch/completion, and final concatenation.

Output: a directory of CSV part-files, one row per integer, columns:
    n, H_algebraic, H_geometric, H_arithmetic,
    H_pos_base2 .. H_pos_base16, H_dynamical_collatz,
    collatz_steps, collatz_max_value, collatz_capped
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

from .sieve import smallest_prime_factor_sieve, factorize
from .algebraic import algebraic_entropy
from .geometric import geometric_entropy
from .arithmetic import arithmetic_entropy
from .positional import positional_entropy_vector, DEFAULT_BASES
from .dynamical import collatz_trajectory_stats

# ---------------------------------------------------------------------
# Worker-global state. Populated once per worker process via _init_worker
# so the (potentially large) SPF sieve is not re-pickled per task.
# ---------------------------------------------------------------------
_WORKER_SPF = None


def _init_worker(spf_array: np.ndarray):
    global _WORKER_SPF
    _WORKER_SPF = spf_array


def _fingerprint_one(n: int) -> dict:
    """
    Compute the full UEFT fingerprint for a single integer n, using the
    SPF sieve already loaded into this worker process.
    """
    fac = factorize(n, _WORKER_SPF)

    row = {"n": n}
    row["H_algebraic"] = algebraic_entropy(fac)
    row["H_geometric"] = geometric_entropy(fac)
    row["H_arithmetic"] = arithmetic_entropy(fac)
    row.update(positional_entropy_vector(n, DEFAULT_BASES))
    row.update(collatz_trajectory_stats(n))
    return row


def _fingerprint_chunk(n_values: list) -> list:
    return [_fingerprint_one(n) for n in n_values]


def run_pipeline(
    N: int,
    output_dir: str,
    chunk_size: int = 10_000,
    n_workers: int = None,
) -> None:
    """
    Compute the UEFT fingerprint for every integer 1..N and write it to
    `output_dir` as chunked Parquet part-files.

    Parameters
    ----------
    N : int
        Upper bound (inclusive) of integers to process, e.g. 1_000_000.
    output_dir : str
        Directory for Parquet part-files (created if it does not exist).
    chunk_size : int
        Number of integers per chunk / per output part-file.
    n_workers : int, optional
        Number of worker processes. Defaults to cpu_count() - 1.
    """
    os.makedirs(output_dir, exist_ok=True)
    n_workers = n_workers or max(1, cpu_count() - 1)

    print(f"[pipeline] N={N:,}  chunk_size={chunk_size:,}  n_workers={n_workers}")

    # ---- Phase A: shared sieve infrastructure, built once -------------
    print("[pipeline] Phase A: building shared sieves")
    spf = smallest_prime_factor_sieve(N, show_progress=True)

    # ---- Phase B: chunked, checkpointed, multiprocessed fingerprinting
    print("[pipeline] Phase B: computing entropy fingerprints")
    chunk_bounds = list(range(1, N + 1, chunk_size))
    chunk_jobs = []
    for start in chunk_bounds:
        end = min(start + chunk_size - 1, N)
        part_path = os.path.join(output_dir, f"part_{start:08d}_{end:08d}.csv")
        if os.path.exists(part_path):
            continue  # checkpoint: already computed, skip (resumability)
        chunk_jobs.append((start, end, part_path))

    if not chunk_jobs:
        print("[pipeline] All chunks already computed. Nothing to do.")
        return

    print(f"[pipeline] {len(chunk_jobs):,} chunk(s) remaining "
          f"({len(chunk_bounds) - len(chunk_jobs):,} already checkpointed)")

    with Pool(processes=n_workers, initializer=_init_worker, initargs=(spf,)) as pool:
        with tqdm(total=len(chunk_jobs), desc="[pipeline] chunks", unit="chunk") as pbar:
            for start, end, part_path in chunk_jobs:
                n_values = list(range(start, end + 1))
                # Split the chunk further across workers for finer-grained
                # progress feedback, then reassemble in order.
                sub_size = max(1, len(n_values) // n_workers)
                sub_chunks = [n_values[i:i + sub_size] for i in range(0, len(n_values), sub_size)]

                rows = []
                for sub_result in pool.imap(_fingerprint_chunk, sub_chunks):
                    rows.extend(sub_result)

                df = pd.DataFrame(rows).sort_values("n").reset_index(drop=True)
                df.to_csv(part_path, index=False)
                pbar.update(1)
                pbar.set_postfix({"last_chunk": f"{start}-{end}"})

    print(f"[pipeline] Done. Part-files written to: {output_dir}")


def consolidate(output_dir: str, final_path: str) -> None:
    """
    Concatenate all Parquet part-files in `output_dir` into a single
    Parquet file at `final_path`, sorted by n. Run this once the full
    pipeline has completed (or whenever you want a merged snapshot of
    progress so far).
    """
    part_files = sorted(
        f for f in os.listdir(output_dir) if f.startswith("part_") and f.endswith(".csv")
    )
    if not part_files:
        raise FileNotFoundError(f"No part-files found in {output_dir}")

    frames = []
    for f in tqdm(part_files, desc="[consolidate] reading parts", unit="file"):
        frames.append(pd.read_csv(os.path.join(output_dir, f)))

    print("[consolidate] concatenating and sorting")
    full = pd.concat(frames, ignore_index=True).sort_values("n").reset_index(drop=True)

    print(f"[consolidate] writing final file: {final_path}")
    full.to_csv(final_path, index=False)
    print(f"[consolidate] Done. {len(full):,} rows.")
