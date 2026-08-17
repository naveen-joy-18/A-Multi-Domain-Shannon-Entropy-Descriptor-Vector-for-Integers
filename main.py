"""
main.py
=======
CLI entry point. This is the front door -- the person running the heist
overnight should not need to open a single other file.

Usage
-----
    python -m ueft.main --N 1000000 --output ./ueft_output --chunk-size 10000
    python -m ueft.main --N 1000000 --output ./ueft_output --consolidate-only --final ./ueft_fingerprints.parquet

Notes
-----
- Safe to Ctrl+C and re-run: completed chunks are checkpointed and skipped.
- `--consolidate-only` merges existing part-files without recomputation --
  useful for pulling a snapshot of progress mid-run without stopping it.
"""

from __future__ import annotations
import argparse
from .pipeline import run_pipeline, consolidate


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Universal Entropy Fingerprint Theory (UEFT) pipeline"
    )
    parser.add_argument("--N", type=int, default=1_000_000,
                         help="Upper bound (inclusive) of integers to process")
    parser.add_argument("--output", type=str, default="./ueft_output",
                         help="Directory for chunked Parquet part-files")
    parser.add_argument("--chunk-size", type=int, default=10_000,
                         help="Integers per chunk / per part-file")
    parser.add_argument("--n-workers", type=int, default=None,
                         help="Number of worker processes (default: cpu_count - 1)")
    parser.add_argument("--consolidate-only", action="store_true",
                         help="Skip computation; just merge existing part-files")
    parser.add_argument("--final", type=str, default="./ueft_fingerprints.csv",
                         help="Path for the final consolidated CSV file")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if not args.consolidate_only:
        run_pipeline(
            N=args.N,
            output_dir=args.output,
            chunk_size=args.chunk_size,
            n_workers=args.n_workers,
        )

    consolidate(args.output, args.final)


if __name__ == "__main__":
    main()
