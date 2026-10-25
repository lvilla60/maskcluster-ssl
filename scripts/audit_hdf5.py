"""Run the read-only training/validation HDF5 audit."""

import argparse
import sys
from pathlib import Path

from maskcluster_ssl.data.audit import audit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="Input/raw")
    parser.add_argument("--output-dir", default="outputs/audit")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    try:
        run, result, critical = audit(args.data_dir, args.output_dir, root, args.batch_size, args.seed)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Audit failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(f"Report: {run.relative_to(root)}")
    print(f"Critical: {result['issue_counts'].get('critical', 0)}; warnings: {result['issue_counts'].get('warning', 0)}; "
          f"valid validation queries: {result['valid_queries']}")
    return 1 if critical else 0


if __name__ == "__main__":
    sys.exit(main())
