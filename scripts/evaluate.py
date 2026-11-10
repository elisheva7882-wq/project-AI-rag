"""Explicit evaluation entrypoint; never used by the interactive bot."""

from __future__ import annotations

import argparse
import sys

from config import ConfigurationError, load_settings
from src.evaluation import EvaluationError, run_evaluation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate original questions without indexing the answer key. Semantic review remains manual.")
    parser.add_argument("--mode", choices=("answers", "retrieval"), default="answers")
    parser.add_argument("--offline", action="store_true", help="Explicit isolated offline retrieval; does not measure semantic quality.")
    parser.add_argument("--ids", nargs="+", help="Unique IDs such as Q01 Q18; omitted means all 45.")
    parser.add_argument("--output-dir", help="A separate directory without an existing report.")
    parser.add_argument("--interval-seconds", type=float, default=6.0)
    parser.add_argument("--max-consecutive-errors", type=int, default=2)
    args = parser.parse_args(argv)
    try:
        report = run_evaluation(load_settings(), mode=args.mode, offline=args.offline, question_ids=args.ids,
                                output_dir=args.output_dir, request_interval_seconds=args.interval_seconds,
                                max_consecutive_errors=args.max_consecutive_errors, progress=lambda text: print(text, flush=True))
    except (ConfigurationError, EvaluationError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"Mode: {report['mode']}; structural diagnostics only; no automatic semantic grade.")
    print(f"Completed: {report['summary']['completed']}; errors: {report['summary']['errors']}; not run: {report['summary']['not_run']}.")
    print(f"JSON: {report['report_path']}\nReview: {report['markdown_path']}")
    return 2 if any(report["summary"][key] for key in ("errors", "not_run", "structural_failures")) else 0


if __name__ == "__main__":
    raise SystemExit(main())
