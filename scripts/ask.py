"""Ask the configured Gemini-backed archive service from PowerShell."""

from __future__ import annotations

import argparse
import sys

from config import ConfigurationError, load_settings
from src.models import QuestionError
from src.qa_service import QAService


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ask KingdomAI using only the validated archive and the configured Gemini models.")
    parser.add_argument("--question", required=True, help="Original question; no query expansion is used.")
    args = parser.parse_args(argv)
    try:
        answer = QAService(load_settings()).ask(args.question)
    except (ConfigurationError, QuestionError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(answer.text)
    if answer.sources:
        print("\nמקורות מהארכיון:")
        for source in answer.sources:
            print(f"{source.document_id} — {source.title}; {source.section}; {source.document_type}; {source.period}")
            print(f"ציטוט: {source.quote}")
    return 2 if answer.status == "error" else 0


if __name__ == "__main__":
    raise SystemExit(main())
