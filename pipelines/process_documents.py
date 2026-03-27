from __future__ import annotations

import argparse

from app.logging import get_logger
from pipelines.text_processing import process_company_documents


logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Process stored raw company documents into cleaned chunks.")
    parser.add_argument("--ticker", default="NVDA", help="Ticker to process. MVP currently supports NVDA.")
    args = parser.parse_args()
    summary = process_company_documents(args.ticker)
    logger.info("Document processing complete: %s", summary)


if __name__ == "__main__":
    main()
