"""Project-wide settings. Override paths and the SEC contact through environment variables."""
import os
from pathlib import Path

ROOT = Path(os.environ.get("MLM_ROOT", Path(__file__).resolve().parents[2]))
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"

# SEC asks every automated client to identify itself with a name and contact email,
# e.g. SEC_USER_AGENT="Jane Doe research jane@example.com". Requests without one may be blocked.
SEC_USER_AGENT = os.environ.get("SEC_USER_AGENT", "margin-loan-monitor research")
SEC_MAX_RPS = 8  # SEC fair-access limit is 10 requests per second

# Filings window and search terms for pledge disclosures
EDGAR_START = os.environ.get("EDGAR_START", "2024-01-01")
EDGAR_END = os.environ.get("EDGAR_END", "2025-12-31")
EDGAR_PHRASES = [
    '"pledged as collateral"', '"have been pledged"', '"shares are pledged"', '"margin loan"',
    '"pledge agreement"', '"pledged to secure"', '"are pledged"', '"has pledged"', '"have pledged"',
]
EDGAR_FORMS = {
    "proxy": "DEF 14A",
    "13d": "SC 13D,SC 13D/A,SCHEDULE 13D,SCHEDULE 13D/A",  # form codes changed in Dec 2024
}

# Spark / GraphFrames
GRAPHFRAMES_PACKAGE = "graphframes:graphframes:0.8.4-spark3.5-s_2.12"
SPARK_PACKAGES_REPO = "https://repos.spark-packages.org"


def ensure_dirs() -> None:
    for d in (RAW, PROCESSED):
        d.mkdir(parents=True, exist_ok=True)
