"""Load the Thera Bank loan data and check it matches what the analysis assumes.

This module only loads and validates. It does not clean or drop anything:
cleaning is decided after the EDA (see eda.py).
"""

import hashlib
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "Bank_Personal_Loan_Modelling.csv"

# SHA-256 of the committed CSV (LF line endings, pinned by .gitattributes).
EXPECTED_SHA256 = "513c2b9963fadd84afc019df62607e5210073239f45a838bb31c2c087900b5b5"
EXPECTED_ROWS = 5000
EXPECTED_POSITIVES = 480

TARGET = "Personal Loan"
EXPECTED_COLUMNS = [
    "ID",
    "Age",
    "Experience",
    "Income",
    "ZIP Code",
    "Family",
    "CCAvg",
    "Education",
    "Mortgage",
    "Personal Loan",
    "Securities Account",
    "CD Account",
    "Online",
    "CreditCard",
]
BINARY_COLUMNS = ["Personal Loan", "Securities Account", "CD Account", "Online", "CreditCard"]


class DataCheckError(ValueError):
    """The data does not match an assumption the analysis depends on."""


def sha256_of(path: Path) -> str:
    """Return the SHA-256 fingerprint of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(df: pd.DataFrame) -> None:
    """Raise DataCheckError if the data breaks any assumption the analysis relies on."""
    if list(df.columns) != EXPECTED_COLUMNS:
        raise DataCheckError(f"Columns differ from expected.\n  got:      {list(df.columns)}\n  expected: {EXPECTED_COLUMNS}")
    if len(df) != EXPECTED_ROWS:
        raise DataCheckError(f"Expected {EXPECTED_ROWS} rows, got {len(df)}.")
    if df.isna().any().any():
        missing = df.isna().sum()
        raise DataCheckError(f"Unexpected missing values:\n{missing[missing > 0]}")
    if not df["ID"].is_unique:
        raise DataCheckError("ID is not unique: some customers appear twice.")
    for col in BINARY_COLUMNS:
        values = {int(v) for v in df[col].unique()}
        if not values <= {0, 1}:
            raise DataCheckError(f"{col} should be 0/1, found {sorted(values)}.")
    positives = int(df[TARGET].sum())
    if positives != EXPECTED_POSITIVES:
        raise DataCheckError(f"Expected {EXPECTED_POSITIVES} accepted offers, got {positives}.")


def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the CSV after checking its fingerprint, then validate its contents."""
    actual = sha256_of(path)
    if actual != EXPECTED_SHA256:
        raise DataCheckError(
            f"{path.name} is not the expected file.\n  got:      {actual}\n  expected: {EXPECTED_SHA256}\n"
            "If you are on Windows, check that .gitattributes is present and re-check out the CSV."
        )
    df = pd.read_csv(path)
    validate(df)
    return df
