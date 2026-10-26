"""The data is the file we think it is, and the checks catch it when it is not."""

import pandas as pd
import pytest

from loan_targeting.data import (
    DATA_PATH,
    DROPPED,
    EXPECTED_SHA256,
    MODEL_INPUTS,
    TARGET,
    DataCheckError,
    load_raw,
    model_inputs,
    sha256_of,
    validate,
)


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return load_raw()


def test_csv_fingerprint_matches():
    assert sha256_of(DATA_PATH) == EXPECTED_SHA256


def test_loaded_data_has_expected_shape(df):
    assert len(df) == 5000
    assert int(df[TARGET].sum()) == 480


def test_changed_file_is_rejected(tmp_path):
    copy = tmp_path / "changed.csv"
    copy.write_bytes(DATA_PATH.read_bytes() + b"\n")  # one extra byte
    with pytest.raises(DataCheckError, match="not the expected file"):
        load_raw(copy)


def test_non_binary_flag_is_rejected(df):
    broken = df.copy()
    broken.loc[0, "Online"] = 2
    with pytest.raises(DataCheckError, match="Online should be 0/1"):
        validate(broken)


def test_model_inputs_drop_three_columns(df):
    X, y = model_inputs(df)
    assert list(X.columns) == MODEL_INPUTS
    assert len(MODEL_INPUTS) == 10
    assert not set(DROPPED) & set(X.columns)
    assert y.name == TARGET and len(y) == len(X)


def test_unassigned_column_is_rejected(df):
    extra = df.assign(NewColumn=0)
    with pytest.raises(DataCheckError, match="NewColumn"):
        model_inputs(extra)
