import pandas as pd
import pytest

from kitchenrank.config import DATA_FILE
from kitchenrank.data import load_inspections, split_blocks


def test_vendored_table_loads_and_is_clean():
    df = load_inspections(DATA_FILE)
    assert len(df) == 27600
    assert df["inspection_id"].is_unique
    assert df["date"].is_monotonic_increasing
    assert set(df["any_critical"].unique()) == {0, 1}
    # the critical count must equal the sum of the first fourteen violation codes
    crit = df[[f"v{c:02d}" for c in range(1, 15)]].sum(axis=1)
    assert (crit == df["critical"]).all()


def test_blocks_are_disjoint_and_ordered():
    df = load_inspections(DATA_FILE)
    b = split_blocks(df)
    assert not set(b["train"]) & set(b["valid"]) and not set(b["valid"]) & set(b["test"])
    assert df.loc[b["train"], "date"].max() < df.loc[b["valid"], "date"].min()
    assert df.loc[b["valid"], "date"].max() < df.loc[b["test"], "date"].min()
    assert len(b["train"]) + len(b["valid"]) + len(b["test"]) == len(df)


def test_missing_column_is_reported(tmp_path):
    p = tmp_path / "x.csv"
    pd.DataFrame({"inspection_id": [1], "date": ["2012-01-01"]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        load_inspections(p)


def test_duplicate_ids_rejected(tmp_path, frame):
    p = tmp_path / "dup.csv"
    d = frame.head(5).copy()
    d["inspection_id"] = 1
    d.to_csv(p, index=False)
    with pytest.raises(ValueError, match="unique"):
        load_inspections(p)


def test_empty_block_is_an_error(frame):
    with pytest.raises(ValueError, match="empty"):
        split_blocks(frame, train_end="2001-01-01", valid_end="2002-01-01")
