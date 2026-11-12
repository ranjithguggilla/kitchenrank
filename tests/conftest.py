import numpy as np
import pandas as pd
import pytest

from kitchenrank.features import code_columns


def make_frame(n_lic: int = 300, visits: int = 5, seed: int = 0) -> pd.DataFrame:
    """A small inspection table with a real signal: earlier critical findings make another one likelier."""
    rng = np.random.default_rng(seed)
    rows = []
    iid = 1
    for lic in range(1, n_lic + 1):
        base = rng.uniform(0.03, 0.25)
        lat, lon = 41.85 + rng.normal(0, 0.04), -87.65 + rng.normal(0, 0.04)
        day = pd.Timestamp("2011-09-05") + pd.Timedelta(days=int(rng.integers(0, 120)))
        prev = 0
        for _ in range(visits):
            crit = int(rng.random() < min(0.95, base + 0.45 * prev))
            row = {
                "inspection_id": iid,
                "license": lic,
                "date": day,
                "facility": "Restaurant" if lic % 3 else "Other",
                "risk": float(1 + lic % 3),
                "lat": lat,
                "lon": lon,
                "result": "Fail" if crit and rng.random() < 0.6 else "Pass",
                "critical": crit,
                "serious": int(rng.integers(0, 2)),
                "minor": int(rng.integers(0, 5)),
                "inspector": ["blue", "green", "orange"][lic % 3],
                "temperatureMax": float(rng.uniform(20, 90)),
                "precipIntensity": 0.0,
                "humidity": 0.6,
                "windSpeed": 8.0,
                "city_first_record": 0,
                "city_past_fail": 0,
                "business_age": float(rng.uniform(0, 12)),
                "heat_burglary": float(rng.uniform(0, 50)),
                "heat_garbage": float(rng.uniform(0, 50)),
                "heat_sanitation": float(rng.uniform(0, 50)),
                "lic_tavern": int(rng.random() < 0.1),
            }
            for c in range(1, 45):
                row[f"v{c:02d}"] = int(crit and c == 3) if c <= 14 else int(rng.random() < 0.2)
            rows.append(row)
            iid += 1
            prev = crit
            day = day + pd.Timedelta(days=int(rng.integers(60, 300)))
    df = pd.DataFrame(rows).sort_values(["date", "inspection_id"]).reset_index(drop=True)
    df["any_critical"] = (df["critical"] > 0).astype(int)
    df["fail_flag"] = (df["result"] == "Fail").astype(int)
    df["week"] = df["date"].dt.to_period("W-SUN").dt.start_time
    return df


@pytest.fixture(scope="session")
def frame() -> pd.DataFrame:
    return make_frame()


@pytest.fixture(scope="session")
def codes(frame):
    return code_columns(frame)


@pytest.fixture(scope="session")
def dense(frame) -> pd.DataFrame:
    """Same rows, grouped into quarters so every 'week' holds enough inspections to rank."""
    d = frame.copy()
    d["week"] = d["date"].dt.to_period("Q").dt.start_time
    return d
