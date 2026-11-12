"""Paths, split dates, seeds and the cost assumptions used by the economics module."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data" / "inspections.csv.gz"
OUT_DIR = ROOT / "outputs"
TABLE_DIR = OUT_DIR / "tables"
ASSET_DIR = ROOT / "assets"

SEED = 7

# Forward-in-time split. Nothing from a later block is ever used to fit an earlier one.
TRAIN_END = "2013-06-30"
VALID_END = "2013-12-31"  # test is everything after this date (calendar 2014)

# Neighbourhood features look at inspections this close and this recent.
NEIGHBOUR_RADIUS_M = 600.0
NEIGHBOUR_WINDOW_DAYS = 180
NEIGHBOUR_PRIOR_STRENGTH = 8.0  # pseudo-visits that pull a sparse neighbourhood toward the city rate

# The first 14 codes on the Chicago inspection form are the critical ones.
CRITICAL_CODES = tuple(range(1, 15))

# Planning horizons and delay caps used by the scheduling study, in days.
HORIZONS = (7, 14, 28)
MAX_DELAYS = (0, 3, 7, 14, 28)
HEADLINE_HORIZON = 28
HEADLINE_MAX_DELAY = 7

N_BOOT = 400


@dataclass(frozen=True)
class Costs:
    """Illustrative value of removing one critical violation for one day.

    These are not measured. They are stated so a reader can replace them with the figure their own
    department trusts, and the study reports the measured number of exposure days next to every dollar total.
    """

    per_exposure_day: tuple[float, ...] = (0.5, 2.0, 5.0)
    visit_cost: float = 150.0  # assumed loaded cost of one inspection visit (inspector time, travel, admin)


COSTS = Costs()
