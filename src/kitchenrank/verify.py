"""Compare two metrics.json files and fail if the headline numbers moved.

python -m kitchenrank.verify committed.json fresh.json
"""

from __future__ import annotations

import json
import sys

CHECKS = [
    ("test auc", ("test", "auc", "value"), 0.01),
    ("recall in top 20%", ("test", "recall_top20", "value"), 0.02),
    ("days earlier per critical finding", ("schedule", "headline", "days_saved_per_critical"), 0.15),
    ("validation-chosen model", ("selection", "chosen"), None),
]


def dig(d: dict, path: tuple[str, ...]):
    for k in path:
        d = d[k]
    return d


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) != 2:
        print(__doc__)
        return 2
    a, b = (json.load(open(p)) for p in argv)
    failed = False
    for name, path, tol in CHECKS:
        x, y = dig(a, path), dig(b, path)
        same = x == y if tol is None else abs(x - y) <= tol
        print(f"{'ok  ' if same else 'DIFF'} {name}: {x} vs {y}")
        failed |= not same
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
