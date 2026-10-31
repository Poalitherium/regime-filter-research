"""Audit generated results without building presentation or report artifacts."""
import json
import pandas as pd
from backtest.resplit_common import OUT
from backtest.resplit_report import audit


def main():
    result = audit(pd.read_csv(OUT/'summary.csv'), pd.read_csv(OUT/'comparisons.csv'))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
