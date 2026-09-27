import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import baseline_suite.eval_pmc as base
from baseline_suite.fullscope_data import all_datasets

base.all_datasets = all_datasets

if __name__ == "__main__":
    base.main()
