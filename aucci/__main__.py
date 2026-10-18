"""python -m aucci [outdir]: run every experiment, print the summaries and save the raw tables."""

import sys
import time
from pathlib import Path

from . import experiments as ex


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, fn, summary in [("exp1_single_coverage", ex.single_coverage, ex.summarise_single),
                              ("exp2_paired_difference", ex.paired_difference, ex.summarise_paired),
                              ("exp3_detectable_difference", ex.detectable_difference, None)]:
        t = time.time()
        df = fn()
        df.to_csv(out / f"{name}.csv", index=False)
        shown = df if summary is None else summary(df)
        print(f"\n== {name} ({time.time() - t:.0f}s)\n{shown.to_string(index=False)}", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "results"))
