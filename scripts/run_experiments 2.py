"""Run the NewsSum V2 experiments from the command line.

Examples:
    python scripts/run_experiments.py --data "BBC News Summary" --out outputs --cpu-only
    python scripts/run_experiments.py --data "BBC News Summary" --out outputs --hours 8
    python scripts/run_experiments.py --out outputs --smoke --hours 1
"""

import argparse
from pathlib import Path

from newssum import report
from newssum.pipeline import add_bertscore, prepare_data, run_baselines
from newssum.runner import launch, plan, status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="folder containing 'News Articles' and 'Summaries'")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--hours", type=float, default=11.0, help="time budget for GPU jobs")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--models", nargs="*", default=None, help="subset of model keys (see models.py)")
    ap.add_argument("--smoke", action="store_true", help="tiny subset, one epoch")
    ap.add_argument("--cpu-only", action="store_true", help="audit and baselines only")
    ap.add_argument("--no-bertscore", action="store_true")
    a = ap.parse_args()

    out = Path(a.out)
    if not (out / "data" / "meta.json").exists():
        prepare_data(out, a.data)
    if not (out / "runs" / "oracle__main__s0" / "result.json").exists():
        print(run_baselines(out, smoke=a.smoke))
    if not a.cpu_only:
        jobs = plan("smoke" if a.smoke else "full", seeds=a.seeds, models=a.models)
        print(launch(out, jobs, hours=a.hours, smoke=a.smoke).to_string())
        if not a.no_bertscore:
            add_bertscore(out)
    tables = report.write_all(out)
    if tables:
        print(tables["main"].to_string())


if __name__ == "__main__":
    main()
