"""
RMATS Experiment — One-Click Runner
=====================================
Runs the complete experiment pipeline:
  Step 1: Download data
  Step 2: Train & run baselines
  Step 3: Train & run RMATS
  Step 4: Evaluate & generate tables/charts

Usage:
  python run_all.py

Expected runtime on CPU:
  Step 1: ~3 min  (network download)
  Step 2: ~15 min (DQN training)
  Step 3: ~10 min (HMM training + backtest)
  Step 4: ~2 min  (evaluation)
  Total:  ~30 min
"""

import subprocess
import sys
import os
import time

STEPS = [
    ("Step 1: Data Download",   "data/download_data.py"),
    ("Step 2: Baselines",       "baselines/run_baselines.py"),
    ("Step 3: RMATS System",    "agents/run_rmats.py"),
    ("Step 4: Evaluation",      "evaluation/evaluate.py"),
]

def check_packages():
    """Check all required packages are installed."""
    required = [
        "yfinance", "pandas", "numpy", "scipy", "sklearn",
        "matplotlib", "seaborn", "torch", "stable_baselines3",
        "cvxpy", "hmmlearn",
    ]
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"❌ Missing packages: {missing}")
        print(f"\nInstall with:\n  pip install -r requirements.txt")
        return False
    print("✅ All packages available")
    return True


def run_step(name, script):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, script],
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n❌ {name} FAILED after {elapsed:.0f}s")
        print("   Check error above and re-run this step manually.")
        sys.exit(1)
    print(f"\n✅ {name} completed in {elapsed:.0f}s")


def main():
    print("╔══════════════════════════════════════════════════════╗")
    print("║     RMATS — Complete Experiment Pipeline             ║")
    print("║     For: IEEE Access Submission                      ║")
    print("╚══════════════════════════════════════════════════════╝")

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    if not check_packages():
        sys.exit(1)

    total_start = time.time()
    for name, script in STEPS:
        run_step(name, script)

    total = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  🎉 All steps complete in {total/60:.1f} minutes!")
    print(f"  📁 Results:  results/")
    print(f"  📊 Figures:  results/figures/")
    print(f"  📄 Summary:  results/results_summary.json")
    print(f"\n  Next: Update paper Section VI with real numbers from")
    print(f"        results/results_summary.json")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
