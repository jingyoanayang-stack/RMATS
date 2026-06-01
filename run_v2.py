"""
RMATS v2 — One-click runner
Usage: python run_v2.py

Runs only the new v2 components (assumes data already downloaded).
  Step 1: agents/run_rmats_v2.py    (~10 min)
  Step 2: evaluation/convergence_analysis.py  (~1 min)
"""
import subprocess, sys, os, time

BASE = os.path.dirname(os.path.abspath(__file__))

def run(name, script):
    print(f"\n{'='*55}\n  {name}\n{'='*55}")
    t0 = time.time()
    r  = subprocess.run([sys.executable, script], cwd=BASE)
    if r.returncode != 0:
        print(f"\n❌ {name} failed"); sys.exit(1)
    print(f"✅ {name} done in {time.time()-t0:.0f}s")

run("RMATS v2 Backtest",     "agents/run_rmats_v2.py")
run("Convergence Analysis",  "evaluation/convergence_analysis.py")

print("""
╔══════════════════════════════════════════════════════╗
║  v2 Complete — new outputs in results/               ║
║                                                      ║
║  rmats_v2_portfolio_value.csv  — backtest returns    ║
║  rmats_v2_trace.csv            — full round log      ║
║  convergence_stats.json        — Section VI.D data   ║
║  example_trace_table.csv       — Table IV for paper  ║
║  figures/fig3_convergence_curve.png                  ║
║  figures/fig4_rounds_histogram.png                   ║
║  figures/fig5_stress_vs_normal.png                   ║
╚══════════════════════════════════════════════════════╝
""")
