"""
patch_convergence.py
=====================
调整收敛参数：
  epsilon : 0.005 -> 0.008  (更宽松的收敛判定)
  r_max   : 10   -> 8       (减少振荡轮次)
  r_min   : 2    -> 2       (保持不变)

同时修复 Risk Agent 在后期轮次的振荡问题：
  - circuit breaker 只在 r=1 完整评估，后续轮次使用上一轮结果
  - 避免 CB 反复触发/解除导致 delta 在 epsilon 附近震荡

运行: python patch_convergence.py
"""
import re, os, subprocess, sys

BASE = os.path.dirname(os.path.abspath(__file__))

# ── 1. 修改 protocol.py 的超参数 ──────────────────────────────────────────────
proto_path = os.path.join(BASE, "agents/protocol.py")
with open(proto_path) as f: src = f.read()

src = src.replace("EPSILON    = 0.005", "EPSILON    = 0.008")
src = src.replace("R_MAX      = 10",    "R_MAX      = 8")

with open(proto_path, "w") as f: f.write(src)
print("✅ protocol.py: epsilon=0.008, r_max=8")

# ── 2. 修改 run_rmats_v2.py 的 RecursiveProtocol 初始化 ────────────────────────
ag_path = os.path.join(BASE, "agents/run_rmats_v2.py")
with open(ag_path) as f: src = f.read()

src = src.replace(
    "epsilon=0.005, r_min=2, r_max=10",
    "epsilon=0.008, r_min=2, r_max=8"
)

# ── 3. 修复 Risk Agent 振荡：CB 状态在 r>1 时锁定 ──────────────────────────────
# 增加一个 _cb_locked 状态，r>1 时不重新评估 CB
old_cb_block = """        cb = (self._sigma["drawdown"] > theta_dd or
              broadcast.global_geo_risk > theta_geo or
              recent_vol > theta_vol)

        if cb:"""

new_cb_block = """        # Lock CB decision after r=1 to prevent inter-round oscillation
        if round_num == 1:
            cb = (self._sigma["drawdown"] > theta_dd or
                  broadcast.global_geo_risk > theta_geo or
                  recent_vol > theta_vol)
            self._cb_locked = cb
        else:
            cb = self._cb_locked   # hold CB state across rounds within a step

        if cb:"""

src = src.replace(old_cb_block, new_cb_block)

# 在 __init__ 里加 _cb_locked
src = src.replace(
    "        self.portfolio_peak  = 1.0\n        self.portfolio_value = 1.0",
    "        self.portfolio_peak  = 1.0\n        self.portfolio_value = 1.0\n        self._cb_locked = False"
)

with open(ag_path, "w") as f: f.write(src)
print("✅ run_rmats_v2.py: epsilon=0.008, r_max=8, CB lock applied")

# ── 4. 运行 v2 ──────────────────────────────────────────────────────────────
print("\n🚀 Running patched v2...\n")
r = subprocess.run([sys.executable, "run_v2.py"], cwd=BASE)
sys.exit(r.returncode)
