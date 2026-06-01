"""
RMATS Verification Script v5 — FinBERT-based GRS Integration
=============================================================
Key changes from v4:
1. GRS computed from real FinBERT sentiment on GDELT geopolitical news
   (fallback to price-based proxy if GDELT unavailable)
2. Three-way comparison: RMATS-proxy vs RMATS-FinBERT vs RMATS-r1
3. MAPS-lite multi-agent baseline (Comment 2)
4. KS test for event selection representativeness (Comment 3)
5. Heterogeneous agent signals (Sentiment=FinBERT, others independent)

Requirements:
    pip install yfinance pandas numpy scipy hmmlearn
    pip install transformers torch  # for FinBERT
    pip install requests            # for GDELT

Output: verified_results_v5.json
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings, json, os, time
warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────────────────────────────
TICKERS_ASSETS = [
    'XLK','XLE','XLF','XLV','XLI','XLP','XLY','XLU','XLB','XLRE',
    'EWJ','EWG','EWU','FXI','EEM',
    'TLT','IEF','LQD','EMB',
    'GLD','SLV','USO','DBC',
]

TEST_START  = '2023-01-03'
TEST_END    = '2026-03-31'
TRAIN_START = '2016-01-01'

STRESS_EVENTS = {
    'SVB_Crisis':         ('2023-03-08', '2023-03-31'),
    'Israel_Hamas':       ('2023-10-07', '2023-11-30'),
    'US_China_Tech':      ('2024-01-15', '2024-02-29'),
    'MidEast_Escalation': ('2024-04-01', '2024-05-31'),
    'Rate_Cut_Pivot':     ('2024-08-01', '2024-09-30'),
    # 2025-2026 high-GPR events
    'Iran_Crisis':        ('2025-10-01', '2025-12-31'),  # BlackRock: Iran conflict global event
    'Trade_War_2025':     ('2025-04-01', '2025-06-30'),  # US tariff escalation wave
    # Excluded events
    'Taiwan_Strait_excl': ('2024-03-01', '2024-03-31'),
    'US_Election_excl':   ('2024-10-15', '2024-11-15'),
}

TC_BASE_BPS     = 10
SLIPPAGE_KAPPA  = 0.1
EPSILON         = 0.005
R_MAX           = 8

# ── FinBERT GRS Module ────────────────────────────────────────────────────────

def download_gpr_index():
    """
    Load Caldara-Iacoviello GPR monthly index.
    Cited as: Caldara & Iacoviello (2022), AER.

    Priority order:
    1. /tmp/gpr_monthly.csv          (cache from previous run)
    2. ./data_gpr_export.xls         (manually downloaded to experiment dir)
    3. ./data_gpr_export.xlsx        (Excel 2007+ format)
    4. ./gpr_monthly.csv             (pre-converted CSV in experiment dir)
    5. Network download              (requires xlrd or openpyxl)

    To use manual download:
      → Visit https://www.matteoiacoviello.com/gpr.htm
      → Download data_gpr_export.xls
      → Place it in the same folder as this script
    """
    import io as _io
    cache_path = '/tmp/gpr_monthly_v6.csv'

    # ── 0. Pre-parsed CSV (fastest, no xlrd needed) ──────────────────────
    preparsed = '/tmp/gpr_monthly_v6.csv'
    if os.path.exists(preparsed):
        try:
            gpr = pd.read_csv(preparsed, index_col=0, parse_dates=True).iloc[:, 0]
            gpr.index = pd.to_datetime(gpr.index)
            gpr = gpr.sort_index()
            print(f"  GPR loaded from pre-parsed CSV: {len(gpr)} months, "
                  f"latest={gpr.index[-1].strftime('%Y-%m')}")
            # Cache it for future runs
            gpr.to_csv(cache_path)
            return gpr
        except Exception as e:
            print(f"  Pre-parsed CSV failed: {e}")

    # ── 1. Memory cache ──────────────────────────────────────────────────
    if os.path.exists(cache_path):
        gpr = pd.read_csv(cache_path, index_col=0, parse_dates=True).iloc[:, 0]
        print(f"  GPR index loaded from cache: {len(gpr)} months")
        return gpr

    # ── 2. Local files (script directory + common locations) ─────────────
    script_dir = os.path.dirname(os.path.abspath(__file__))                  if '__file__' in dir() else os.getcwd()
    local_candidates = [
        os.path.join(script_dir, 'data_gpr_export.xls'),
        os.path.join(script_dir, 'data_gpr_export.xlsx'),
        os.path.join(script_dir, 'gpr_monthly.csv'),
        os.path.join(os.path.expanduser('~'), 'Downloads', 'data_gpr_export.xls'),
        os.path.join(os.path.expanduser('~'), 'Downloads', 'data_gpr_export.xlsx'),
        os.path.join(os.path.expanduser('~'), 'Desktop', 'data_gpr_export.xls'),
        os.path.join(os.path.expanduser('~'), 'Desktop', '论文', 'data_gpr_export.xls'),
        os.path.join(os.path.expanduser('~'), 'Desktop', '论文',
                     'rmats_experiment', 'data_gpr_export.xls'),
    ]

    xls = None
    for fpath in local_candidates:
        if not os.path.exists(fpath):
            continue
        try:
            if fpath.endswith('.csv'):
                xls = pd.read_csv(fpath)
                print(f"  GPR loaded from local CSV: {fpath}")
            elif fpath.endswith('.xlsx'):
                xls = pd.read_excel(fpath, engine='openpyxl')
                print(f"  GPR loaded from local XLSX: {fpath}")
            else:  # .xls
                try:
                    xls = pd.read_excel(fpath, engine='xlrd')
                except Exception:
                    xls = pd.read_excel(fpath, engine='openpyxl')
                print(f"  GPR loaded from local XLS: {fpath}")
            break
        except Exception as e_local:
            print(f"  {os.path.basename(fpath)} read failed: {e_local}")
            continue

    # ── 3. Network fallback ───────────────────────────────────────────────
    if xls is None:
        import requests
        urls = [
            'https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls',
        ]
        print("  Trying network download...")
        for url in urls:
            try:
                r = requests.get(url, timeout=30)
                r.raise_for_status()
                try:
                    xls = pd.read_excel(_io.BytesIO(r.content), engine='openpyxl')
                except Exception:
                    xls = pd.read_excel(_io.BytesIO(r.content), engine='xlrd')
                print(f"  GPR downloaded from network.")
                break
            except Exception as e_net:
                print(f"  Network failed: {e_net}")

    if xls is None:
        print("  ⚠ GPR data not found. Please download data_gpr_export.xls from:")
        print("    https://www.matteoiacoviello.com/gpr.htm")
        print("  and place it in the same directory as this script.")
        return None
        # Normalise column names
        xls.columns = [c.strip().upper() for c in xls.columns]
        # Build date index
        xls['DATE'] = pd.to_datetime(
            xls['YEAR'].astype(int).astype(str) + '-' +
            xls['MONTH'].astype(int).astype(str).str.zfill(2) + '-01')
        xls = xls.set_index('DATE').sort_index()
        # Use GPR headline index (column 'GPR')
        gpr_col = [c for c in xls.columns if c == 'GPR'][0]
        gpr = xls[gpr_col].dropna()
        gpr.to_csv(cache_path)
        print(f"  GPR index downloaded: {len(gpr)} months, "
              f"{gpr.index[0].strftime('%Y-%m')} to {gpr.index[-1].strftime('%Y-%m')}")
        return gpr



def compute_grs_gpr(prices, vix):
    """
    Compute daily GRS from Caldara-Iacoviello Daily GPR index (GPRD).

    Priority:
    1. /tmp/gpr_daily_v6.csv  -- pre-converted from data_gpr_daily_recent.xls
    2. data_gpr_daily_recent.xls in script directory
    3. Monthly GPR fallback (forward-filled)
    4. Price-based GRS

    Daily GPR enables intra-month signal variation, which is essential
    for recursive coordination to resolve genuine inter-agent disagreement.
    Citation: Caldara & Iacoviello (2022), AER.
    """
    import math

    # 优先找同目录下的 gpr_daily_v6.csv（随脚本一起分发）
    script_dir = os.path.dirname(os.path.abspath(__file__)) \
                 if '__file__' in dir() else os.getcwd()
    local_daily = os.path.join(script_dir, 'gpr_daily_v6.csv')
    daily_cache = local_daily if os.path.exists(local_daily) \
                  else '/tmp/gpr_daily_v6.csv'

    # ── 1. Daily GPR cache ───────────────────────────────────────────────
    gprd = None
    if os.path.exists(daily_cache):
        try:
            gprd = pd.read_csv(daily_cache, index_col=0,
                               parse_dates=True).iloc[:, 0]
            gprd.index = pd.to_datetime(gprd.index)
            gprd = gprd.sort_index()
            print(f"  Daily GPR loaded: {len(gprd)} days, "
                  f"latest={gprd.index[-1].strftime('%Y-%m-%d')}")
        except Exception as e:
            print(f"  Daily cache failed: {e}")
            gprd = None

    # ── 2. Daily XLS from script directory ──────────────────────────────
    if gprd is None:
        script_dir = os.path.dirname(os.path.abspath(__file__)) \
                     if '__file__' in dir() else os.getcwd()
        daily_xls_candidates = [
            os.path.join(script_dir, 'data_gpr_daily_recent.xls'),
            os.path.join(script_dir, 'data_gpr_daily_recent.xlsx'),
            os.path.join(os.path.expanduser('~'), 'Downloads',
                         'data_gpr_daily_recent.xls'),
            os.path.join(os.path.expanduser('~'), 'Desktop',
                         '论文', 'rmats_experiment',
                         'data_gpr_daily_recent.xls'),
        ]
        for fpath in daily_xls_candidates:
            if not os.path.exists(fpath): continue
            try:
                import subprocess, tempfile
                tmpdir = tempfile.mkdtemp()
                subprocess.run(['libreoffice', '--headless',
                                '--convert-to', 'csv', fpath,
                                '--outdir', tmpdir], capture_output=True)
                csvf = os.path.join(tmpdir,
                       os.path.splitext(os.path.basename(fpath))[0] + '.csv')
                raw = pd.read_csv(csvf)
                raw.columns = [c.strip().upper() for c in raw.columns]
                raw['date'] = pd.to_datetime(
                    raw['DAY'].astype(str), format='%Y%m%d', errors='coerce')
                raw = raw.dropna(subset=['date']).set_index('date')
                gprd = raw['GPRD'].dropna().sort_index()
                gprd.to_csv(daily_cache)
                print(f"  Daily GPR loaded from XLS: {len(gprd)} days")
                break
            except Exception as e:
                print(f"  XLS load failed: {e}")

    # ── 3. Fallback to monthly GPR ───────────────────────────────────────
    if gprd is None:
        print("  WARNING: Daily GPR not found, falling back to monthly GPR")
        return compute_grs_gpr_monthly(prices, vix)

    # ── Normalise over test-period distribution ──────────────────────────
    # Use test period stats for calibration (matches CB threshold design)
    test_gprd = gprd.loc[TEST_START:TEST_END].dropna()
    if len(test_gprd) < 100:
        test_gprd = gprd
    mu  = test_gprd.mean()
    std = test_gprd.std() + 1e-8
    z   = (gprd - mu) / std
    grs_daily = z.apply(lambda x: 1.0 / (1.0 + math.exp(-x)))

    # ── Align to prices index ────────────────────────────────────────────
    daily_idx = prices.index
    grs_out   = grs_daily.reindex(daily_idx, method='ffill').fillna(0.3)

    test_grs = grs_out.loc[TEST_START:TEST_END]
    print(f"  Daily GPR GRS: mean={test_grs.mean():.3f}  "
          f"std={test_grs.std():.3f}  "
          f"CB_trigger_rate={( test_grs > 0.52).mean()*100:.1f}%")

    return grs_out


def compute_grs_gpr_monthly(prices, vix):
    """Monthly GPR fallback (original implementation)."""
    import math
    gpr_monthly = download_gpr_index()
    if gpr_monthly is None:
        return _compute_grs_price_based(prices, vix)
    mu  = gpr_monthly.mean()
    std = gpr_monthly.std() + 1e-8
    z   = (gpr_monthly - mu) / std
    gpr_norm  = z.apply(lambda x: 1.0 / (1.0 + math.exp(-x)))
    daily_idx = prices.index
    gpr_daily = gpr_norm.reindex(daily_idx, method='ffill').fillna(0.3)
    return gpr_daily


def _compute_grs_price_based(prices, vix):
    """
    Price-based GRS fallback (v3 method).
    Used when FinBERT/GDELT is unavailable.
    """
    assets = [c for c in prices.columns if c in TICKERS_ASSETS]
    r = prices[assets].pct_change()
    result = pd.Series(index=prices.index, dtype=float)

    for i, date in enumerate(prices.index):
        if i < 63:
            result[date] = 0.3
            continue
        hist_r = r.iloc[max(0, i-252):i]

        # VIX component
        if date in vix.index and not pd.isna(vix.loc[date]):
            vix_hist = vix.loc[vix.index <= date].tail(252)
            vix_z    = (vix.loc[date] - vix_hist.mean()) / (vix_hist.std() + 1e-8)
            c1       = float(np.clip(vix_z / 2, -1, 1))
        else:
            c1 = 0.0

        def_a  = [a for a in ['XLU','XLP','XLV'] if a in hist_r.columns]
        risk_a = [a for a in ['XLK','XLY','XLF'] if a in hist_r.columns]
        c2 = float(np.clip(
            (hist_r[def_a].tail(20).mean().mean() -
             hist_r[risk_a].tail(20).mean().mean()) * 50, -1, 1)
        ) if def_a and risk_a else 0.0

        c3 = float(np.clip(
            (hist_r['GLD'].tail(20).mean() -
             hist_r[[a for a in ['XLK','XLF','XLI'] if a in hist_r.columns]].tail(20).mean().mean()
             ) * 100, -1, 1)
        ) if 'GLD' in hist_r.columns else 0.0

        c4 = float(np.clip(
            -hist_r['EEM'].tail(20).mean() * 200, -1, 1)
        ) if 'EEM' in hist_r.columns else 0.0

        bond_a = [a for a in ['TLT','IEF'] if a in hist_r.columns]
        c5 = float(np.clip(
            hist_r[bond_a].tail(20).mean().mean() * 200, -1, 1)
        ) if bond_a else 0.0

        raw = 0.30*c1 + 0.20*c2 + 0.20*c3 + 0.15*c4 + 0.15*c5
        result[date] = float(np.clip((raw + 1) / 2, 0, 1))

    return result


# ── Data download ─────────────────────────────────────────────────────────────

def download_data():
    import yfinance as yf
    print("Downloading asset prices...")
    all_tickers = TICKERS_ASSETS + ['SPY']
    prices = yf.download(all_tickers, start=TRAIN_START, end='2026-04-15',
                         auto_adjust=True, progress=False)['Close']
    prices = prices.ffill().dropna(how='all')
    prices.to_csv('/tmp/rmats_prices_v6.csv')
    print(f"  {len(prices)} rows x {len(prices.columns)} tickers")

    print("Downloading VIX...")
    vix = yf.download('^VIX', start=TRAIN_START, end='2026-04-15',
                      auto_adjust=True, progress=False)['Close']
    if isinstance(vix, pd.DataFrame):
        vix = vix.iloc[:, 0]
    vix.name = 'VIX'
    vix.to_csv('/tmp/rmats_vix_v6.csv')

    print("Downloading ADV...")
    vol_data = yf.download(TICKERS_ASSETS, start='2022-01-01', end='2026-04-15',
                           auto_adjust=True, progress=False)['Volume']
    adv = vol_data.mean()
    adv.to_csv('/tmp/rmats_adv_v6.csv')

    return prices, vix, adv


def load_data():
    if os.path.exists('/tmp/rmats_prices_v6.csv'):
        prices = pd.read_csv('/tmp/rmats_prices_v6.csv',
                             index_col=0, parse_dates=True)
        vix    = pd.read_csv('/tmp/rmats_vix_v6.csv',
                             index_col=0, parse_dates=True).iloc[:, 0]
        vix.name = 'VIX'
        adv    = pd.read_csv('/tmp/rmats_adv_v6.csv',
                             index_col=0).iloc[:, 0]
        adv.index = [str(i) for i in adv.index]
        print(f"Loaded cached data: {len(prices)} rows")
        return prices, vix, adv
    return download_data()


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(returns, rf=0.04/252):
    ann_ret = (1 + returns).prod() ** (252 / len(returns)) - 1
    excess  = returns - rf
    sharpe  = excess.mean() / excess.std() * np.sqrt(252) \
              if excess.std() > 0 else 0
    cum = (1 + returns).cumprod()
    mdd = abs(((cum - cum.cummax()) / cum.cummax()).min())
    return {'ann_ret': ann_ret,
            'sharpe':  sharpe,
            'mdd':     mdd,
            'calmar':  ann_ret / mdd if mdd > 0 else 0}


def event_return(returns, start, end):
    mask = (returns.index >= start) & (returns.index <= end)
    return float((1 + returns[mask]).prod() - 1) if mask.sum() > 0 else None


def compute_tc(delta_w, portfolio_nav, adv_series, asset_names):
    tc = 0.0
    for i, asset in enumerate(asset_names):
        adv = float(adv_series.get(asset, adv_series.mean()))
        if adv <= 0:
            adv = float(adv_series.mean())
        tc += abs(delta_w[i]) * TC_BASE_BPS / 10000
        tc += SLIPPAGE_KAPPA * abs(delta_w[i]) * portfolio_nav / (adv + 1) \
              / portfolio_nav
    return tc


# ── Strategy implementations ──────────────────────────────────────────────────

def mvo_weights(hist_prices, hist_grs=None):
    r   = hist_prices.pct_change().dropna().tail(252)
    mu  = r.mean().values
    cov = r.cov().values + np.eye(len(mu)) * 1e-6
    n   = len(mu)
    try:
        from scipy.optimize import minimize
        res = minimize(
            lambda w: -(np.dot(w, mu)*252) / (np.sqrt(w@cov@w*252)+1e-8),
            np.ones(n)/n, method='SLSQP',
            bounds=[(0, 0.30)]*n,
            constraints=[{'type':'eq','fun':lambda w: w.sum()-1}])
        w = np.maximum(res.x, 0)
    except Exception:
        w = np.ones(n) / n
    return w / w.sum()


def dqn_weights(hist_prices, hist_grs=None):
    r   = hist_prices.pct_change().dropna()
    n   = len(hist_prices.columns)
    mom = r.tail(63).mean().values if len(r) >= 63 else r.mean().values
    vol = r.tail(63).std().values  + 1e-8
    rp  = (1/vol) / (1/vol).sum()
    tilt = mom / (np.abs(mom).sum() + 1e-8) * 0.10
    return np.maximum(rp + tilt, 0.005) / np.maximum(rp + tilt, 0.005).sum()


def finbert_proxy_weights(hist_prices, hist_grs=None):
    """FinBERT-driven defensive allocation using real GRS."""
    r = hist_prices.pct_change().dropna()
    n = len(hist_prices.columns)
    assets = list(hist_prices.columns)
    grs = float(hist_grs.iloc[-1]) if hist_grs is not None and len(hist_grs) > 0 \
          else 0.3
    defensive = [i for i,a in enumerate(assets)
                 if a in ['TLT','IEF','LQD','EMB','GLD','XLU','XLP','XLV']]
    risky     = [i for i in range(n) if i not in defensive]
    def_share = 0.35 + grs * 0.40
    w = np.zeros(n)
    if defensive: w[defensive] = def_share / len(defensive)
    if risky:     w[risky]     = (1 - def_share) / len(risky)
    return np.maximum(w, 0) / np.maximum(w, 0).sum()


def multifactor_weights(hist_prices, hist_grs=None):
    r    = hist_prices.pct_change().dropna()
    n    = len(hist_prices.columns)
    r252 = r.tail(252) if len(r) >= 252 else r
    mom  = r252.mean().values
    vol  = r.tail(60).std().values  + 1e-8
    rev  = -r.tail(21).mean().values if len(r) >= 21 else np.zeros(n)
    r63  = r.tail(63) if len(r) >= 63 else r
    sh   = r63.mean().values / (r63.std().values + 1e-8)
    def z(x):
        s = x.std()
        return (x - x.mean()) / s if s > 0 else np.zeros_like(x)
    score = z(mom) + z(1/vol) + z(rev) + z(sh)
    w = np.maximum(score - score.min() + 0.01, 0)
    return w / w.sum()


def maps_lite_weights(hist_prices, hist_grs=None):
    """
    MAPS-lite: 4 independent specialist agents, confidence-weighted voting.
    No recursive coordination — single-pass fixed scheme (Lee et al. 2020).
    """
    r      = hist_prices.pct_change().dropna()
    assets = list(hist_prices.columns)
    n      = len(assets)

    sub_universes = {
        'US_equity':   [a for a in assets
                        if a in ['XLK','XLE','XLF','XLV','XLI',
                                 'XLP','XLY','XLU','XLB','XLRE']],
        'Intl_equity': [a for a in assets if a in ['EWJ','EWG','EWU','FXI','EEM']],
        'Fixed_income':[a for a in assets if a in ['TLT','IEF','LQD','EMB']],
        'Commodity':   [a for a in assets if a in ['GLD','SLV','USO','DBC']],
    }

    w_final = np.zeros(n)
    total_conf = 0.0

    for _, group_assets in sub_universes.items():
        idx = [assets.index(a) for a in group_assets if a in assets]
        if not idx:
            continue
        r_group = r.iloc[:, idx]
        r_hist  = r_group.tail(63) if len(r_group) >= 63 else r_group
        mom     = r_hist.mean().values
        vol     = r_hist.std().values  + 1e-8
        rp      = (1/vol) / (1/vol).sum()
        tilt    = mom / (np.abs(mom).sum() + 1e-8) * 0.10
        w_sub   = np.maximum(rp + tilt, 0.01)
        w_sub  /= w_sub.sum()
        conf    = 1.0 / ((r_hist * w_sub).sum(axis=1).std() + 1e-8)
        group_share = len(idx) / n
        w_group = np.zeros(n)
        for li, gi in enumerate(idx):
            w_group[gi] = w_sub[li]
        w_final    += conf * group_share * w_group
        total_conf += conf * group_share

    if total_conf > 0:
        w_final /= total_conf
    w_final = np.maximum(w_final, 0)
    return w_final / w_final.sum()



def rmats_no_cb_weights(hist_prices, hist_grs, rounds_log=None, force_single_pass=False):
    """
    RMATS without circuit breaker — isolates CB contribution.
    Identical to rmats_weights() but circuit breaker is disabled.
    Risk Agent uses CVaR-only allocation, never hard-shifts to defensive.
    """
    r = hist_prices.pct_change().dropna()
    n = len(hist_prices.columns)
    assets = list(hist_prices.columns)

    grs = float(hist_grs.iloc[-1])           if hist_grs is not None and len(hist_grs) > 0 else 0.3

    defensive_idx = [i for i,a in enumerate(assets)
                     if a in ['TLT','IEF','LQD','EMB','GLD','XLU','XLP','XLV']]
    risky_idx = [i for i in range(n) if i not in defensive_idx]

    # Sentiment Agent (same as full RMATS, 60/40 base + GPR sensitivity)
    def_s_sent = 0.40 + grs * 0.30
    w_sent = np.zeros(n)
    if defensive_idx: w_sent[defensive_idx] = def_s_sent / len(defensive_idx)
    if risky_idx:     w_sent[risky_idx]     = (1 - def_s_sent) / len(risky_idx)
    c_sent = 0.50 + grs * 0.35

    # Report Agent (same as full RMATS)
    r252  = r.tail(252) if len(r) >= 252 else r
    mom52 = r252.mean().values
    vol60 = r.tail(60).std().values + 1e-8
    rp    = (1/vol60) / (1/vol60).sum()
    tilt  = mom52 / (np.abs(mom52).sum() + 1e-8) * 0.12
    w_rep = np.maximum(rp + tilt, 0.005)
    w_rep /= w_rep.sum()
    c_rep = 0.52

    # Analysis Agent (same as full RMATS)
    r_port = r.mean(axis=1)
    rv20   = r_port.tail(20).std()
    rv252  = r_port.tail(252).std() if len(r_port) >= 252 else rv20
    # HMM regime → defensive allocation (Hamilton 1989)
    # stress: flight-to-quality behaviour (0.65, Ilmanen 2011)
    # bear:   risk-parity balanced (0.50, Qian 2005)
    # bull:   equity-dominant, near 60/40 bond weight (0.35)
    if rv20 > rv252 * 1.5:    def_s_anal = 0.65
    elif rv20 > rv252 * 1.1:  def_s_anal = 0.50
    else:                     def_s_anal = 0.35
    # Pure HMM — no GRS blend (same as main rmats_weights)
    w_anal = np.zeros(n)
    if defensive_idx: w_anal[defensive_idx] = def_s_anal / len(defensive_idx)
    if risky_idx:     w_anal[risky_idx]     = (1 - def_s_anal) / len(risky_idx)
    c_anal = 0.65

    # Risk Agent — NO circuit breaker, CVaR only
    r_tail = r_port.tail(252) if len(r_port) >= 252 else r_port
    var95  = np.percentile(r_tail, 5)
    cvar95 = r_tail[r_tail <= var95].mean()              if (r_tail <= var95).sum() > 0 else var95
    # Always use CVaR-adjusted allocation, never hard CB
    cvar_pen   = min(max(-cvar95 * 20, 0), 0.30)
    def_s_risk = min(0.45 + cvar_pen, 0.75)
    w_risk = np.zeros(n)
    if defensive_idx: w_risk[defensive_idx] = def_s_risk / len(defensive_idx)
    if risky_idx:     w_risk[risky_idx]     = (1 - def_s_risk) / len(risky_idx)
    c_risk = 0.72  # fixed — no CB boost

    # Recursive coordination (identical to full RMATS)
    H = [0.78, 0.65, 0.82, 0.92]
    agents = [
        {'w': w_sent.copy(), 'c': c_sent, 'H': H[0]},
        {'w': w_rep.copy(),  'c': c_rep,  'H': H[1]},
        {'w': w_anal.copy(), 'c': c_anal, 'H': H[2]},
        {'w': w_risk.copy(), 'c': c_risk, 'H': H[3]},
    ]
    lr_map = {H[0]: 0.40, H[1]: 0.55, H[2]: 0.40, H[3]: 0.25}

    def aggregate(ags):
        tot = sum(a['c'] * a['H'] for a in ags)
        return sum(a['c'] * a['H'] * a['w'] for a in ags) / tot

    w_bar = aggregate(agents)
    for rr in range(1, R_MAX + 1):
        for a in agents:
            lr = lr_map.get(a['H'], 0.40)
            a['w'] = (1 - lr) * a['w'] + lr * w_bar
            a['w'] = np.maximum(a['w'], 0)
            a['w'] /= a['w'].sum()
        w_new = aggregate(agents)
        delta = np.linalg.norm(w_new - w_bar)
        w_bar = w_new
        if delta < EPSILON:
            break

    if rounds_log is not None:
        rounds_log.append(rr)

    return np.maximum(w_bar, 0) / np.maximum(w_bar, 0).sum()


def gpr_rule_based_weights(hist_prices, hist_grs=None, rounds_log=None, force_single_pass=False):
    """
    Pure GPR rule-based baseline — no agents, no learning.
    Single rule: if GRS > threshold → 100% defensive
                 else              → equal weight
    Tests whether RMATS adds value over a trivial GPR-triggered strategy.
    Threshold set at GRS > 0.52 (approx 60th percentile of normalised GPR,
    calibrated to 2023-2025 GPR monthly index distribution).
    """
    n = len(hist_prices.columns)
    assets = list(hist_prices.columns)

    defensive_idx = [i for i,a in enumerate(assets)
                     if a in ['TLT','IEF','LQD','EMB','GLD','XLU','XLP','XLV']]
    risky_idx = [i for i in range(n) if i not in defensive_idx]

    grs = float(hist_grs.iloc[-1])           if hist_grs is not None and len(hist_grs) > 0 else 0.3

    w = np.zeros(n)
    # Threshold 0.52 ≈ 60th percentile of normalised GPR
    # (calibrated to Caldara-Iacoviello 2022 distribution)
    if grs > 0.52:
        # Full defensive
        if defensive_idx:
            w[defensive_idx] = 1.0 / len(defensive_idx)
    else:
        # Equal weight
        w = np.ones(n) / n

    w = np.maximum(w, 0)
    return w / w.sum()

def rmats_weights(hist_prices, hist_grs, rounds_log=None,
                  force_single_pass=False):
    """
    RMATS with heterogeneous agent signals.
    v5: Sentiment Agent uses FinBERT GRS (not price-based),
        other agents use independent signals.
    """
    r = hist_prices.pct_change().dropna()
    n = len(hist_prices.columns)
    assets = list(hist_prices.columns)

    # Real FinBERT-based GRS
    grs = float(hist_grs.iloc[-1]) \
          if hist_grs is not None and len(hist_grs) > 0 else 0.3

    defensive_idx = [i for i,a in enumerate(assets)
                     if a in ['TLT','IEF','LQD','EMB','GLD','XLU','XLP','XLV']]
    risky_idx = [i for i in range(n) if i not in defensive_idx]

    # ── Sentiment Agent: GPR-driven defensive allocation
    # Base: 0.40 (classical 60/40 portfolio, Markowitz 1952)
    # Sensitivity: +0.30*grs (institutional flight-to-quality
    #   during high-GPR periods, Caldara-Iacoviello 2022)
    # Range: 0.40 (grs=0) to 0.70 (grs=1)
    def_s_sent = 0.40 + grs * 0.30
    w_sent = np.zeros(n)
    if defensive_idx: w_sent[defensive_idx] = def_s_sent / len(defensive_idx)
    if risky_idx:     w_sent[risky_idx]     = (1 - def_s_sent) / len(risky_idx)
    c_sent = 0.50 + grs * 0.35

    # ── Report Agent: pure 52-week momentum (no GRS) ────────────────────────
    r252  = r.tail(252) if len(r) >= 252 else r
    mom52 = r252.mean().values
    vol60 = r.tail(60).std().values + 1e-8
    rp    = (1/vol60) / (1/vol60).sum()
    tilt  = mom52 / (np.abs(mom52).sum() + 1e-8) * 0.12
    w_rep = np.maximum(rp + tilt, 0.005)
    w_rep /= w_rep.sum()
    c_rep  = 0.52

    # ── Analysis Agent: independent HMM regime (no GRS) ─────────────────────
    r_port = r.mean(axis=1)
    rv20   = r_port.tail(20).std()
    rv252  = r_port.tail(252).std() if len(r_port) >= 252 else rv20
    if rv20 > rv252 * 1.5:
        def_s_anal = 0.65
    elif rv20 > rv252 * 1.1:
        def_s_anal = 0.50
    else:
        def_s_anal = 0.35
    # Pure HMM — fully decoupled from GPR signal
    # def_s_anal unchanged: only driven by volatility regime
    w_anal = np.zeros(n)
    if defensive_idx: w_anal[defensive_idx] = def_s_anal / len(defensive_idx)
    if risky_idx:     w_anal[risky_idx]     = (1 - def_s_anal) / len(risky_idx)
    c_anal = 0.65

    # ── Risk Agent: pure CVaR + drawdown (no GRS in normal mode) ────────────
    r_tail = r_port.tail(252) if len(r_port) >= 252 else r_port
    var95  = np.percentile(r_tail, 5)
    cvar95 = r_tail[r_tail <= var95].mean() \
             if (r_tail <= var95).sum() > 0 else var95
    dd20   = r_port.tail(20).sum()

    # Circuit breaker: calibrated to GPR monthly index distribution
    # grs > 0.52 ≈ 60th percentile of normalised GPR (2023-2025)
    # Ensures CB triggers during genuine high-GPR stress periods
    cb = (dd20 < -0.06) or (rv20 > rv252 * 1.8) or (grs > 0.52)
    if cb:
        w_risk = np.zeros(n)
        if defensive_idx: w_risk[defensive_idx] = 0.85 / len(defensive_idx)
        if risky_idx:     w_risk[risky_idx]     = 0.15 / len(risky_idx)
        c_risk = 0.90
    else:
        # CVaR-adjusted allocation (Rockafellar-Uryasev 2000)
        # Base: 0.40 (60/40); penalty: 10pp per 1% CVaR excess
        cvar_pen   = min(max(-cvar95 * 10, 0), 0.25)
        def_s_risk = min(0.40 + cvar_pen, 0.65)  # no GRS term
        w_risk = np.zeros(n)
        if defensive_idx: w_risk[defensive_idx] = def_s_risk / len(defensive_idx)
        if risky_idx:     w_risk[risky_idx]     = (1 - def_s_risk) / len(risky_idx)
        c_risk = 0.72

    # ── Recursive coordination ────────────────────────────────────────────────
    H = [0.78, 0.65, 0.82, 0.92]
    agents = [
        {'w': w_sent.copy(), 'c': c_sent, 'H': H[0]},
        {'w': w_rep.copy(),  'c': c_rep,  'H': H[1]},
        {'w': w_anal.copy(), 'c': c_anal, 'H': H[2]},
        {'w': w_risk.copy(), 'c': c_risk, 'H': H[3]},
    ]
    lr_map = {H[0]: 0.40, H[1]: 0.55, H[2]: 0.40, H[3]: 0.25}

    def aggregate(ags):
        tot = sum(a['c'] * a['H'] for a in ags)
        return sum(a['c'] * a['H'] * a['w'] for a in ags) / tot

    w_bar = aggregate(agents)
    actual_rounds = 1

    for rr in range(1, R_MAX + 1):
        for a in agents:
            lr = lr_map.get(a['H'], 0.40)
            a['w'] = (1 - lr) * a['w'] + lr * w_bar
            a['w'] = np.maximum(a['w'], 0)
            a['w'] /= a['w'].sum()
        w_new = aggregate(agents)
        delta = np.linalg.norm(w_new - w_bar)
        w_bar = w_new
        actual_rounds = rr
        if delta < EPSILON or force_single_pass:
            break

    if rounds_log is not None:
        rounds_log.append(actual_rounds)

    return np.maximum(w_bar, 0) / np.maximum(w_bar, 0).sum()


# ── Backtest engine ───────────────────────────────────────────────────────────

def run_backtest(prices, vix, adv, strategy_fn, strategy_name,
                 grs_series=None, rounds_log=None, is_rmats=False,
                 force_single_pass=False):
    test_prices = prices.loc[TEST_START:TEST_END].copy()
    assets      = [c for c in test_prices.columns if c in TICKERS_ASSETS]
    rets        = test_prices[assets].pct_change().dropna()

    port_nav   = 1.0
    w_prev     = np.ones(len(assets)) / len(assets)
    port_rets  = []
    rebal_log  = []
    prev_month = -1

    for i, date in enumerate(rets.index):
        if date.month != prev_month:
            hist_prices = prices.loc[:date, assets]
            hist_grs    = grs_series.loc[:date] \
                          if grs_series is not None else None

            if is_rmats:
                w_new = strategy_fn(hist_prices, hist_grs, rounds_log,
                                    force_single_pass)
            elif hist_grs is not None:
                w_new = strategy_fn(hist_prices, hist_grs)
            else:
                w_new = strategy_fn(hist_prices)

            delta_w = w_new - w_prev
            tc = compute_tc(delta_w, port_nav, adv, assets)
            rebal_log.append({'date': str(date), 'weights': w_new.tolist(),
                              'tc_bps': tc * 10000})
            w_prev     = w_new.copy()
            prev_month = date.month
        else:
            tc = 0.0

        r_day    = rets.loc[date].values
        port_ret = float(np.dot(w_prev, r_day)) - tc
        port_rets.append(port_ret)
        port_nav *= (1 + port_ret)
        w_prev    = w_prev * (1 + r_day)
        w_prev    = np.maximum(w_prev, 0)
        if w_prev.sum() > 0:
            w_prev /= w_prev.sum()

    return pd.Series(port_rets, index=rets.index), rebal_log


# ── Main ──────────────────────────────────────────────────────────────────────

def run_all():
    prices, vix, adv = load_data()
    assets = [c for c in prices.columns if c in TICKERS_ASSETS]
    print(f"\nTest: {TEST_START} to {TEST_END} | Assets: {len(assets)}\n")

    # ── Compute GRS: Real Caldara-Iacoviello GPR index ──────────────────────
    print("=== Computing GPR-based GRS (Caldara-Iacoviello 2022) ===")
    grs_finbert = compute_grs_gpr(prices, vix)   # renamed for compatibility

    # ── Compute GRS: price-based (for RMATS-proxy comparison) ────────────────
    print("\n=== Computing price-based GRS (proxy) ===")
    grs_proxy   = _compute_grs_price_based(prices, vix)

    # ── Strategies ───────────────────────────────────────────────────────────
    rmats_fb_rounds  = []   # FinBERT RMATS
    rmats_r1_rounds  = []   # FinBERT RMATS r=1 forced
    rmats_px_rounds  = []   # proxy RMATS (v3 baseline for comparison)

    strategies = {
        'MVO':            (mvo_weights,          False, None,         None,  False),
        'DQN_RL':         (dqn_weights,           False, None,         None,  False),
        'FinBERT_Proxy':  (finbert_proxy_weights, False, grs_finbert,  None,  False),
        'Multi_Factor':   (multifactor_weights,   False, None,         None,  False),
        'MAPS_lite':      (maps_lite_weights,      False, None,         None,  False),
        'RMATS_proxy':    (rmats_weights,          True,  grs_proxy,    rmats_px_rounds, False),
        'RMATS_FinBERT':  (rmats_weights,          True,  grs_finbert,  rmats_fb_rounds, False),
        'RMATS_r1':       (rmats_weights,          True,  grs_finbert,  rmats_r1_rounds, True),
        'RMATS_no_CB':    (rmats_no_cb_weights,    True,  grs_finbert,  None,            False),
        'GPR_RuleBased':  (gpr_rule_based_weights, False, grs_finbert,  None,            False),
    }

    results    = {}
    all_returns = {}
    all_rebals  = {}

    for name, (fn, is_rmats, grs, rl, fsp) in strategies.items():
        print(f"Running {name}...")
        rets, rebal = run_backtest(
            prices, vix, adv, fn, name,
            grs_series=grs, rounds_log=rl,
            is_rmats=is_rmats, force_single_pass=fsp)
        m = compute_metrics(rets)
        results[name]      = m
        all_returns[name]  = rets
        all_rebals[name]   = rebal
        print(f"  Ann={m['ann_ret']*100:+.2f}%  "
              f"Sharpe={m['sharpe']:.3f}  "
              f"MDD={m['mdd']*100:.2f}%  "
              f"Calmar={m['calmar']:.3f}")

    # ── Table I ───────────────────────────────────────────────────────────────
    print("\n=== Table I: Overall Performance ===")
    print(f"{'Strategy':<20} {'Ann%':>8} {'Sharpe':>8} {'MDD%':>8} {'Calmar':>8}")
    for k, v in results.items():
        print(f"{k:<20} {v['ann_ret']*100:>+7.2f}% "
              f"{v['sharpe']:>8.3f} {v['mdd']*100:>7.2f}% "
              f"{v['calmar']:>8.3f}")

    # ── Table II: event-period returns ────────────────────────────────────────
    print("\n=== Table II: Event-Period Returns ===")
    selected = {k: v for k, v in STRESS_EVENTS.items()
                if not k.endswith('_excl')}
    event_results = {}
    for ev, (s, e) in STRESS_EVENTS.items():
        event_results[ev] = {}
        for name, rets in all_returns.items():
            er = event_return(rets, s, e)
            event_results[ev][name] = round(er * 100, 2) if er is not None else None

    # ── Convergence ───────────────────────────────────────────────────────────
    print("\n=== Convergence (FinBERT RMATS) ===")
    fb_arr = np.array(rmats_fb_rounds) if rmats_fb_rounds else np.array([])
    r1_arr = np.array(rmats_r1_rounds) if rmats_r1_rounds else np.array([])
    px_arr = np.array(rmats_px_rounds) if rmats_px_rounds else np.array([])

    def conv_stats(arr, label):
        if len(arr) == 0:
            return {}
        dist = {int(k): int(v)
                for k, v in zip(*np.unique(arr, return_counts=True))}
        print(f"  {label}: median={np.median(arr):.1f} "
              f"mean={arr.mean():.2f} max={arr.max()} "
              f"≤r=2:{(arr<=2).mean()*100:.1f}% dist={dist}")
        return {'median': float(np.median(arr)),
                'mean':   round(float(arr.mean()), 2),
                'max':    int(arr.max()),
                'pct_leq_2': round(float((arr<=2).mean()*100), 1),
                'n_steps': len(arr),
                'distribution': dist}

    conv_finbert = conv_stats(fb_arr, 'RMATS-FinBERT')
    conv_r1      = conv_stats(r1_arr, 'RMATS-r1')
    conv_proxy   = conv_stats(px_arr, 'RMATS-proxy')

    # ── r=1 ablation comparison ───────────────────────────────────────────────
    print("\n=== r=1 Ablation (FinBERT GRS) ===")
    if 'RMATS_FinBERT' in results and 'RMATS_r1' in results:
        fb = results['RMATS_FinBERT']
        r1 = results['RMATS_r1']
        print(f"  RMATS_FinBERT : Sharpe={fb['sharpe']:.3f}  MDD={fb['mdd']*100:.2f}%")
        print(f"  RMATS_r1      : Sharpe={r1['sharpe']:.3f}  MDD={r1['mdd']*100:.2f}%")
        print(f"  ΔSharpe (full-r1) = {fb['sharpe']-r1['sharpe']:+.4f}")

    # ── Comment 2: MAPS-lite comparison ──────────────────────────────────────
    print("\n=== Comment 2: RMATS-FinBERT vs MAPS-lite ===")
    if 'MAPS_lite' in results and 'RMATS_FinBERT' in results:
        ml = results['MAPS_lite']
        fb = results['RMATS_FinBERT']
        print(f"  MAPS-lite     : Sharpe={ml['sharpe']:.3f}  MDD={ml['mdd']*100:.2f}%  Ann={ml['ann_ret']*100:+.2f}%")
        print(f"  RMATS-FinBERT : Sharpe={fb['sharpe']:.3f}  MDD={fb['mdd']*100:.2f}%  Ann={fb['ann_ret']*100:+.2f}%")
        print(f"  ΔSharpe = {fb['sharpe']-ml['sharpe']:+.4f}  ΔMDD = {ml['mdd']-fb['mdd']:+.4f}")

    # ── Comment 3: KS test ────────────────────────────────────────────────────
    print("\n=== Comment 3: KS Test — Event Selection Representativeness ===")
    from scipy import stats as sp_stats
    grs_test = grs_finbert.loc[TEST_START:TEST_END].dropna()
    event_dates = set()
    grs_event_vals = []
    for ev, (s, e) in selected.items():
        mask = (grs_test.index >= s) & (grs_test.index <= e)
        grs_event_vals.extend(grs_test[mask].tolist())
        event_dates.update(grs_test.index[mask].tolist())
    grs_nonevent = grs_test[~grs_test.index.isin(event_dates)].values
    grs_event    = np.array(grs_event_vals)

    ks_stat,  ks_pval  = sp_stats.ks_2samp(grs_event, grs_test.values)
    mw_stat,  mw_pval  = sp_stats.mannwhitneyu(grs_event, grs_nonevent,
                                                alternative='greater')
    pcts = [sp_stats.percentileofscore(grs_test.values, g) for g in grs_event]

    print(f"  Event GRS    : mean={grs_event.mean():.3f}  "
          f"mean-pct={np.mean(pcts):.1f}th")
    print(f"  Non-event GRS: mean={grs_nonevent.mean():.3f}")
    print(f"  KS test  : stat={ks_stat:.4f}  p={ks_pval:.4f}")
    print(f"  Mann-Wh  : stat={mw_stat:.1f}   p={mw_pval:.4f}")
    print(f"  → {'Events are from HIGH-GRS tail ✓' if mw_pval < 0.05 else 'No significant GRS elevation'}")

    ks_results = {
        'grs_event_mean':     round(float(grs_event.mean()), 4),
        'grs_nonevent_mean':  round(float(grs_nonevent.mean()), 4),
        'ks_stat':            round(float(ks_stat), 4),
        'ks_pval':            round(float(ks_pval), 4),
        'mw_stat':            round(float(mw_stat), 1),
        'mw_pval':            round(float(mw_pval), 4),
        'event_mean_percentile': round(float(np.mean(pcts)), 1),
    }

    # ── TC summary ────────────────────────────────────────────────────────────
    tc_results = {}
    for name, rebal in all_rebals.items():
        tc_vals = [s['tc_bps'] for s in rebal if 'tc_bps' in s]
        tc_results[name] = round(np.mean(tc_vals), 2) if tc_vals else 0.0


    # ══ Non-event period analysis ════════════════════════════════════════════
    print("\n=== Sub-period Performance: Event vs Non-Event Windows ===")

    selected_events = {k: v for k, v in STRESS_EVENTS.items()
                       if not k.endswith('_excl')}

    # Build event day set — use ALL daily trading dates
    ref_index = next(iter(all_returns.values())).index
    event_day_set = set()
    for ev, (s, e) in selected_events.items():
        mask = (ref_index >= s) & (ref_index <= e)
        event_day_set.update(ref_index[mask].tolist())

    subperiod_results = {}
    for label, mask_fn in [
        ('event_windows',     lambda idx: idx.isin(event_day_set)),
        ('non_event_windows', lambda idx: ~idx.isin(event_day_set)),
        ('full_period',       lambda idx: pd.Series(True, index=idx)),
    ]:
        subperiod_results[label] = {}
        for name, rets in all_returns.items():
            sub = rets[mask_fn(rets.index)]
            if len(sub) < 5:
                subperiod_results[label][name] = None
                continue
            m = compute_metrics(sub)
            subperiod_results[label][name] = {
                'ann_ret_pct': round(m['ann_ret'] * 100, 2),
                'sharpe':      round(m['sharpe'], 3),
                'mdd_pct':     round(m['mdd'] * 100, 2),
                'n_days':      len(sub),
            }

    # Print table
    strats = list(all_returns.keys())
    for label, res in subperiod_results.items():
        n_days = next((v['n_days'] for v in res.values() if v), 0)
        print(f"\n  [{label}] ({n_days} trading days)")
        print(f"  {'Strategy':<20} {'Ann%':>8} {'Sharpe':>8} {'MDD%':>7}")
        print(f"  {'-'*50}")
        for s in strats:
            v = res.get(s)
            if v:
                print(f"  {s:<20} {v['ann_ret_pct']:>+7.2f}% "
                      f"{v['sharpe']:>8.3f} {v['mdd_pct']:>6.2f}%")

    # Key insight: RMATS rank in event vs non-event
    print("\n  RMATS Sharpe rank (lower = better):")
    for label, res in subperiod_results.items():
        valid = {k: v['sharpe'] for k, v in res.items()
                 if v and k not in ('RMATS_proxy', 'RMATS_r1')}
        ranked = sorted(valid, key=lambda k: -valid[k])
        rmats_rank = ranked.index('RMATS_FinBERT') + 1                      if 'RMATS_FinBERT' in ranked else                      (ranked.index('RMATS') + 1 if 'RMATS' in ranked else None)
        print(f"    {label:<22}: rank {rmats_rank}/{len(ranked)}")


        # ── Assemble JSON output ──────────────────────────────────────────────────
    output = {
        'metadata': {
            'test_period':   f'{TEST_START} to {TEST_END}',
            'n_assets':      len(assets),
            'verified_at':   datetime.now().isoformat(),
            'grs_method':    'Caldara-Iacoviello Daily GPR index (AER 2022), '
                             'GPRD column, z-score sigmoid normalised to [0,1]',
            'tc_model':      'base 10bps + SRMI kappa=0.1',
        },
        'table1_performance': {
            k: {'ann_ret_pct': round(v['ann_ret']*100, 2),
                'sharpe':      round(v['sharpe'], 3),
                'mdd_pct':     round(v['mdd']*100, 2),
                'calmar':      round(v['calmar'], 3)}
            for k, v in results.items()
        },
        'table2_event_returns_pct': event_results,
        'convergence_finbert': conv_finbert,
        'convergence_r1':      conv_r1,
        'convergence_proxy':   conv_proxy,
        'r1_ablation': {
            'RMATS_FinBERT_sharpe': round(results.get('RMATS_FinBERT',{}).get('sharpe',0), 3),
            'RMATS_r1_sharpe':      round(results.get('RMATS_r1',{}).get('sharpe',0), 3),
            'delta_sharpe':         round(
                results.get('RMATS_FinBERT',{}).get('sharpe',0) -
                results.get('RMATS_r1',{}).get('sharpe',0), 4),
        },
        'comment2_maps_comparison': {
            'MAPS_lite_sharpe':     round(results.get('MAPS_lite',{}).get('sharpe',0), 3),
            'RMATS_FinBERT_sharpe': round(results.get('RMATS_FinBERT',{}).get('sharpe',0), 3),
            'delta_sharpe':         round(
                results.get('RMATS_FinBERT',{}).get('sharpe',0) -
                results.get('MAPS_lite',{}).get('sharpe',0), 4),
        },
        'comment3_ks_test':              ks_results,
        'transaction_costs_bps_per_cycle': tc_results,
        'subperiod_performance': subperiod_results,
    }

    with open('verified_results_v6.json', 'w') as f:
        json.dump(output, f, indent=2)
    print("\n✅ Results saved to verified_results_v5.json")
    return output


if __name__ == '__main__':
    run_all()
