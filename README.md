# RMATS Experiment Code
## Recursive Multi-Agent Trading System — IEEE Access

---

## 项目结构

```
rmats_experiment/
├── run_all.py                  ← 一键运行全流程
├── requirements.txt            ← 依赖包
├── data/
│   └── download_data.py        ← Step 1: 下载数据
├── baselines/
│   └── run_baselines.py        ← Step 2: 4个baseline策略
├── agents/
│   └── run_rmats.py            ← Step 3: RMATS完整系统
├── evaluation/
│   └── evaluate.py             ← Step 4: 评估 + 生成表格/图
└── results/                    ← 自动生成
    ├── baseline_portfolio_values.csv
    ├── rmats_portfolio_value.csv
    ├── results_summary.json    ← 论文数据来源
    └── figures/
        ├── fig1_cumulative_returns.png
        └── fig2_drawdown.png
```

---

## 快速开始

### 1. 安装依赖
```bash
cd rmats_experiment
pip install -r requirements.txt
```

### 2. 一键运行
```bash
python run_all.py
```

预计运行时间（CPU）：约 30 分钟

### 3. 或分步运行
```bash
python data/download_data.py        # ~3 min
python baselines/run_baselines.py   # ~15 min
python agents/run_rmats.py          # ~10 min
python evaluation/evaluate.py       # ~2 min
```

---

## 实验设置

| 参数 | 值 |
|------|----|
| 数据来源 | yfinance（免费） |
| 资产范围 | 24个ETF（美股/国际/债券/大宗商品） |
| 训练集 | 2016-01-01 ~ 2020-12-31 |
| 验证集 | 2021-01-01 ~ 2022-12-31 |
| 测试集 | 2023-01-01 ~ 2024-12-31 |
| 再平衡频率 | 每21个交易日（月度） |
| 交易成本 | 10bps per trade |

---

## 策略说明

### Baseline 1: MVO
- Markowitz均值方差优化
- 最大单资产权重30%，完全做多约束

### Baseline 2: DQN
- 3动作离散DQN（equal-weight / momentum-tilt / defensive）
- 在训练集上训练200个episode
- 状态：动量、波动率、Sharpe、均值回归信号

### Baseline 3: FinBERT Sentiment (Proxy)
- 用市场衍生信号代理FinBERT情绪（CPU友好）
- 恐惧指标（VIX代理）+ 防御/周期轮动信号

### Baseline 4: Multi-Factor Quant
- 12月动量（跳过最近1月） + 低波动 + 均值回归 + 滚动Sharpe
- Z-score标准化后等权组合

### RMATS
- Manager Agent + Sentiment + Report + Analysis + Risk 五个Agent
- HMM三状态市场机制识别（牛市/熊市/压力）
- 递归协调，最多8轮收敛
- CVaR风险评估 + 熔断机制

---

## 输出文件说明

`results/results_summary.json` 包含：
- `table_1_performance`: 所有策略的完整指标（直接用于论文Table I）
- `table_2_event_drawdowns`: 地缘政治事件回撤数据（论文Table II）

`results/figures/` 包含：
- `fig1_cumulative_returns.png`: 累积收益曲线对比（论文Figure 1）
- `fig2_drawdown.png`: 回撤对比图（论文Figure 2）

---

## 将真实结果替换进论文

1. 运行完毕后打开 `results/results_summary.json`
2. 用其中的真实数字替换 `.docx` 论文 Section VI 中的数据表
3. 用 `results/figures/` 中的图替换论文中的示例数据

---

## 注意事项

- yfinance数据有时不稳定，如下载失败请重试 `python data/download_data.py`
- XLRE历史数据从2015年才有，如缺失会自动跳过
- DQN训练有随机性，每次结果略有不同（正常现象）
- 所有结果基于真实市场数据，可用于IEEE投稿
