# 跨资产 ETF 影子研究引擎 v0.6.3

这是 v0.6.2 / v6 的向前兼容挑战者研究包，不是交易系统。它不连接券商、不读取或改写真实持仓、不生成订单，也不会自动把挑战者晋级为 champion。公开版本不包含任何真实持仓、家庭配置或权重文件；演示权重完全等权且为合成数据。

## 这版新增

- 固定挑战者：1/N、逆波动、HRP、CVaR 风险预算、最小 CDaR、收缩均值–风险。
- 逆波动对零/近零波动资产使用预先登记的 0.5% 年化波动率下限，并把被处理资产写入筛选轨迹；这不是静默回退。
- 主验证：anchored walk-forward；次验证：purged / embargoed CPCV。
- 约束诊断：跟踪误差、单边换手、约束是否饱和、原始与约束后目标距离、影子约束成本代理。
- 双重回测：向量化引擎与逐日事件循环必须在容差内一致；VectorBT 只作为可选第三引擎，不存在静默回退。
- 数据指纹、预注册参数预算、稳定区而非峰值收益、结构化拒绝原因和追加式 manifest。

## 快速验证

```bash
python -m unittest discover -s tests -v
python run_shadow_v0_6_3.py --demo --output-root demo_runs
```

真实输入为宽表日收益 CSV，第一列必须叫 `date`，其余列为资产；生产数据必须是可复核的总收益/后复权口径，并只包含 as-of 当时可见信息。

```bash
python run_shadow_v0_6_3.py \
  --returns-csv returns.csv \
  --current-weights current_shadow_weights.csv \
  --benchmark-weights benchmark_weights.csv \
  --as-of 2026-07-17 \
  --output-root runs_v063
```

权重文件格式为 `asset,weight`。若真实数据、时间边界、权重、约束守恒、双重回测或必需产物任一失败，程序返回非零并进入 `FAILED_CLOSED`。

生产运行必须显式提供两份权重文件；程序没有真实组合默认值。请只在本地或受控环境保存这些文件，禁止提交到公开仓库。

## 可选开源集成

`skfolio` 和 `vectorbt` 是研究增强项。默认 `--optional-engines record` 会把依赖状态写入证据，不会冒充已经运行；只有显式使用 `--optional-engines require` 才把缺失依赖视为失败。核心固定挑战者与双重回测可在基础依赖下独立复现。
