# 跨资产 ETF v0.6.3-data.2 数据层挑战者

该模块在冻结的 `etf_shadow_v063` 之上增加双收益口径和晋级证据门禁，不修改原模型参数、v0.4.1 champion 或 v0.6.2 基线。它始终处于研究锁：不连接券商、不生成订单、不自动晋级。

## 解决的问题

- **执行收益**：双源前复权 ETF 收益，用于跟踪误差、换手约束和可交易回测。
- **经济收益**：官方总收益指数、官方复权净值或基于官方分配记录重构的收益，用于形成候选组合。
- **本地 A 股篮子**：可在受控环境按真实成分构建；当前快照只能在生效日之后向前使用，禁止回填。生效后任一共同交易日缺失都会 `FAILED_CLOSED`。
- **QDII 分解**：要求市场收益与“境外标的本币收益 × 汇率 × 溢价 × 费率/跟踪残差”逐日乘法守恒，并与执行面板逐日一致。
- **独立验证**：晋级模式固定要求 `skfolio==0.20.1`；`vectorbt` 是可选第三引擎。缺失或版本不符会明确写入证据，绝不冒充已运行。

## 数据等级

- `Grade B`：只有公开双源前复权执行代理。允许纯影子研究，禁止晋级。
- `Grade A`：官方经济收益、覆盖完整回测期的本地点时历史篮子、带冻结来源 manifest 的 QDII 分解和固定版本 skfolio 均通过。即使 Grade A 也只代表“可提交人工晋级审查”，不会自动替换模型。

## 验证

```bash
PYTHONPATH=../etf_shadow_v063:. python -m unittest discover -s tests -v
python run_shadow_v0_6_3_data2.py --demo --output-root demo_runs
```

基础影子运行明确允许执行收益代理充当经济收益的降级口径，但会标记 `EXPLICIT_EXECUTION_PROXY_FALLBACK` 和 Grade B：

```bash
python run_shadow_v0_6_3_data2.py \
  --execution-returns-csv /secure/production_data/returns.csv \
  --execution-manifest /secure/production_data/data_manifest.json \
  --current-weights /secure/current_shadow_weights.csv \
  --benchmark-weights /secure/benchmark_weights.csv \
  --as-of 2026-07-17 \
  --profile shadow \
  --output-root /secure/runs_v063_data2
```

经济面板、本地篮子和 QDII 分解均为显式可选输入；QDII 必须同时提供 CSV 和来源 manifest。一旦传入就必须通过完整数据身份、原始来源哈希、as-of、覆盖和守恒校验。`--profile promotion` 会把所有 Grade A 证据和 skfolio 固定版本设为硬门禁。

本地篮子构建器的输入格式：

- 成分价格：`date,asset,close`
- 本地映射：`component_asset,model_asset,weight,effective_from`

```bash
python build_local_proxy.py \
  --component-prices-csv /secure/components.csv \
  --mapping-csv /secure/local_mapping.csv \
  --as-of 2026-07-17 \
  --output-dir /secure/local_proxy
```

映射和生成物必须留在本地或受控存储。公开仓库不包含真实成分、权重、持仓或运行产物。

完整变更边界见 [PRE_REGISTRATION.md](PRE_REGISTRATION.md)。
