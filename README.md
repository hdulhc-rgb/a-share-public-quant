# a-share-public-quant

公开、可复核、默认失败关闭的量化研究仓库。当前包含两条彼此隔离的影子研究线：

- `models/etf_shadow_v063`：跨资产 ETF 固定挑战者矩阵、walk-forward / CPCV、双回测一致性、约束诊断和完整证据链。
- `models/meituan_options_v12`：从单一股票售卖模型抽离的公开安全期权证据层，只验证报价与反事实，不触发 Covered Call 或交易。

## 安全边界

仓库不包含也不接收真实持仓、账户余额、成本价、税务信息、未归属股份、家庭资产负债、券商导出、私人研报或运行时行情文件。所有示例均为确定性的合成数据。

两个模块都遵守以下硬约束：

- 研究与执行隔离；无券商连接、无订单载荷、无自动交易。
- as-of 数据边界与严格下一可交易时点，禁止未来函数。
- 失败关闭；缺失证据用 `NOT_EVALUABLE`，不以 0 或旧缓存替代。
- 模型变更先预注册、后回测；报告稳定区，不按单次峰值收益自动挑选或晋级。
- 所有 PR 必须通过隐私守卫和单元测试。

## 快速开始

```bash
python -m pip install -r requirements.txt
python scripts/privacy_guard.py
(cd models/etf_shadow_v063 && python -m unittest discover -s tests -v)
(cd models/meituan_options_v12 && python -m unittest discover -s tests -v)
```

演示命令只验证软件契约，不构成历史业绩或投资建议：

```bash
(cd models/etf_shadow_v063 && python run_shadow_v0_6_3.py --demo --output-root demo_runs)
(cd models/meituan_options_v12 && python run_meituan_options_v12.py --demo --output-dir demo_run)
```

协作规则见 [CONTRIBUTING.md](CONTRIBUTING.md)，数据边界见 [docs/data-boundary.md](docs/data-boundary.md)，模型治理见 [docs/model-governance.md](docs/model-governance.md)。
