# 单一股票期权证据验证器 v1.2（公开安全版）

这是从美团七因子售卖模型中抽离出的只读、配置驱动证据组件。公开仓只包含通用验证逻辑和合成测试，不包含公司专属售卖区间、真实持仓、成本、未归属股份、个人税务或家庭财务信息。

组件校验来源时间、报价新鲜度、Bid/Mid/Ask、无套利边界、OI、成交量、价差、IV 区间和模型残差，并生成事件快照、结构化拒绝原因和反事实 Covered Call 影子表。

硬边界：

- 期权证据永远不能自动触发 Covered Call；
- 期权证据不能改写调用方传入的价格锚点；
- 风险覆盖层不能绕过调用方传入的保护地板；
- 不连接券商、不生成订单；
- 生产 HKEX 美式期权必须显式使用 QuantLib；缺依赖写 `DEPENDENCY_MISSING`，不静默改用欧式模型。

## 验证

```bash
python -m unittest discover -s tests -v
python run_meituan_options_v12.py --demo --output-dir demo_run
```

真实行情按 `options_chain_template.csv` 填写，所有时点必须带时区。`observed_at` 是源观测时刻，`available_at` 是模型首次可用时刻，`quote_time` 是报价时刻。生产的 `exercise_style` 与 `pricing_model` 必须明确填写，不做默认猜测。

生产运行还必须提供一个不会提交到仓库的 `--policy-json`：

```json
{
  "price": 100.0,
  "protection_floor": 90.0,
  "hard_risk_count": 0,
  "base_sale_units": 1000,
  "risk_overlay_multiplier": 1.0,
  "lot_size": 100,
  "anchors": [
    {"price": 90.0, "reference_units": 0},
    {"price": 110.0, "reference_units": 1000}
  ]
}
```

以上数字均为合成示例，不代表任何真实组合。真实策略配置、行情原文件和运行产物必须留在本地或受控存储中。
