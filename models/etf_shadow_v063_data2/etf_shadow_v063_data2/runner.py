from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from etf_shadow_v063.backtest import differential_check, metrics, stability_score, vectorized_returns
from etf_shadow_v063.challengers import CHALLENGERS, INVERSE_VOLATILITY_FLOOR_ANNUAL, build_challenger
from etf_shadow_v063.core import (
    ResearchClosed,
    ResearchPolicy,
    annual_tracking_error,
    cap_tracking_error,
    cap_turnover,
    create_run_directory,
    data_fingerprint,
    next_tradable_time,
    one_way_turnover,
    policy_dict,
    sha256_file,
    stable_json_hash,
    utc_now,
    validate_returns,
    write_json,
)
from etf_shadow_v063.validation import anchored_walk_forward, combinatorial_purged_cv

from .data_contract import MODEL_VERSION, return_basis_diagnostics
from .external_validation import run_external_validation


def _aggregate(rows: list[dict[str, object]], challenger: str) -> dict[str, object]:
    subset = pd.DataFrame([row for row in rows if row["challenger"] == challenger and row["status"] == "OK"])
    if subset.empty:
        return {"challenger": challenger, "status": "NOT_EVALUABLE"}
    return {
        "challenger": challenger,
        "status": "OK",
        "closed_splits": int(len(subset)),
        "median_cagr": float(subset["cagr"].median()),
        "worst_max_drawdown": float(subset["max_drawdown"].min()),
        "median_cvar_95_daily": float(subset["cvar_95_daily"].median()),
        "median_cdar_95": float(subset["cdar_95"].median()),
        "median_turnover": float(subset["one_way_turnover"].median()),
        "max_turnover": float(subset["one_way_turnover"].max()),
        "median_tracking_error": float(subset["tracking_error_after"].median()),
        "differential_gate": "PASS" if (subset["differential_status"] == "PASS").all() else "FAIL",
        "promotion": "NO_AUTO_PROMOTION",
    }


def run_research_data2(
    economic_returns: pd.DataFrame,
    execution_returns: pd.DataFrame,
    benchmark: pd.Series,
    current: pd.Series,
    as_of: pd.Timestamp,
    output_root: Path,
    economic_source_path: Path | None,
    execution_source_path: Path | None,
    economic_source_id: str,
    execution_source_id: str,
    policy: ResearchPolicy,
    profile: str,
    data2_assessment: Mapping[str, object],
    economic_source_attestation: Mapping[str, object] | None,
    execution_source_attestation: Mapping[str, object],
    local_proxy_evidence: Mapping[str, object] | None = None,
    qdii_evidence: Mapping[str, object] | None = None,
) -> Path:
    validate_returns(economic_returns, as_of, policy.min_train_observations + policy.test_observations)
    validate_returns(execution_returns, as_of, policy.min_train_observations + policy.test_observations)
    if not economic_returns.index.equals(execution_returns.index):
        raise ResearchClosed("DATA2_DUAL_PANEL_CALENDAR_MISMATCH")
    if list(economic_returns.columns) != list(execution_returns.columns):
        raise ResearchClosed("DATA2_DUAL_PANEL_ASSET_MISMATCH")
    challenger_names = list(CHALLENGERS)
    if len(challenger_names) > policy.parameter_budget:
        raise ResearchClosed("PARAMETER_BUDGET_EXCEEDED")

    timestamp = utc_now()
    run_id = f"v063d2_{as_of.strftime('%Y%m%d')}_{timestamp.strftime('%H%M%S')}_{stable_json_hash({'as_of': as_of.isoformat(), 'n': len(economic_returns), 'economic': float(economic_returns.iloc[-1].sum()), 'execution': float(execution_returns.iloc[-1].sum())})[:8]}"
    run_dir = create_run_directory(output_root, run_id)
    started = timestamp.isoformat()
    economic_fingerprint = data_fingerprint(
        economic_returns,
        economic_source_path,
        as_of,
        economic_source_id,
        source_attestation=economic_source_attestation,
    )
    execution_fingerprint = data_fingerprint(
        execution_returns,
        execution_source_path,
        as_of,
        execution_source_id,
        source_attestation=execution_source_attestation,
    )
    combined_fingerprint = {
        "model_version": MODEL_VERSION,
        "economic": economic_fingerprint,
        "execution": execution_fingerprint,
        "calendar_identical": True,
        "asset_identity_identical": True,
        "data_grade": data2_assessment["data_grade"],
        "promotion_gate_passed": data2_assessment["promotion_gate_passed"],
    }
    write_json(run_dir / "data_fingerprint.json", combined_fingerprint)
    write_json(run_dir / "pre_registration.json", {
        "model_version": MODEL_VERSION,
        "primary_validation": "ANCHORED_WALK_FORWARD",
        "secondary_validation": "CPCV_PURGED_EMBARGOED",
        "challengers": challenger_names,
        "parameter_budget": policy.parameter_budget,
        "selection_rule": "REPORT_STABILITY_REGIONS; NEVER_SELECT_PEAK_RETURN; NO_AUTO_PROMOTION",
        "risk_measures": ["CVaR95", "CDaR95", "MaxDrawdown", "TrackingError", "Turnover"],
        "inverse_volatility_floor_annual": INVERSE_VOLATILITY_FLOOR_ANNUAL,
        "inverse_volatility_floor_policy": "EXPLICIT_FLOOR_FOR_ZERO_OR_NEAR_ZERO_VOLATILITY_ASSETS; RECORD_FLOORED_ASSETS_IN_TRACE",
        "return_basis_policy": "BUILD_ON_ECONOMIC_RETURNS; CONSTRAIN_AND_SCORE_ON_EXECUTION_RETURNS",
        "data_profile": profile,
        "data_grade_policy": "GRADE_B_ALLOWED_ONLY_UNDER_RESEARCH_LOCK; GRADE_A_REQUIRED_FOR_PROMOTION",
    })

    return_basis_diagnostics(economic_returns, execution_returns).to_csv(
        run_dir / "return_basis_diagnostics.csv", index=False
    )
    write_json(run_dir / "data2_gate.json", {
        **dict(data2_assessment),
        "local_proxy": dict(local_proxy_evidence) if local_proxy_evidence is not None else None,
        "qdii_decomposition": dict(qdii_evidence) if qdii_evidence is not None else None,
    })

    wf_splits = anchored_walk_forward(len(economic_returns), policy.min_train_observations, policy.test_observations, policy.walk_forward_step)
    if not wf_splits:
        raise ResearchClosed("NO_WALK_FORWARD_SPLITS")
    rows: list[dict[str, object]] = []
    constraint_rows: list[dict[str, object]] = []
    trace_rows: list[dict[str, object]] = []
    replay_rows: list[dict[str, object]] = []
    weight_history: dict[str, list[pd.Series]] = defaultdict(list)
    latest_targets: dict[str, dict[str, float]] = {}
    combined_snapshot_hash = stable_json_hash(combined_fingerprint)

    for split in wf_splits:
        train_economic = economic_returns.iloc[split.train]
        train_execution = execution_returns.iloc[split.train]
        test_execution = execution_returns.iloc[split.test]
        signal_time = train_economic.index[-1]
        execution_time = next_tradable_time(economic_returns.index, signal_time)
        if execution_time != test_execution.index[0]:
            raise ResearchClosed("WALK_FORWARD_EXECUTION_ALIGNMENT_FAILED")
        covariance = train_execution.cov()
        for name in challenger_names:
            result = build_challenger(name, train_economic)
            trace_rows.append({
                "split_id": split.split_id,
                "candidate": name,
                "stage": "CHALLENGER_BUILD",
                "rule_id": f"BUILD_{name.upper()}",
                "reason_code": "PASS" if result.status == "OK" else result.status,
                "metric": "solver_status",
                "threshold": "OK",
                "observed": result.status,
                "evidence_gap": "" if result.status == "OK" else result.message,
                "source_ref": combined_snapshot_hash,
                "method": result.method,
                "diagnostics": result.diagnostics,
            })
            if result.status != "OK" or result.weights is None:
                rows.append({"split_id": split.split_id, "challenger": name, "status": result.status})
                continue
            raw = result.weights
            te_constrained, te_info = cap_tracking_error(raw, benchmark, covariance, policy.annual_tracking_error_max)
            actual, turn_info = cap_turnover(current, te_constrained, policy.one_way_turnover_max)
            diff = differential_check(test_execution, actual, policy.differential_tolerance)
            if diff["status"] != "PASS":
                raise ResearchClosed("DIFFERENTIAL_BACKTEST_DIVERGENCE")
            score = metrics(vectorized_returns(test_execution, actual), float(turn_info["actual_turnover"]), policy.transaction_cost_bps)
            row = {
                "split_id": split.split_id,
                "signal_time": signal_time.isoformat(),
                "execution_time": execution_time.isoformat(),
                "test_end": test_execution.index[-1].isoformat(),
                "challenger": name,
                "status": "OK",
                **score,
                **te_info,
                **turn_info,
                "differential_status": diff["status"],
                "differential_max_abs": diff["max_abs_divergence"],
            }
            rows.append(row)
            constraint_rows.append({
                "split_id": split.split_id,
                "challenger": name,
                "raw_target_distance_l1": float((raw - benchmark).abs().sum()),
                "constrained_target_distance_l1": float((actual - benchmark).abs().sum()),
                **te_info,
                **turn_info,
                "binding_constraints": ";".join([key for key, value in [("TRACKING_ERROR", te_info["tracking_error_binding"]), ("TURNOVER", turn_info["turnover_binding"])] if value]) or "NONE",
                "shadow_cost_proxy": float((raw - actual).abs().sum()),
            })
            replay_rows.append({
                "split_id": split.split_id,
                "challenger": name,
                "state": "CLOSED_REPLAY",
                "frozen_snapshot_hash": combined_snapshot_hash,
                "candidate_universe_hash": stable_json_hash(list(economic_returns.columns)),
                "signal_time": signal_time.isoformat(),
                "execution_time": execution_time.isoformat(),
                "holding_period_end": test_execution.index[-1].isoformat(),
                "net_total_return": score["total_return"],
                "no_auto_promotion": True,
            })
            weight_history[name].append(actual)
            latest_targets[name] = {asset: float(weight) for asset, weight in actual.items()}

    detail = pd.DataFrame(rows)
    detail.to_csv(run_dir / "walk_forward_results.csv", index=False)
    challenge_matrix = pd.DataFrame([_aggregate(rows, name) for name in challenger_names])
    stability_rows = []
    for name in challenger_names:
        if weight_history[name]:
            score = stability_score(weight_history[name])
            interpretation = "STABLE_REGION" if score["stable_region_share"] >= 0.70 else "UNSTABLE_REGION"
        else:
            score = {"median_l1_distance": np.nan, "max_l1_distance": np.nan, "stable_region_share": np.nan}
            interpretation = "NOT_EVALUABLE"
        stability_rows.append({"challenger": name, **score, "interpretation": interpretation})
    stability = pd.DataFrame(stability_rows)
    challenge_matrix = challenge_matrix.merge(stability[["challenger", "median_l1_distance", "max_l1_distance", "stable_region_share", "interpretation"]], on="challenger", how="left")
    challenge_matrix.to_csv(run_dir / "challenge_matrix.csv", index=False)
    stability.to_csv(run_dir / "stability_regions.csv", index=False)
    pd.DataFrame(constraint_rows).to_csv(run_dir / "constraint_diagnostics.csv", index=False)

    cpcv_rows: list[dict[str, object]] = []
    cpcv_splits = combinatorial_purged_cv(len(economic_returns), policy.cpcv_folds, policy.cpcv_test_folds, policy.cpcv_embargo_observations, max_paths=max(1, policy.parameter_budget // max(len(challenger_names), 1) * len(challenger_names)))
    for split in cpcv_splits:
        train_economic = economic_returns.iloc[split.train]
        train_execution = execution_returns.iloc[split.train]
        test_execution = execution_returns.iloc[split.test]
        if len(train_economic) < policy.min_train_observations:
            continue
        covariance = train_execution.cov()
        for name in challenger_names:
            result = build_challenger(name, train_economic)
            if result.status != "OK" or result.weights is None:
                cpcv_rows.append({"split_id": split.split_id, "challenger": name, "status": result.status})
                continue
            target, te_info = cap_tracking_error(result.weights, benchmark, covariance, policy.annual_tracking_error_max)
            actual, turn_info = cap_turnover(current, target, policy.one_way_turnover_max)
            score = metrics(vectorized_returns(test_execution.sort_index(), actual), float(turn_info["actual_turnover"]), policy.transaction_cost_bps)
            cpcv_rows.append({"split_id": split.split_id, "challenger": name, "status": "OK", **score, **te_info, **turn_info})
    pd.DataFrame(cpcv_rows).to_csv(run_dir / "cpcv_results.csv", index=False)

    trace_frame = pd.DataFrame(trace_rows)
    with (run_dir / "candidate_filter_trace.jsonl").open("w", encoding="utf-8") as handle:
        for record in trace_rows:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    rejected = trace_frame[trace_frame["reason_code"] != "PASS"].copy()
    promotion_rejections = pd.DataFrame([{
        "candidate": name,
        "stage": "PROMOTION_GATE",
        "rule_id": "NO_AUTO_PROMOTION",
        "reason_code": "RESEARCH_LOCK",
        "metric": "promotion_authority",
        "threshold": "HUMAN_REVIEW",
        "observed": "AUTOMATION_RUN",
        "evidence_gap": "Minimum forward sample and human approval are not satisfied by this run alone.",
        "source_ref": "pre_registration.json",
    } for name in challenger_names])
    rejected = pd.concat([rejected, promotion_rejections], ignore_index=True, sort=False)
    rejected.to_csv(run_dir / "rejected_candidates.csv", index=False)
    replay = pd.DataFrame(replay_rows)
    replay.to_csv(run_dir / "shadow_replay.csv", index=False)
    with (run_dir / "shadow_replay.jsonl").open("w", encoding="utf-8") as handle:
        for record in replay_rows:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    final_test = execution_returns.iloc[wf_splits[-1].test]
    attribution_target_name = "equal_weight"
    attribution_weights = weight_history[attribution_target_name][-1]
    transaction_cost = policy.transaction_cost_bps / 10_000.0 * one_way_turnover(current, attribution_weights)
    attribution_rows: list[dict[str, object]] = []
    for day_number, (date, row_returns) in enumerate(final_test.iterrows()):
        asset_values = {asset: float(attribution_weights[asset] * row_returns[asset]) for asset in final_test.columns}
        cost_value = -transaction_cost if day_number == 0 else 0.0
        component_values = {
            **{f"ASSET:{asset}": value for asset, value in asset_values.items()},
            "ALLOCATION_SIGNAL": 0.0,
            "MARKET_DRIFT": 0.0,
            "CONSTRAINT_TURNOVER_SCALING": 0.0,
            "TRANSACTION_COST_ESTIMATE": cost_value,
            "CASH_RETURN": 0.0,
            "REALIZED_SHADOW_PNL": 0.0,
            "SHADOW_CASH_FLOW": 0.0,
        }
        gross_return = float(sum(asset_values.values()))
        net_return = gross_return + cost_value
        identity_difference = float(sum(component_values.values()) - net_return)
        for component, value in component_values.items():
            classification = "UNREALIZED_SHADOW_PNL" if component.startswith("ASSET:") else component
            attribution_rows.append({
                "date": date.isoformat(),
                "component": component,
                "value": value,
                "classification": classification,
                "gross_portfolio_return": gross_return,
                "net_portfolio_return": net_return,
                "identity_difference": identity_difference,
                "identity_status": "PASS" if abs(identity_difference) <= 1e-12 else "FAIL",
            })
        if abs(identity_difference) > 1e-12:
            raise ResearchClosed("ATTRIBUTION_IDENTITY_FAILED")
    pd.DataFrame(attribution_rows).to_csv(run_dir / "performance_attribution.csv", index=False)

    external_validation = run_external_validation(
        returns=execution_returns,
        weights=current,
        profile=profile,
    )
    write_json(run_dir / "external_validation.json", external_validation)
    write_json(run_dir / "dependency_gates.json", external_validation)

    module_root = Path(__file__).resolve().parents[1]
    frozen_code_paths = sorted(Path(__file__).parent.glob("*.py")) + [
        module_root / "run_shadow_v0_6_3_data2.py",
        module_root / "build_local_proxy.py",
        module_root / "PRE_REGISTRATION.md",
    ]
    code_hashes = {
        path.relative_to(module_root).as_posix(): sha256_file(path)
        for path in frozen_code_paths
    }
    decision_snapshot = {
        "run_id": run_id,
        "model_version": MODEL_VERSION,
        "status": "ACTIVE_SHADOW_LOCKED",
        "signal_date": economic_returns.index[wf_splits[-1].train[-1]].isoformat(),
        "signal_date_semantics": "LATEST_COMPLETE_WALK_FORWARD_TRAINING_CUTOFF; NOT_A_LIVE_ORDER_SIGNAL",
        "as_of_cutoff": as_of.isoformat(),
        "next_tradable_time": economic_returns.index[wf_splits[-1].test[0]].isoformat(),
        "data_fingerprint": combined_fingerprint,
        "data2_assessment": dict(data2_assessment),
        "policy_hash": stable_json_hash(policy_dict(policy)),
        "code_hashes": code_hashes,
        "previous_actual_shadow_weights": {asset: float(value) for asset, value in current.items()},
        "benchmark_weights": {asset: float(value) for asset, value in benchmark.items()},
        "challenger_actual_shadow_targets": latest_targets,
        "champion": "v0.4.1_EXTERNAL_NOT_REPLACED",
        "v0.6.2": "EXTERNAL_BASELINE_NOT_REPLACED",
        "v0.6.3_data2_role": "DUAL_RETURN_CHALLENGER_RESEARCH_ONLY",
        "promotion": "NO_AUTO_PROMOTION",
        "broker_connection": False,
        "orders_generated": False,
    }
    write_json(run_dir / "decision_snapshot.json", decision_snapshot)

    differential_records = detail[["split_id", "challenger", "differential_status", "differential_max_abs"]].dropna().to_dict(orient="records")
    write_json(run_dir / "differential_backtest.json", {
        "status": "PASS" if all(row["differential_status"] == "PASS" for row in differential_records) else "FAIL",
        "tolerance": policy.differential_tolerance,
        "records": differential_records,
        "vectorbt_adapter": external_validation["vectorbt"]["status"],
        "silent_fallback": False,
    })

    required_files = [
        "data_fingerprint.json", "data2_gate.json", "dependency_gates.json", "external_validation.json",
        "return_basis_diagnostics.csv", "pre_registration.json", "walk_forward_results.csv",
        "challenge_matrix.csv", "stability_regions.csv", "constraint_diagnostics.csv", "cpcv_results.csv",
        "candidate_filter_trace.jsonl", "rejected_candidates.csv", "shadow_replay.csv", "shadow_replay.jsonl",
        "performance_attribution.csv", "decision_snapshot.json", "differential_backtest.json",
    ]
    artifacts = []
    for filename in required_files:
        path = run_dir / filename
        if not path.exists() or path.stat().st_size == 0:
            raise ResearchClosed(f"MISSING_REQUIRED_ARTIFACT:{filename}")
        with path.open("r", encoding="utf-8") as handle:
            line_count = sum(1 for _ in handle)
        artifacts.append({"path": filename, "bytes": path.stat().st_size, "lines": line_count, "sha256": sha256_file(path)})
    manifest = {
        "run_id": run_id,
        "model_version": MODEL_VERSION,
        "started_at": started,
        "completed_at": utc_now().isoformat(),
        "status": "ACTIVE_SHADOW_LOCKED",
        "integrity": "PASS",
        "data_grade": data2_assessment["data_grade"],
        "promotion_gate_passed": data2_assessment["promotion_gate_passed"],
        "artifacts": artifacts,
        "research_lock": True,
        "broker_connection": False,
        "orders_generated": False,
    }
    write_json(run_dir / "run_manifest.json", manifest)
    return run_dir
