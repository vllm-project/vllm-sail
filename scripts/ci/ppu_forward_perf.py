# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "scripts/ci/ppu_forward_models.json"


def cases(config):
    dp = config["data_parallel_size"]
    return {
        f"decode_b{batch}_k{config['kv_len']}": batch // dp
        for batch in config["global_batch_sizes"]
    }


def command(config, model, output):
    argv = [
        "torchrun",
        "--standalone",
        f"--nproc-per-node={config['device_count']}",
        "-m",
        "vllm.entrypoints.cli.main",
        "bench",
        "forward",
        "--model",
        str(model),
        *config["engine_args"],
        "--load-format",
        "auto",
        "--routing",
        "real",
        "--kv-len-jitter",
        "0",
        "--seed",
        "0",
        "--exclude-drafter",
        "--iters",
        str(config["iters"]),
        "--samples",
        str(config["samples"]),
        "--output-json",
        str(output),
    ]
    for name, count in cases(config).items():
        argv.extend(["--case", f"{name}={count}*1@{config['kv_len']}"])
    return argv


def positive(value, label):
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{label}: expected a numeric measurement")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label}: expected a finite positive measurement")
    return value


def validate_results(results, config):
    expected = cases(config)
    if not isinstance(results, list) or len(results) != len(expected):
        raise ValueError("forward output must contain exactly the configured cases")
    names = [row["case"] for row in results]
    if len(set(names)) != len(names) or set(names) != set(expected):
        raise ValueError("forward output has missing, duplicate or unexpected cases")
    for row in results:
        name = row["case"]
        request = {
            "query_len": 1,
            "kv_len": config["kv_len"],
            "num_draft_tokens": 0,
            "chunked": False,
        }
        if row["requests"] != [request] * expected[name]:
            raise ValueError(f"{name}: request shapes differ from the configured case")
        positive(row["median_us"], f"{name}.median_us")
        if len(row["ranks"]) != config["device_count"]:
            raise ValueError(f"{name}: incomplete rank results")
        for rank in row["ranks"]:
            if rank["num_tokens"] != expected[name] or rank["routing"] != "real":
                raise ValueError(f"{name}: unexpected token count or routing")
            if rank["timing_scope"] != "target_only":
                raise ValueError(f"{name}: expected target-only forward timing")
            if rank["finite"] is not True:
                raise ValueError(f"{name}: non-finite model output")
            latency = rank["latency_us"]
            if latency["timer"] != "graph_events_rank_max":
                raise ValueError(f"{name}: expected cross-rank MAX graph-event timing")
            if latency["count"] != config["iters"] * config["samples"]:
                raise ValueError(f"{name}: incomplete timed samples")
            samples = latency["samples_us"]
            if len(samples) != latency["count"]:
                raise ValueError(f"{name}: incomplete raw timing samples")
            for value in samples:
                positive(value, f"{name}.samples_us")
            positive(latency["median"], f"{name}.median")
            positive(latency["p90"], f"{name}.p90")
            if not math.isclose(row["median_us"], latency["median"], rel_tol=1e-6):
                raise ValueError(f"{name}: inconsistent cross-rank median")
            spread = latency["sample_spread_pct"]
            if not isinstance(spread, int | float) or not math.isfinite(spread):
                raise ValueError(f"{name}: invalid sample spread")
            if not 0 <= spread <= config["max_sample_spread_pct"]:
                raise ValueError(f"{name}: unstable timing, sample spread={spread}%")
    return results


def report(results, config, identity, golden_root, output):
    sys.path.insert(0, str(ROOT))
    from tests.e2e.perf._baseline_store import (
        LOWER_BETTER,
        BaselineStore,
        MetricSample,
    )

    if (output / "baselines").resolve() == golden_root.resolve():
        raise ValueError("candidate output must not overwrite the reviewed golden")
    store = BaselineStore(
        golden_root, config["chip"], "model", output_root=output / "baselines"
    )
    candidate = BaselineStore(output / "baselines", config["chip"], "model")
    candidate.metrics().clear()
    fingerprint = {
        "benchmark": "forward",
        "timing_scope": "target_only",
        "configuration": config,
        "model_revision": identity["model_revision"],
        "model_config_sha256": identity["model_config_sha256"],
        "runtime_image_id": identity["runtime_image_id"],
        "image_stack": identity["image_stack"],
    }
    summaries = []
    for result in results:
        key = f"{config['case_id']}/{result['case']}/median_us"
        baseline = store.metrics().get(key)
        if baseline is not None:
            if baseline["captured"]["config"] != fingerprint:
                raise ValueError(
                    f"{key}: golden configuration differs; review a new baseline"
                )
            positive(baseline["median"], f"{key}.golden")
            if baseline["unit"] != "us" or baseline["direction"] != LOWER_BETTER:
                raise ValueError(f"{key}: incompatible golden metric")
        latency = result["ranks"][0]["latency_us"]
        sample = MetricSample(
            key,
            result["median_us"],
            LOWER_BETTER,
            unit="us",
            p90=latency["p90"],
            samples=latency["count"],
            config=fingerprint,
        )
        row = store.record(sample)
        candidate.record(sample)
        batch = cases(config)[result["case"]] * config["data_parallel_size"]
        summaries.append(
            {
                "case": result["case"],
                "global_batch_size": batch,
                "per_rank_batch_size": batch // config["data_parallel_size"],
                "median_us": row.current,
                "p90_us": latency["p90"],
                "shape_tokens_per_second": batch * 1_000_000 / row.current,
                "sample_spread_pct": max(
                    rank["latency_us"]["sample_spread_pct"] for rank in result["ranks"]
                ),
                "baseline_us": row.baseline,
                "delta_pct": row.delta_pct,
                "status": row.status,
            }
        )
    candidate.save()
    summary = store.report_json() | {"identity": identity, "cases": summaries}
    (output / "perf-summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    text = store.report_markdown()
    text += "\nShape token/s is a synthetic forward rate, not serving throughput.\n"
    text += "810E: not tested (runner unavailable).\n"
    (output / "perf-summary.md").write_text(text)
    rows = "".join(
        "<tr>"
        + "".join(
            f"<td>{html.escape(str(value))}</td>"
            for value in (
                row["case"],
                row["global_batch_size"],
                f"{row['median_us']:.2f}",
                "-" if row["baseline_us"] is None else f"{row['baseline_us']:.2f}",
                "-" if row["delta_pct"] is None else f"{row['delta_pct']:+.1%}",
                row["status"],
            )
        )
        + "</tr>"
        for row in summaries
    )
    (output / "perf-summary.html").write_text(
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>模型 Forward 性能回归</title><style>"
        'body{font:15px/1.7 "PingFang SC","Microsoft YaHei",sans-serif;padding:24px}'
        ".table-wrap{overflow-x:auto}table{border-collapse:collapse}"
        "th,td{border:1px solid #ccc;padding:8px}"
        "</style><h1>DeepSeek V4 MXFP4 · M890 · DP8+EP</h1>"
        "<p>纯 target forward，KV=4096；不含 drafter、采样和服务开销。"
        "810E 未验证。缺少已审阅 golden 为 PENDING，不是性能通过。</p>"
        '<div class="table-wrap"><table><tr><th>Case</th><th>全局 batch</th><th>当前 μs</th>'
        "<th>Golden μs</th><th>相对变化</th><th>状态</th></tr>"
        f"{rows}</table></div></html>\n"
    )
    print(text)
    return 1 if store.failures() else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument(
        "--golden-root", type=Path, default=ROOT / "tests/e2e/perf/baselines"
    )
    args = parser.parse_args()
    output = args.results_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads(CONFIG.read_text())
    identity = json.loads((output / "runtime-identity.json").read_text())
    identity.update(
        model_id=config["model_id"],
        model_path=str(args.model.resolve()),
        model_revision=args.model_revision,
        model_config_sha256=hashlib.sha256(
            (args.model / "config.json").read_bytes()
        ).hexdigest(),
        image_stack=json.loads((output / "image-stack.json").read_text()),
    )
    argv = command(config, args.model.resolve(), output / "forward.json")
    (output / "benchmark-command.json").write_text(json.dumps(argv, indent=2) + "\n")
    (output / "benchmark-identity.json").write_text(
        json.dumps(identity, indent=2) + "\n"
    )
    print(json.dumps(argv), flush=True)
    subprocess.run(argv, cwd="/tmp", check=True)
    results = validate_results(
        json.loads((output / "forward.json").read_text()), config
    )
    return report(results, config, identity, args.golden_root, output)


if __name__ == "__main__":
    sys.exit(main())
