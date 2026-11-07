"""Typhoon-period gate: blend a trained model with the raw ensemble, chosen per forecast regime.

Why: post-processing helps overall, but at some sites (e.g. K Marado) the trained model is still worse
than the raw ensemble during typhoons. The gate lets validation decide, per (regime x lead band), how much
of the raw ensemble to mix back in.

Regime (decision-time information only): "tc_near" if a tropical cyclone was within `radius_km` of the
site at any time in the 24 h before the decision time (real-time positions are known operationally;
no future track, no future observation is used), else "background".

Blend: q = (1 - w) * q_model + w * q_raw, a quantile average (both sets sorted, so the result is sorted).
w is picked from a grid on the validation split by minimising  mean CRPS + tw_weight * mean twCRPS(threshold)
(both proper scores; samples are never selected by the observed outcome). A tc_near cell with fewer than
`min_events` named cyclone events keeps w = 0 (not enough evidence to switch).

Stages:
  validate  uses the finished validation run (its *_val.npz files), writes <output_dir>/tc_gate/weights.json
            (frozen) and validation_report.csv / .md. No retraining; takes minutes.
  test      needs the experiment's frozen test run (<output_dir>/test/quantiles.npz) and weights.json;
            applies the frozen weights once and writes test_report.csv / .md.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import feature_names
from .experiment import load_sites, build_all, write_json
from .metrics import score, paired_bootstrap, quantile_at

GRID = [round(x, 2) for x in np.linspace(0, 1, 11)]
BANDS = ["D1-3", "D4-7"]                # + "D8-<horizon_days>", added from the config


def twcrps(q, y, t):
    return score(np.maximum(q, t), np.maximum(y, t))


def decision_tc_distance(sites, samples):
    """Minimum cyclone distance (km) over the 24 h up to each sample's decision time."""
    labels = {str(c["site_id"]): obs["label_tc_distance_km"] if "label_tc_distance_km" in obs else None
              for c, obs, _ in sites}
    out = []
    for s in samples:
        m = s["metadata"][0]
        lab = labels[str(m["site_id"])]
        if lab is None:
            out.append(np.nan)
            continue
        d = pd.Timestamp(m["decision_time"])
        window = lab.loc[(lab.index > d - pd.Timedelta(hours=24)) & (lab.index <= d)]
        out.append(float(window.min()) if window.notna().any() else np.nan)
    return np.asarray(out)


def case_frame(samples, q, raw, regimes, label, opts):
    rows = pd.DataFrame([m for s in samples for m in s["metadata"]])
    y = np.concatenate([s["y"] for s in samples])
    rows["model"] = label
    rows["regime"] = np.repeat(regimes, [len(s["y"]) for s in samples])
    rows["crps"] = score(q, y)
    rows["twcrps"] = twcrps(q, y, opts["threshold_m"])
    rows["raw_crps"] = score(raw, y)
    rows["covered80"] = (y >= quantile_at(q, .1)) & (y <= quantile_at(q, .9))
    return rows


def objective(crps, tw, opts):
    return float(np.mean(crps) + opts["tw_weight"] * np.mean(tw))


def model_quantiles(root, cfg, model, ablation, split):
    if split == "val":
        files = [root / f"{model}_{ablation}_seed{s}_val.npz" for s in cfg["seeds"]]
        missing = [str(f) for f in files if not f.exists()]
        if missing:
            raise SystemExit(f"Validation predictions missing (run validation first): {missing}")
        return np.mean([np.load(f)["quantiles"] for f in files], axis=0)
    f = root / "test" / "quantiles.npz"
    if not f.exists():
        raise SystemExit(f"{f} missing: run the frozen test stage of the experiment first "
                         "(python run_training.py <config> test), then this gate's test stage")
    return np.load(f)[model]


def prepare(cfg, model, split, opts):
    root = Path(cfg["output_dir"])
    selection = json.loads((root / "selection.json").read_text(encoding="utf-8"))
    ablation = selection["best"][model]["ablation"]
    sites, _ = load_sites(cfg)
    samples, _, _ = build_all(sites, cfg, feature_names(cfg, ablation))
    samples = samples[split]
    q = model_quantiles(root, cfg, model, ablation, split)
    n = sum(len(s["y"]) for s in samples)
    if len(q) != n:
        raise SystemExit(f"Prediction rows {len(q)} != sample rows {n}; config or data changed since training")
    if any(s["raw_quantiles"] is None for s in samples):
        raise SystemExit("forecasts.csv has no raw ensemble quantiles q000-q098; the gate needs them")
    raw = np.concatenate([s["raw_quantiles"] for s in samples])
    y = np.concatenate([s["y"] for s in samples])
    dist = decision_tc_distance(sites, samples)
    regimes = np.where(dist <= opts["radius_km"], "tc_near", "background")
    per_row_regime = np.repeat(regimes, [len(s["y"]) for s in samples])
    bands = np.concatenate([[m["lead_band"] for m in s["metadata"]] for s in samples])
    events = np.concatenate([[m["event_id"] or "" for m in s["metadata"]] for s in samples])
    return root, selection, ablation, samples, q, raw, y, per_row_regime, regimes, bands, events


def report(frame, path_csv, path_md, title, weights=None):
    groups = []
    for keys in (["model"], ["model", "site_id"], ["model", "regime"], ["model", "site_id", "regime"],
                 ["model", "regime", "lead_band"]):
        g = frame.groupby(keys).agg(n=("crps", "size"), crps=("crps", "mean"), twcrps=("twcrps", "mean"),
                                    coverage80=("covered80", "mean")).reset_index()
        g.insert(0, "grouping", "×".join(keys[1:]) or "overall")
        groups.append(g)
    out = pd.concat(groups, ignore_index=True)
    out.to_csv(path_csv, index=False, encoding="utf-8-sig")
    lines = [f"# {title}", ""]
    if weights:
        lines += ["## 冻结的混合权重 w（0 = 纯模型，1 = 纯原始集合）", "", "| 天气型 | 预见期 | w | 样本 | 具名台风事件 | 说明 |",
                  "|---|---|---:|---:|---:|---|"]
        for k, v in weights.items():
            r, b = k.split("|")
            lines.append(f"| {r} | {b} | {v['w']:.1f} | {v['n_cases']} | {v['n_events']} | {v['note']} |")
        lines.append("")
    lines += ["## 成绩（CRPS、阈值加权 CRPS，单位 m）", "", "| 分组 | 模型 | 站点 | 天气型 | 预见期 | 样本 | CRPS | twCRPS | 80% 覆盖率 |",
              "|---|---|---|---|---|---:|---:|---:|---:|"]
    for _, r in out.iterrows():
        lines.append(f"| {r.grouping} | {r.model} | {r.get('site_id', '') if pd.notna(r.get('site_id', '')) else ''} | "
                     f"{r.get('regime', '') if pd.notna(r.get('regime', '')) else ''} | "
                     f"{r.get('lead_band', '') if pd.notna(r.get('lead_band', '')) else ''} | {r.n} | "
                     f"{r.crps:.4f} | {r.twcrps:.4f} | {r.coverage80:.1%} |")
    Path(path_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def validate(cfg, model, opts):
    root, selection, ablation, samples, q, raw, y, rr, regimes, bands, events = prepare(cfg, model, "val", opts)
    out = root / "tc_gate"
    out.mkdir(exist_ok=True)
    print(f"Gate candidate: {model}_{ablation}; validation rows {len(y)}; "
          f"forecast runs in tc_near regime: {(regimes == 'tc_near').sum()} / {len(regimes)}", flush=True)
    weights = {}
    for regime in ["background", "tc_near"]:
        for band in BANDS + [f"D8-{cfg['horizon_days']}"]:
            k = (rr == regime) & (bands == band)
            n_events = len({e for e in events[k] if e})
            if k.sum() == 0:
                weights[f"{regime}|{band}"] = dict(w=0.0, n_cases=0, n_events=0, note="no cases")
                continue
            scores = {w: objective(score((1 - w) * q[k] + w * raw[k], y[k]),
                                   twcrps((1 - w) * q[k] + w * raw[k], y[k], opts["threshold_m"]), opts) for w in GRID}
            best_w = min(scores, key=scores.get)
            note = f"objective w=0 {scores[0.0]:.4f} → w={best_w} {scores[best_w]:.4f}"
            if regime == "tc_near" and n_events < opts["min_events"]:
                note = f"only {n_events} events < {opts['min_events']}: keep model (best would be w={best_w})"
                best_w = 0.0
            weights[f"{regime}|{band}"] = dict(w=best_w, n_cases=int(k.sum()), n_events=n_events, note=note)
            print(f"  {regime:10s} {band:6s}: w = {best_w:.1f}  ({note})", flush=True)
    w_row = np.array([weights[f"{r}|{b}"]["w"] for r, b in zip(rr, bands)])[:, None]
    gated = (1 - w_row) * q + w_row * raw
    frame = pd.concat([case_frame(samples, raw, raw, regimes, "raw_ensemble", opts),
                       case_frame(samples, q, raw, regimes, f"{model}_{ablation}", opts),
                       case_frame(samples, gated, raw, regimes, "tc_gated", opts)], ignore_index=True)
    a, b = frame[frame.model == "tc_gated"], frame[frame.model == f"{model}_{ablation}"]
    ci = paired_bootstrap(a, b, cfg["bootstrap_repeats"])
    frozen = dict(config_hash=hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
                  model=model, ablation=ablation, options=opts, weights=weights,
                  validation_crps_change_vs_model=ci,
                  warning="Weights chosen on validation; validation scores are optimistic. Report test once.")
    write_json(out / "weights.json", frozen)
    report(frame, out / "validation_report.csv", out / "validation_report.md", "台风期门控：验证集（选权重用，成绩偏乐观）", weights)
    print(f"Gated vs {model}_{ablation} on validation: CRPS change {ci['delta']:+.4f} m "
          f"(95% CI {ci['ci_low']:+.4f} … {ci['ci_high']:+.4f}). Frozen: {out / 'weights.json'}", flush=True)


def test(cfg, model, opts):
    root = Path(cfg["output_dir"])
    wf = root / "tc_gate" / "weights.json"
    if not wf.exists():
        raise SystemExit(f"{wf} missing: run the gate's validate stage first")
    frozen = json.loads(wf.read_text(encoding="utf-8"))
    if frozen["model"] != model or frozen["options"] != opts:
        raise SystemExit("Model or options differ from the frozen validation gate")
    root, selection, ablation, samples, q, raw, y, rr, regimes, bands, events = prepare(cfg, model, "test", opts)
    w_row = np.array([frozen["weights"][f"{r}|{b}"]["w"] for r, b in zip(rr, bands)])[:, None]
    gated = (1 - w_row) * q + w_row * raw
    frame = pd.concat([case_frame(samples, raw, raw, regimes, "raw_ensemble", opts),
                       case_frame(samples, q, raw, regimes, f"{model}_{ablation}", opts),
                       case_frame(samples, gated, raw, regimes, "tc_gated", opts)], ignore_index=True)
    out = root / "tc_gate"
    report(frame, out / "test_report.csv", out / "test_report.md", "台风期门控：测试集（冻结权重，只评一次）", frozen["weights"])
    ci = paired_bootstrap(frame[frame.model == "tc_gated"], frame[frame.model == f"{model}_{ablation}"],
                          cfg["bootstrap_repeats"])
    print(f"Test: gated vs {model}_{ablation} CRPS change {ci['delta']:+.4f} m "
          f"(95% CI {ci['ci_low']:+.4f} … {ci['ci_high']:+.4f}). Report: {out / 'test_report.md'}", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description="Typhoon-period gate between a trained model and the raw ensemble")
    p.add_argument("--config", required=True)
    p.add_argument("--stage", choices=["validate", "test"], default="validate")
    p.add_argument("--model", choices=["tide", "transformer", "gru", "cnn_bigru"], default="transformer")
    p.add_argument("--radius-km", type=float, default=1000)
    p.add_argument("--threshold-m", type=float, default=2.5)
    p.add_argument("--tw-weight", type=float, default=1.0)
    p.add_argument("--min-events", type=int, default=3)
    a = p.parse_args(argv)
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    opts = dict(radius_km=a.radius_km, threshold_m=a.threshold_m, tw_weight=a.tw_weight, min_events=a.min_events)
    (validate if a.stage == "validate" else test)(cfg, a.model, opts)


if __name__ == "__main__":
    main()
