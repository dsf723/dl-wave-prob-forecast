"""复现核对（10-28）：在解压后的复现包里（任何目录、任何机器，CPU 也可）用冻结权重重新预测，与已报告的结果逐项比对。

    python verify_release.py              # 约 10–30 分钟（CPU）；结果写 verify_report.md
    python verify_release.py --retrain    # 另外从头重训两个集成成员（只跑验证阶段，GPU 建议，约 1 小时），比较验证集 CRPS
检查项：
  1. 文件完整：MANIFEST.sha256 里每个文件的 SHA256；
  2. 冻结配置：三个运行的配置指纹（与机器无关的 config_hash_portable）与冻结时一致；
  3. 2017–2019 测试集：各模型重新预测的分位数与当时保存的 test/quantiles.npz 的最大差、CRPS；
     集成、EMOS（重新拟合）、原始集合、气候态的 CRPS 与报告值（results/测试集最终结果_15天集成.md）；
  4. 独立业务检验（_oper、_oper_hsbridge）：同样与 results/operational/<A>_ens<后缀>/quantiles.npz 和报告值比对。
判定：分位数最大差 < 0.001 m（CPU 与 GPU 的浮点差在 1e-5 量级），CRPS 与报告值相差 ≤ 0.0001 m，
      相对变化与报告值相差 ≤ 0.1 个百分点。测试集只用于核对已冻结的预测，不做任何挑选。
--retrain 只重跑验证阶段（训练 + 验证集挑选），不碰测试集；GPU 运算不完全确定，结果应接近而非逐位相同（判定：验证集 CRPS 相差 < 2%）。
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'outputs', 'wave_forecast'))
from project_paths import resolve_config                                     # noqa: E402
import torch                                                                 # noqa: E402
from torch.utils.data import DataLoader                                      # noqa: E402
from evaluate_operational import frozen, operational_cfg                     # noqa: E402
from wavecast.data import feature_names, Scaler                              # noqa: E402
from wavecast.experiment import load_sites, build_all, climatology, make_model, predict, predict_te, tensors  # noqa: E402
from wavecast.metrics import score                                           # noqa: E402
from wavecast.postproc import emos_benchmark                                 # noqa: E402

# reported values: results/测试集最终结果_15天集成.md, results/operational/*/report.md
REPORTED = {
    'test': dict(ensemble=0.2272, transformer=0.2295, gru=0.2279, emos_ln=0.2625, raw_ensemble=0.3011, climatology=0.3171,
                 rel_raw=-0.245, rel_emos=-0.134),
    '_oper': dict(ensemble=0.2236, transformer=0.2265, gru=0.2237, emos_ln=0.2453, raw_ensemble=0.2860, climatology=0.3128,
                  rel_raw=-0.218, rel_emos=-0.088),
    '_oper_hsbridge': dict(ensemble=0.2225, transformer=0.2257, gru=0.2222, emos_ln=0.2445, raw_ensemble=0.2703,
                           climatology=0.3128, rel_raw=-0.177, rel_emos=-0.090),
}
Q_TOL, CRPS_TOL, REL_TOL = 1e-3, 1e-4, 1e-3
rows, fails = [], []


def check(item, ok, detail):
    rows.append(f'| {item} | {"通过" if ok else "**不通过**"} | {detail} |')
    if not ok:
        fails.append(item)
    print(f'[{"通过" if ok else "不通过"}] {item}：{detail}', flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def predict_seeds(cfg, model, ablation, samples, device):
    qs, tes = [], []
    for seed in cfg['seeds']:
        state = torch.load(os.path.join(cfg['output_dir'], f'{model}_{ablation}_seed{seed}.pt'), map_location=device, weights_only=False)
        scaler = Scaler(); scaler.mean = np.array(state['scaler']['mean']); scaler.std = np.array(state['scaler']['std'])
        net = make_model(model, state['n_features'], state['history'], state['horizon'], cfg).to(device)
        net.load_state_dict(state['model'])
        loader = DataLoader(tensors(samples, scaler), batch_size=cfg['batch_size'])
        qs.append(predict(net, loader, device))
        if cfg.get('te_target'):
            tes.append(predict_te(net, loader, device))
    return np.mean(qs, axis=0), (np.mean(tes, axis=0) if tes else None)


def compare(tag, name, q, y, recorded=None, key=None, reported=None):
    crps = score(q, y).mean()
    if recorded is not None and key in recorded:
        d = float(np.max(np.abs(q - recorded[key])))
        check(f'{tag} {name} 分位数', d < Q_TOL, f'与保存的预测最大差 {d:.2e} m')
    if reported is not None:
        check(f'{tag} {name} CRPS', abs(crps - reported) <= CRPS_TOL, f'重算 {crps:.4f}，报告 {reported:.4f}')
    return crps


def section(tag, runs, samples, train, sites, names, recorded_by_run, recorded_all, rep, device, cfg_target):
    y = np.concatenate([s['y'] for s in samples]).astype(float)
    print(f'\n== {tag}：{len(samples)} 次站点起报，{len(y)} 个时刻', flush=True)
    rows.append(f'| **{tag}**（{len(y)} 个时刻） | | |')
    q = {}
    for run, (cfg, model, ablation) in runs.items():
        q[model], _ = predict_seeds(cfg, model, ablation, samples, device)
        compare(tag, model, q[model], y, recorded_by_run.get(run), model, rep.get(model))
    q['ensemble'] = sum(q.values()) / len(q)
    compare(tag, '集成', q['ensemble'], y, recorded_all, 'ensemble', rep.get('ensemble'))
    emos = emos_benchmark(train, samples, names)['ln']
    compare(tag, 'EMOS 对数正态', emos, y, recorded_all, 'emos_ln', rep.get('emos_ln'))
    raw = np.concatenate([s['raw_quantiles'] for s in samples]).astype(float)
    compare(tag, 'GEFS 原始集合', raw, y, recorded_all, 'raw_ensemble', rep.get('raw_ensemble'))
    clim = climatology(sites, samples, cfg_target)
    compare(tag, '气候态', clim, y, recorded_all, 'climatology', rep.get('climatology'))
    se = score(q['ensemble'], y).sum()
    for k, ref in (('rel_raw', raw), ('rel_emos', emos)):
        rel = se / score(ref, y).sum() - 1
        check(f'{tag} 集成相对{"原始集合" if k == "rel_raw" else " EMOS"}', abs(rel - rep[k]) <= REL_TOL,
              f'重算 {rel:+.1%}，报告 {rep[k]:+.1%}')


def retrain(run, device):
    src = json.load(open(os.path.join(ROOT, 'model_configs', f'{run}.json'), encoding='utf-8'))
    src['output_dir'] = f'model_runs/_retrain/{run}'
    path = os.path.join(ROOT, 'model_configs', f'_retrain_{run}.json')
    json.dump(src, open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'\n== 重新训练 {run}（只跑验证阶段）→ {src["output_dir"]}', flush=True)
    subprocess.run([sys.executable, os.path.join(ROOT, 'run_training.py'), path, 'validate'], check=True)
    cfg, model, ablation = frozen(run)
    new = json.load(open(resolve_config(path), encoding='utf-8'))
    sites, _ = load_sites(cfg)
    samples, _, _ = build_all(sites, cfg, feature_names(cfg, ablation))
    y = np.concatenate([s['y'] for s in samples['val']]).astype(float)
    sel = json.load(open(os.path.join(new['output_dir'], 'selection.json'), encoding='utf-8'))
    same = sel['best'] == json.load(open(os.path.join(cfg['output_dir'], 'selection.json'), encoding='utf-8'))['best']
    check(f'重训 {run} 挑选结果', same, f'重训 {sel["best"]}')
    val = lambda d, abl: np.mean([np.load(os.path.join(d, f'{model}_{abl}_seed{s}_val.npz'))['quantiles'] for s in cfg['seeds']], axis=0)
    old_c = score(val(cfg['output_dir'], ablation), y).mean()
    new_c = score(val(new['output_dir'], ablation), y).mean()
    check(f'重训 {run} 验证集 CRPS', abs(new_c / old_c - 1) < 0.02, f'冻结 {old_c:.4f}，重训 {new_c:.4f}（{new_c / old_c - 1:+.1%}）')


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    p = argparse.ArgumentParser()
    p.add_argument('--retrain', action='store_true')
    p.add_argument('--skip-manifest', action='store_true')
    a = p.parse_args()
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rel = json.load(open(os.path.join(ROOT, 'release.json'), encoding='utf-8'))
    rows.extend(['| 检查项 | 结果 | 说明 |', '|---|---|---|'])

    # 1. manifest
    if not a.skip_manifest:
        bad, n = [], 0
        for line in open(os.path.join(ROOT, 'MANIFEST.sha256'), encoding='utf-8'):
            h, f = line.rstrip('\n').split('  ', 1)
            n += 1
            path = os.path.join(ROOT, f)
            if not os.path.exists(path) or sha256(path) != h:
                bad.append(f)
        check('文件完整（SHA256）', not bad, f'{n} 个文件' + (f'，不一致 {len(bad)} 个：{bad[:5]}' if bad else '全部一致'))

    # 2. frozen configs
    runs = {}
    for run in rel['runs']:
        try:
            runs[run] = frozen(run)
            check(f'冻结配置 {run}', True, f'{runs[run][1]} / {runs[run][2]}，指纹一致')
        except SystemExit as e:
            check(f'冻结配置 {run}', False, str(e))
    if fails:
        raise SystemExit('冻结配置核对不通过，停止')

    # 3. 2017–2019 test set
    ens = {r: runs[r] for r in rel['ensemble']}
    cfgA, modelA, ablA = ens[rel['ensemble'][0]]
    if any(feature_names(c, ab) != feature_names(cfgA, ablA) for c, _, ab in ens.values()):
        raise SystemExit('集成成员的输入特征不同')
    sitesA, _ = load_sites(cfgA)
    samplesA, _, names = build_all(sitesA, cfgA, feature_names(cfgA, ablA))
    rec = {r: dict(np.load(os.path.join(c['output_dir'], 'test', 'quantiles.npz')))
           for r, (c, _, _) in ens.items() if rel['test_recorded'].get(r)}
    emos_path = os.path.join(ROOT, 'results', 'postproc', f'{rel["ensemble"][0]}_test_emos.npz')
    rec_all = {}
    if len(rec) == len(ens):
        rec_all['ensemble'] = sum(rec[r][runs[r][1]] for r in ens) / len(ens)
    if os.path.exists(emos_path):
        rec_all['emos_ln'] = np.load(emos_path)['ln']
    section('测试集 2017–2019', ens, samplesA['test'], samplesA['train'], sitesA, names, rec, rec_all, REPORTED['test'], device, cfgA['target'])
    for run, (cfg, model, ablation) in runs.items():          # the period (Te) multitask run, not part of the ensemble
        if run in ens or not rel['test_recorded'].get(run):
            continue
        sites, _ = load_sites(cfg)
        smp, _, _ = build_all(sites, cfg, feature_names(cfg, ablation))
        q, te = predict_seeds(cfg, model, ablation, smp['test'], device)
        old = np.load(os.path.join(cfg['output_dir'], 'test', 'quantiles.npz'))[model]
        check(f'测试集 {run} Hs 分位数', float(np.max(np.abs(q - old))) < Q_TOL, f'最大差 {np.max(np.abs(q - old)):.2e} m')
        tpath = os.path.join(cfg['output_dir'], 'test', 'te_quantiles.npz')
        if te is not None and os.path.exists(tpath):
            d = float(np.nanmax(np.abs(te - np.load(tpath)['quantiles'])))
            check(f'测试集 {run} Te 分位数', d < Q_TOL * 10, f'最大差 {d:.2e} s')

    # 4. operational check
    for suffix in rel['operational_suffixes']:
        op = operational_cfg(cfgA, suffix, list(range(2020, 2026)))
        sites_op, _ = load_sites(op)
        smp_op, _, names_op = build_all(sites_op, op, feature_names(cfgA, ablA))
        recorded = dict(np.load(os.path.join(ROOT, 'results', 'operational', f'{rel["ensemble"][0]}_ens{suffix}', 'quantiles.npz')))
        section(f'业务检验 {suffix}', ens, smp_op['test'], samplesA['train'], sites_op, names_op,
                {r: recorded for r in ens}, recorded, REPORTED[suffix], device, cfgA['target'])

    if a.retrain:
        for run in rel['ensemble']:
            retrain(run, device)

    head = [f'# 复现核对报告', '', f'运行于 {time.strftime("%Y-%m-%d %H:%M")}，设备 {device}，用时 {(time.time() - t0) / 60:.0f} 分钟；'
            f'复现包生成于 {rel["created"]}（{rel["machine"]}）。', '',
            f'**结论：{"全部通过" if not fails else f"{len(fails)} 项不通过"}**', '']
    open(os.path.join(ROOT, 'verify_report.md'), 'w', encoding='utf-8').write('\n'.join(head + rows) + '\n')
    print(f'\n{"全部通过" if not fails else "不通过：" + "、".join(fails)}；报告 → verify_report.md', flush=True)
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
