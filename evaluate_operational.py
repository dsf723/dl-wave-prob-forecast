"""独立业务检验：冻结的最终模型（Transformer + GRU 分位数平均集成）在 2020-09 起的 GEFSv12 业务预报上评分（10-27）。

为什么：2017–2019 测试集已用于 14 天模型的评价；业务预报（download_gefs_operational.py 下载、
prepare_reforecast_dataset.py --sources gefs_operational --suffix _oper 整理）是模型从未见过的全新数据。
做法（不训练、不挑选、不调参）：
  - 读两个冻结运行的 selection.json，按 run_training.py 同样的方式解析配置并核对配置指纹，不一致就拒绝；
  - 只把各站 data_dir 换成 processed/site_<ID>_reforecast<后缀>、test_years 换成业务年份（训练、验证年份不变，
    气候态仍按训练年计算）；读各种子的权重预测，种子平均；集成 = 两者分位数平均（与 export --ensemble-with 相同）；
  - 基准：GEFS 业务原始集合（11 成员）、EMOS 截断正态 / 对数正态（系数用原回算训练年拟合，不用业务数据）、
    气候态、持续性、原始集合平均；可选：验证集定好的共形平移量（results/postproc/<A>_ens_conformal_shifts.json）；
  - 按 block_id 配对自助 2000 次；有 2017–2019 测试结果时并列比较"回算测试集 vs 业务"的相对提升。
输出：results/operational/<A>_ens<后缀>/ 下 report.md、case_scores.csv、grouped_scores.csv、paired_comparisons.json、
      quantiles.npz（各系统分位数）、cases.csv（站点、起报、有效时刻、实测）。
用法：python evaluate_operational.py                                  # 主结果（业务原始波高）
      python evaluate_operational.py --suffix _oper_hsbridge           # 敏感性：波高按 ERA5 对齐
      [--a GKH_h15_transformer_pretrain --b GKH_h15_gru_pretrain --years 2020 2025 --threshold 2.5]
11-04 加（台湾海峡，事先登记）：--eval-sites 只评部分站（其余站没有 2020 年后的实测），站点独热布局仍按冻结配置的全部站；
      --pooled-blocks 块名去掉站点前缀，同一时段（28 天或同一事件）的所有站一起抽样（海峡各站天气相关）。
      python evaluate_operational.py --a TWS_h15_transformer_pretrain --b TWS_h15_gru_pretrain --suffix _oper --years 2020 2024 --eval-sites TW46757B TWC6W08 --pooled-blocks
"""
import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'outputs', 'wave_forecast'))
from project_paths import resolve_config                                     # noqa: E402
from route_models import boot                                                # noqa: E402
import torch                                                                 # noqa: E402
from torch.utils.data import DataLoader                                      # noqa: E402
from wavecast.data import feature_names, Scaler                              # noqa: E402
from wavecast.experiment import (load_sites, build_all, climatology, baseline_tables, write_reports,  # noqa: E402
                                 frozen_match, make_model, predict, tensors)
from wavecast.metrics import score, case_table                               # noqa: E402
from wavecast.postproc import apply_shifts, emos_benchmark                   # noqa: E402

OUT = os.path.join(ROOT, 'results', 'operational')


def tw(q, y, t):
    return score(np.maximum(q, t), np.maximum(y, t))


def frozen(run):
    """按 run_training.py 的方式解析配置，核对冻结指纹；返回 (配置, 选中的消融)。"""
    cfg = json.load(open(resolve_config(os.path.join(ROOT, 'model_configs', f'{run}.json')), encoding='utf-8'))
    sel = json.load(open(os.path.join(cfg['output_dir'], 'selection.json'), encoding='utf-8'))
    if not frozen_match(sel, cfg):
        raise SystemExit(f'{run}：配置与冻结时不一致（指纹不同），拒绝评分')
    model, info = next(iter(sel['best'].items()))
    return cfg, model, info['ablation']


def operational_cfg(cfg, suffix, years, eval_sites=None):
    """只换数据目录与测试年份；训练、验证年份不变（气候态、站点独热布局都与冻结时相同）。eval_sites：只保留这些站。"""
    new = copy.deepcopy(cfg)
    if eval_sites:
        new['sites'] = [s for s in new['sites'] if str(s['site_id']) in eval_sites]
        if len(new['sites']) != len(eval_sites):
            raise SystemExit(f'--eval-sites 里有冻结配置之外的站：{eval_sites}')
    for s in new['sites']:
        s['data_dir'] = os.path.join(ROOT, 'processed', f"site_{s['site_id']}_reforecast{suffix}").replace('\\', '/')
        if not os.path.exists(os.path.join(s['data_dir'], 'manifest.json')):
            raise SystemExit(f"没有 {s['data_dir']}：先 python prepare_reforecast_dataset.py G K H --sources gefs_operational --suffix {suffix}")
        if set(years) & (set(s['train_years']) | set(s['val_years'])):
            raise SystemExit('业务年份与训练 / 验证年份重叠')
        s['test_years'] = list(years)
    return new


def predict_frozen(cfg, model, ablation, samples, device):
    qs = []
    for seed in cfg['seeds']:
        state = torch.load(os.path.join(cfg['output_dir'], f'{model}_{ablation}_seed{seed}.pt'), map_location=device, weights_only=False)
        scaler = Scaler(); scaler.mean = np.array(state['scaler']['mean']); scaler.std = np.array(state['scaler']['std'])
        net = make_model(model, state['n_features'], state['history'], state['horizon'], cfg).to(device)
        net.load_state_dict(state['model'])
        qs.append(predict(net, DataLoader(tensors(samples, scaler), batch_size=cfg['batch_size']), device))
    return np.mean(qs, axis=0)


def case_keys(samples):
    return np.array([f"{m['site_id']}|{pd.Timestamp(m['issue_time'])}|{pd.Timestamp(m['valid_time'])}" for s in samples for m in s['metadata']])


def md(df):
    lines = ['| ' + ' | '.join(df.columns) + ' |', '|' + '---|' * len(df.columns)]
    for _, r in df.iterrows():
        lines.append('| ' + ' | '.join(f'{v:.4f}' if isinstance(v, (float, np.floating)) else str(v) for v in r) + ' |')
    return lines


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    p = argparse.ArgumentParser()
    p.add_argument('--a', default='GKH_h15_transformer_pretrain')
    p.add_argument('--b', default='GKH_h15_gru_pretrain')
    p.add_argument('--suffix', default='_oper')
    p.add_argument('--years', nargs=2, type=int, default=[2020, 2025])
    p.add_argument('--threshold', type=float, default=2.5)
    p.add_argument('--eval-sites', nargs='+')
    p.add_argument('--pooled-blocks', action='store_true')
    a = p.parse_args()
    years = list(range(a.years[0], a.years[1] + 1))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    runs = {}
    for run in (a.a, a.b):
        cfg, model, ablation = frozen(run)
        runs[run] = dict(cfg=cfg, model=model, ablation=ablation, op=operational_cfg(cfg, a.suffix, years, a.eval_sites))
    ca = runs[a.a]
    if feature_names(ca['cfg'], ca['ablation']) != feature_names(runs[a.b]['cfg'], runs[a.b]['ablation']):
        raise SystemExit('两个运行的输入特征不同，集成的样本无法对齐')
    sites_op, manifests = load_sites(ca['op'])
    layout = [str(s['site_id']) for s in ca['cfg']['sites']]                  # one-hot layout of the frozen run
    samples_op, skipped, names = build_all(sites_op, ca['op'], feature_names(ca['cfg'], ca['ablation']), site_ids=layout)
    test = samples_op['test']
    if not test:
        raise SystemExit(f'业务数据里没有可用起报（跳过原因 {skipped}）')
    meta = pd.DataFrame([m for s in test for m in s['metadata']])
    meta['site_id'] = meta.site_id.astype(str)
    y = np.concatenate([s['y'] for s in test]).astype(float)
    print(f'业务检验：{len(test)} 次起报（站点 × 起报），{len(y)} 个时刻；设备 {device}', flush=True)
    q = {}
    for run, r in runs.items():
        print(f'预测：{run}（{r["model"]}_{r["ablation"]}，种子 {r["cfg"]["seeds"]}）', flush=True)
        q[r['model']] = predict_frozen(r['cfg'], r['model'], r['ablation'], test, device)
    models = [runs[a.a]['model'], runs[a.b]['model']]
    q['ensemble'] = (q[models[0]] + q[models[1]]) / 2
    # Original reforecast training split: EMOS coefficients and (if present) the 2017–2019 test comparison.
    sites_orig, _ = load_sites(ca['cfg'])
    samples_orig, _, names_orig = build_all(sites_orig, ca['cfg'], feature_names(ca['cfg'], ca['ablation']))
    if names_orig != names:
        raise SystemExit('业务样本与回算样本的输入列不一致')
    clim = climatology(sites_op, test, ca['cfg']['target'])
    print('基准：EMOS（回算训练年拟合）、GEFS 业务原始集合、气候态…', flush=True)
    emos = emos_benchmark(samples_orig['train'], test, names)
    tables = baseline_tables(test, clim, ca['op'])
    tables += [case_table(test, emos[k], f'emos_{k}', ca['op'], clim) for k in ('tn', 'ln')]
    shifts_path = os.path.join(ROOT, 'results', 'postproc', f'{a.a}_ens_conformal_shifts.json')
    if os.path.exists(shifts_path):
        sh = {g: np.asarray(v) for g, v in json.load(open(shifts_path, encoding='utf-8'))['shifts'].items()}
        q['ensemble_cqr'] = apply_shifts(q['ensemble'], (meta.site_id + '|' + meta.lead_band).to_numpy(), sh)
    tables += [case_table(test, v, k, ca['op'], clim) for k, v in q.items()]
    tag = f'{a.a}_ens{a.suffix}' + ('_' + '_'.join(a.eval_sites) if a.eval_sites else '')
    out = os.path.join(OUT, tag)
    os.makedirs(out, exist_ok=True)
    write_reports(Path(out), tables, ca['op'])
    allq = dict(q, raw_ensemble=np.concatenate([s['raw_quantiles'] for s in test]).astype(float),
                emos_tn=emos['tn'], emos_ln=emos['ln'], climatology=clim)
    np.savez_compressed(os.path.join(out, 'quantiles.npz'), **allq)
    meta.assign(truth=y)[['site_id', 'issue_time', 'valid_time', 'lead_day', 'lead_band', 'block_id', 'truth']].to_csv(
        os.path.join(out, 'cases.csv'), index=False)

    # ---------- report ----------
    t, rng = a.threshold, np.random.default_rng(0)
    blocks = meta.block_id.astype(str).to_numpy()
    if a.pooled_blocks:
        pre = tuple(f'{x}_' for x in layout)
        blocks = np.array([b[len(next(p for p in pre if b.startswith(p))):] if b.startswith(pre) else b for b in blocks])
    label = {'ensemble': '集成（最终方案）', models[0]: f'{models[0]}（A）', models[1]: f'{models[1]}（B）',
             'ensemble_cqr': '集成 + 共形校正（验证集平移量）', 'raw_ensemble': 'GEFS 业务原始集合', 'emos_ln': 'EMOS 对数正态',
             'emos_tn': 'EMOS 截断正态', 'climatology': '气候态'}
    rows = []
    for k in label:
        if k in allq:
            v = allq[k]
            rows.append(dict(系统=label[k], CRPS=score(v, y).mean(), twCRPS=tw(v, y, t).mean(),
                             覆盖80=((y >= v[:, 9]) & (y <= v[:, 89])).mean(), 宽度80=(v[:, 89] - v[:, 9]).mean()))
    period = f"{meta.issue_time.min():%Y-%m-%d} 至 {meta.issue_time.max():%Y-%m-%d}"
    lines = [f'# 独立业务检验：{a.a} + {a.b} 集成（{a.suffix}）', '',
             f'GEFSv12 业务海浪集合（2020-09-23 起），起报 {period}，{meta.groupby("site_id").issue_time.nunique().to_dict()} 次，'
             f'{len(y)} 个时刻。模型、挑选、后处理全部冻结（配置指纹已核对），这些数据从未用于训练、挑选或调参。', '',
             '输入换算（以 ERA5 为桥梁，不用浮标实测）：']
    for sid, m in manifests.items():
        h = m.get('operational_harmonization') or {}
        lines.append(f"- {sid}：Te 系数 {h.get('te_scale', float('nan')):.3f}，波高系数 {h.get('hs_scale', float('nan')):.3f}"
                     f"（业务 / 回算 Hs 相对 ERA5 {h.get('operational_hs_over_era5_hs', float('nan')):.3f} / {h.get('reforecast_hs_over_era5_hs', float('nan')):.3f}）")
    lines += ['', f'## 总体（twCRPS 阈值 {t} m）', ''] + md(pd.DataFrame(rows))
    lines += ['', '## 集成相对基准（按 block_id 配对自助 2000 次）', '', '| 分组 | 对照 | 指标 | 变化 | 95% 区间 | 判断 |', '|---|---|---|---:|---|---|']
    groups = [('全部', np.ones(len(y), bool))] + [(f'站点 {s}', (meta.site_id == s).to_numpy()) for s in sorted(meta.site_id.unique())] \
        + [(f'预见期 {b}', (meta.lead_band == b).to_numpy()) for b in sorted(meta.lead_band.unique())]
    for gname, k in groups:
        for ref in ('raw_ensemble', 'emos_ln', 'climatology'):
            for mname, f in (('CRPS', score), (f'twCRPS {t} m', lambda qq, yy: tw(qq, yy, t))):
                if gname != '全部' and mname != 'CRPS':
                    continue
                d, lo, hi = boot(f(allq[ref][k], y[k]), f(q['ensemble'][k], y[k]), blocks[k], rng)
                sig = '显著变好' if hi < 0 else ('显著变差' if lo > 0 else '不显著')
                lines.append(f'| {gname} | {label[ref]} | {mname} | {d:+.1%} | [{lo:+.1%}, {hi:+.1%}] | {sig} |')
    # 2017–2019 reforecast test set, if both frozen runs have been scored there
    paths = [os.path.join(runs[r]['cfg']['output_dir'], 'test', 'quantiles.npz') for r in (a.a, a.b)]
    if all(os.path.exists(x) for x in paths) and samples_orig['test']:
        tt = samples_orig['test']
        yt = np.concatenate([s['y'] for s in tt]).astype(float)
        qe = (np.load(paths[0])[models[0]] + np.load(paths[1])[models[1]]) / 2
        if len(qe) == len(yt):
            mt = pd.DataFrame([m for s in tt for m in s['metadata']])
            bt = mt.block_id.astype(str).to_numpy()
            raw_t = np.concatenate([s['raw_quantiles'] for s in tt]).astype(float)
            em_t = emos_benchmark(samples_orig['train'], tt, names)['ln']
            lines += ['', '## 与回算测试集并列（集成的相对提升；回算测试集按原配置的全部站与原分块）', '', '| 数据 | 对照 | CRPS 变化 | 95% 区间 |', '|---|---|---:|---|']
            for dname, qq, yy, raw, em, bb in (('回算测试集', qe, yt, raw_t, em_t, bt),
                                               (f'业务 {period}', q['ensemble'], y, allq['raw_ensemble'], allq['emos_ln'], blocks)):
                for rname, ref in (('GEFS 原始集合', raw), ('EMOS 对数正态', em)):
                    d, lo, hi = boot(score(ref, yy), score(qq, yy), bb, rng)
                    lines.append(f'| {dname} | {rname} | {d:+.1%} | [{lo:+.1%}, {hi:+.1%}] |')
    lines += ['', '说明：业务与回算的差别（格点插值、第 10–16 天 6 小时插值、业务初值、周期换算）见 download_gefs_operational.py；'
              '夏威夷 Hs 相对 ERA5 业务比回算高约 7%，主结果不校正，--suffix _oper_hsbridge 为敏感性试验。']
    open(os.path.join(out, 'report.md'), 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'\n→ {out}')


if __name__ == '__main__':
    main()
