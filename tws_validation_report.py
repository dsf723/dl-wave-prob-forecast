"""台湾海峡模型报告：Transformer + GRU 集成（w = 0.5，与最终模型相同）对 GEFS 原始、EMOS、气候态。默认只看验证集 2015；--split test 只跑一次。

为什么要单写：site_skill.py 只看单个模型，且不和 EMOS 比；GEFS 原始有系统偏差时，相对它的提升可能大部分只是偏差订正，
EMOS（每站 × 每个预见日，用训练年拟合）是这种简单订正的基准。
区间（11-02 改，看测试集之前定）：5 个站相距 100–300 km，同一时段天气相关，每站各算一块会把区间算窄。这里去掉块名的站点前缀，同一 28 天或同一事件的所有站一起抽样。
内容：全部 / 逐站 / 预见期段 / 台风期的 CRPS，集成相对各基准的变化，按 block_id 配对块自助 2000 次的 95% 区间；逐预见日相对气候态；
      总分 S = CRPS + twCRPS(2.5 m) 只算模型（case_scores.csv 没有基准的分位数）。各种子平均后再集成。
输出：屏幕和 results/taiwan_strait/<validation|test>_report.md
用法：python tws_validation_report.py [--split val|test] [--t-dir model_runs/TWS_h15_transformer_pretrain] [--g-dir model_runs/TWS_h15_gru_pretrain] [--out 文件]
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'outputs', 'wave_forecast'))
from wavecast.metrics import TAU, quantile_at, score                         # noqa: E402

BASES = [('raw_ensemble', 'GEFS 原始'), ('emos_tn', 'EMOS 截断正态'), ('emos_ln', 'EMOS 对数正态'), ('climatology', '气候态')]


def seed_mean(run, model):
    files = sorted(glob.glob(os.path.join(run, f'{model}_season_seed*_val.npz')))
    return np.mean([np.load(f)['quantiles'] for f in files], axis=0), len(files)


def boot(a, b, blocks, rng, n=2000):
    df = pd.DataFrame({'a': a, 'b': b, 'k': blocks}).groupby('k').sum()
    idx = rng.integers(0, len(df), (n, len(df)))
    r = df.b.to_numpy()[idx].sum(1) / df.a.to_numpy()[idx].sum(1) - 1
    return df.b.sum() / df.a.sum() - 1, np.percentile(r, 2.5), np.percentile(r, 97.5), len(df)


def tw(q, y):
    return score(np.maximum(q, 2.5), np.maximum(y, 2.5))


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    p = argparse.ArgumentParser()
    p.add_argument('--t-dir', default=os.path.join(ROOT, 'model_runs', 'TWS_h15_transformer_pretrain'))
    p.add_argument('--g-dir', default=os.path.join(ROOT, 'model_runs', 'TWS_h15_gru_pretrain'))
    p.add_argument('--split', choices=['val', 'test'], default='val')
    p.add_argument('--out')
    a = p.parse_args()
    test = a.split == 'test'
    sub, label = ('test', '测试集（2016 至 2017-03）') if test else ('validation', '验证集（2015）')
    out = a.out or os.path.join(ROOT, 'results', 'taiwan_strait', f'{sub}_report.md')
    cs = pd.read_csv(os.path.join(a.t_dir, sub, 'case_scores.csv'), low_memory=False)
    cs['site_id'] = cs.site_id.astype(str)
    ref = cs[cs.model == ('transformer' if test else 'transformer_season')].reset_index(drop=True)
    y = ref.truth.to_numpy(float)
    if test:
        qt, nt = np.load(os.path.join(a.t_dir, 'test', 'quantiles.npz'))['transformer'], 3
        qg, ng = np.load(os.path.join(a.g_dir, 'test', 'quantiles.npz'))['gru'], 3
    else:
        qt, nt = seed_mean(a.t_dir, 'transformer')
        qg, ng = seed_mean(a.g_dir, 'gru')
    # npz 行序必须与 case_scores 一致，否则下面全错
    if not np.allclose(score(qt, y), ref.crps.to_numpy(), atol=1e-4):
        raise SystemExit('Transformer 的 npz 与 case_scores.csv 行序不一致')
    gs = pd.read_csv(os.path.join(a.g_dir, sub, 'case_scores.csv'), usecols=['model', 'truth', 'crps'], low_memory=False)
    gs = gs[gs.model == ('gru' if test else 'gru_season')].reset_index(drop=True)
    if not (np.allclose(gs.truth, y, atol=1e-4) and np.allclose(score(qg, y), gs.crps, atol=1e-4)):
        raise SystemExit('GRU 与 Transformer 的样本不一致')
    q = (qt + qg) / 2
    ens = score(q, y)
    base = {}
    for m, _ in BASES:
        b = cs[cs.model == m].reset_index(drop=True)
        if not (len(b) == len(ref) and np.allclose(b.truth, y, atol=1e-4)):
            raise SystemExit(f'{m} 与模型的样本不一致')
        base[m] = b.crps.to_numpy(float)
    p10, p90 = quantile_at(q, .1), quantile_at(q, .9)
    cov = (y >= p10) & (y <= p90)
    rng = np.random.default_rng(0)
    prefix = re.compile('^(' + '|'.join(map(re.escape, ref.site_id.unique())) + ')_')
    blocks = ref.block_id.astype(str).map(lambda b: prefix.sub('', b)).to_numpy()
    S = {n: score(x, y) + tw(x, y) for n, x in (('t', qt), ('g', qg), ('e', q))}
    dt, dg = boot(S['e'], S['t'], blocks, rng), boot(S['e'], S['g'], blocks, rng)
    L = [f'# 台湾海峡模型：{label}报告', '',
         f'Transformer {nt} 个种子、GRU {ng} 个种子，各自平均后 w = 0.5 集成。{len(y):,} 个时刻，{ref.groupby(["site_id", "issue_time"]).ngroups} 个站点起报。' + ('' if test else '测试集未使用。'), '',
         f'总分 S = CRPS + twCRPS(2.5 m)：Transformer {(score(qt, y) + tw(qt, y)).mean():.4f}，GRU {(score(qg, y) + tw(qg, y)).mean():.4f}，集成 {(ens + tw(q, y)).mean():.4f}（CRPS {ens.mean():.4f}，twCRPS {tw(q, y).mean():.4f}）。', '',
         f'总分相对集成：Transformer {dt[0]:+.1%} [{dt[1]:+.1%}, {dt[2]:+.1%}]，GRU {dg[0]:+.1%} [{dg[1]:+.1%}, {dg[2]:+.1%}]（正 = 单模型比集成差）。', '',
         '## 集成相对各基准的 CRPS 变化（负 = 集成更好，95% 区间来自配对块自助）', '',
         '| 分组 | 时刻 | 块数 | 集成 | 覆盖率 | ' + ' | '.join(f'{n}（变化，区间）' for _, n in BASES) + ' |',
         '|---|---:|---:|---:|---:|' + '---|' * len(BASES)]
    groups = [('全部', np.ones(len(y), bool))] + [(s, (ref.site_id == s).to_numpy()) for s in sorted(ref.site_id.unique())]
    groups += [(x, (ref.lead_band == x).to_numpy()) for x in sorted(ref.lead_band.unique())]
    groups += [('台风期', ref.typhoon.astype(bool).to_numpy()), ('实测 ≥ 2.5 m', y >= 2.5)]
    for name, k in groups:
        if k.sum() == 0:
            continue
        cells, nb = [], 0
        for m, _ in BASES:
            d, lo, hi, nb = boot(base[m][k], ens[k], blocks[k], rng)
            cells.append(f'{base[m][k].mean():.3f}，{d:+.1%} [{lo:+.1%}, {hi:+.1%}]')
        L.append(f'| {name} | {k.sum():,} | {nb} | {ens[k].mean():.3f} | {cov[k].mean():.0%} | ' + ' | '.join(cells) + ' |')
    L += ['', '## 逐预见日（集成相对气候态，看技巧到第几天还在）', '', '| 预见日 | 集成 | GEFS 原始 | EMOS 截断正态 | 气候态 | 相对气候态 | 95% 区间 | 覆盖率 |', '|---:|---:|---:|---:|---:|---:|---|---:|']
    for day in sorted(ref.lead_day.unique()):
        k = (ref.lead_day == day).to_numpy()
        d, lo, hi, _ = boot(base['climatology'][k], ens[k], blocks[k], rng)
        L.append(f"| {day} | {ens[k].mean():.3f} | {base['raw_ensemble'][k].mean():.3f} | {base['emos_tn'][k].mean():.3f} | {base['climatology'][k].mean():.3f} | {d:+.1%} | [{lo:+.1%}, {hi:+.1%}] | {cov[k].mean():.0%} |")
    aud = os.path.join(a.t_dir, 'data_audit.json')
    if os.path.exists(aud) and not test:
        n = json.load(open(aud, encoding='utf-8')).get('ablations', {}).get('season', {}).get('n_samples', {})
        L += ['', f"站点起报数：训练 {n.get('train')}，验证 {n.get('val')}。"]
    text = '\n'.join(L) + '\n'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, 'w', encoding='utf-8').write(text)
    print(text)


if __name__ == '__main__':
    main()
