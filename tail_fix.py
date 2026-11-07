"""大浪上尾偏轻的事后修正（不重新训练），验证集内按年份留一交叉验证。

诊断（postproc_diagnostics.py）：实测 > 2.5 m 时条件 PIT 最高一成 0.17（理想 0.10）；模型报'超 2.5 m 概率 > 0.8'时
实际发生 96%（报 83%）→ 模型自己报大浪时，上尾还不够重。
方案（只用发报时可知的信息分组，不按实况分组）：
  A 按预报分档的共形校正：按 预见期段 × 预报中位数档（< 1、1–1.5、1.5–2、2–2.5、≥ 2.5 m）逐分位数单侧 CQR 平移；
  B 与 EMOS 对数正态线性混合（Wessel 2025）：(1−w)·模型 + w·EMOS，w 在其余年份上按 CRPS + twCRPS 选（嵌套）；
  A+B：先 B 再 A。
事先定的采纳标准：大浪 twCRPS（2.5 m）显著变好，且全部 CRPS 不显著变差（按 block_id 配对自助 2000 次）。
用法：python tail_fix.py --run GKH_h15_transformer_pretrain（需先跑 postproc_diagnostics.py 生成 EMOS 分位数）
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'outputs', 'wave_forecast'))
from route_models import seed_mean, boot                                     # noqa: E402
from wavecast.metrics import TAU, score                                      # noqa: E402
from wavecast.postproc import conformal_shifts, apply_shifts, linear_pool, conditional_pit, cdf_at  # noqa: E402

P50_BINS = [-1, 1, 1.5, 2, 2.5, 99]
W_GRID = [0., .1, .2, .3, .4, .5]
OUT = os.path.join(ROOT, 'results', 'postproc')


def tw(q, y, t):
    return score(np.maximum(q, t), np.maximum(y, t))


def bin_keys(q, band):
    b = np.digitize(q[:, 49], P50_BINS[1:-1])
    return np.array([f'{x}|p50bin{i}' for x, i in zip(band, b)])


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    p = argparse.ArgumentParser()
    p.add_argument('--run', default='GKH_h15_transformer_pretrain')
    p.add_argument('--threshold', type=float, default=2.5)
    a = p.parse_args()
    cs = pd.read_csv(os.path.join(ROOT, 'model_runs', a.run, 'validation', 'case_scores.csv'))
    best = json.load(open(os.path.join(ROOT, 'model_runs', a.run, 'selection.json'), encoding='utf-8'))['best']
    model, info = next(iter(best.items()))
    cs = cs[cs.model == f"{model}_{info['ablation']}"].reset_index(drop=True)
    q0 = seed_mean(a.run)
    em = np.load(os.path.join(OUT, f'{a.run}_val_emos.npz'))['ln']
    y, t = cs.truth.to_numpy(float), a.threshold
    if not (len(q0) == len(em) == len(y)):
        raise SystemExit('行数不一致')
    year = pd.to_datetime(cs.issue_time, utc=True).dt.year.to_numpy()
    band = cs.lead_band.to_numpy()
    blocks = cs.block_id.to_numpy()
    print('计算线性混合…', flush=True)
    pools = {w: (q0 if w == 0 else linear_pool(q0, em, w)) for w in W_GRID}

    def obj(q, k):
        return score(q[k], y[k]).mean() + tw(q[k], y[k], t).mean()

    cands = {'原模型': q0, 'A 分档共形': np.array(q0), 'B 混合 EMOS': np.array(q0), 'A+B': np.array(q0)}
    chosen = []
    for yr in np.unique(year):
        k = year == yr
        sh = conformal_shifts(q0[~k], y[~k], bin_keys(q0[~k], band[~k]))
        cands['A 分档共形'][k] = apply_shifts(q0[k], bin_keys(q0[k], band[k]), sh)
        w = min(W_GRID, key=lambda w: obj(pools[w], ~k))
        chosen.append((yr, w))
        cands['B 混合 EMOS'][k] = pools[w][k]
        qb = pools[w]
        sh = conformal_shifts(qb[~k], y[~k], bin_keys(qb[~k], band[~k]))
        cands['A+B'][k] = apply_shifts(qb[k], bin_keys(qb[k], band[k]), sh)
    rng = np.random.default_rng(0)
    big = y > t
    pf = {n: 1 - cdf_at(q, t) for n, q in cands.items()}
    rows = []
    for n, q in cands.items():
        c, _ = conditional_pit(q, y, t, rng)
        hi = pf[n] > .8
        rows.append(dict(方案=n, CRPS=score(q, y).mean(), twCRPS_25=tw(q, y, t).mean(), twCRPS_35=tw(q, y, 3.5).mean(),
                         覆盖80=((y >= q[:, 9]) & (y <= q[:, 89])).mean(), 大浪条件PIT最高一成=(c > .9).mean(),
                         大浪条件PIT最低一成=(c < .1).mean(), 报超25概率gt08_平均预报=pf[n][hi].mean(),
                         报超25概率gt08_实际=big[hi].mean(), 报超25概率gt08_个数=int(hi.sum())))
    res = pd.DataFrame(rows)
    lines = [f'# 大浪上尾修正（{a.run}，验证集按年份留一）', '', f'各年选出的混合权重 w：{chosen}', '',
             '| ' + ' | '.join(res.columns) + ' |', '|' + '---|' * len(res.columns)]
    for _, r in res.iterrows():
        lines.append('| ' + ' | '.join(f'{v:.4f}' if isinstance(v, float) else str(v) for v in r) + ' |')
    lines += ['', '相对原模型（配对自助 2000 次）：', '', '| 方案 | 指标 | 变化 | 95% 区间 | 判断 |', '|---|---|---:|---|---|']
    s0, w0 = score(q0, y), tw(q0, y, t)
    for n, q in cands.items():
        if n == '原模型':
            continue
        verdict = []
        for name, base, new in (('CRPS', s0, score(q, y)), ('twCRPS 2.5 m', w0, tw(q, y, t)), ('twCRPS 3.5 m', tw(q0, y, 3.5), tw(q, y, 3.5))):
            d, lo, hi = boot(base, new, blocks, rng)
            sig = '显著变好' if hi < 0 else ('显著变差' if lo > 0 else '不显著')
            verdict.append((name, sig))
            lines.append(f'| {n} | {name} | {d:+.2%} | [{lo:+.2%}, {hi:+.2%}] | {sig} |')
        ok = dict(verdict)['twCRPS 2.5 m'] == '显著变好' and dict(verdict)['CRPS'] != '显著变差'
        lines.append(f'| {n} | **按事先标准** | | | {"采纳" if ok else "不采纳"} |')
    path = os.path.join(OUT, f'{a.run}_tail_fix.md')
    open(path, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
