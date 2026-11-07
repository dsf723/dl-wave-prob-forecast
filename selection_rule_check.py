"""检查挑选规则：原规则（0.5 × 全部 CRPS + 0.5 × 实况极端天气子集 CRPS）与恰当规则（全部样本 CRPS + 阈值加权 CRPS，
与训练损失一致）下，模型 A、B 与两者分位数平均（集成）的排序是否相同（验证集）。

依据：Lerch 等 2017（预报者困境）：只在事后实况为极端的子集上评分不是恰当评分，会偏袒报大的预报；
Allen 等 2023、Wessel 等 2025：用全部样本上的阈值加权 CRPS 强调大浪。
差异的 95% 区间：按 block_id（事件 / 4 周背景块）配对自助 2000 次。
用法：python selection_rule_check.py --a GKH_h15_transformer_pretrain --b GKH_h15_gru_pretrain [--threshold 2.5]
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
from route_models import seed_mean                 # noqa: E402
from wavecast.metrics import score                 # noqa: E402


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    p = argparse.ArgumentParser()
    p.add_argument('--a', default='GKH_h15_transformer_pretrain')
    p.add_argument('--b', default='GKH_h15_gru_pretrain')
    p.add_argument('--threshold', type=float, default=2.5)
    a = p.parse_args()
    run = os.path.join(ROOT, 'model_runs', a.a)
    model = next(iter(json.load(open(os.path.join(run, 'selection.json'), encoding='utf-8'))['best']))
    cs = pd.read_csv(os.path.join(run, 'validation', 'case_scores.csv'),
                     usecols=['site_id', 'model', 'truth', 'typhoon', 'cold_surge', 'block_id'])
    cs = cs[cs.model == model].reset_index(drop=True)
    qa, qb = seed_mean(a.a), seed_mean(a.b)
    if not (len(qa) == len(qb) == len(cs)):
        raise SystemExit('行数不一致')
    y, ext, blk = cs.truth.to_numpy(), (cs.typhoon | cs.cold_surge).to_numpy(), cs.block_id.to_numpy()
    t = a.threshold
    cands = {'A': qa, 'B': qb, '集成': (qa + qb) / 2}
    crps = {k: score(q, y) for k, q in cands.items()}
    tw = {k: score(np.maximum(q, t), np.maximum(y, t)) for k, q in cands.items()}

    def old(k, i):
        return .5 * crps[k][i].mean() + .5 * crps[k][i][ext[i]].mean()

    def new(k, i):
        return crps[k][i].mean() + tw[k][i].mean()

    every = np.arange(len(y))
    print(f'A = {a.a}，B = {a.b}，验证集 {len(y)} 个时刻，实况极端天气 {ext.sum()} 个，阈值 {t} m\n')
    print('| 候选 | 全部 CRPS | 阈值加权 CRPS | 原规则得分 | 恰当规则得分（CRPS + twCRPS） |')
    print('|---|---:|---:|---:|---:|')
    for k in cands:
        print(f'| {k} | {crps[k].mean():.4f} | {tw[k].mean():.4f} | {old(k, every):.4f} | {new(k, every):.4f} |')
    ub = np.unique(blk); idx = {b: np.flatnonzero(blk == b) for b in ub}
    rng = np.random.default_rng(0)
    picks = [np.concatenate([idx[b] for b in rng.choice(ub, len(ub))]) for _ in range(2000)]
    print('\n| 比较（相对 A） | 规则 | 差异 | 95% 区间 |')
    print('|---|---|---:|---|')
    for k in ('B', '集成'):
        for name, f in (('原规则', old), ('恰当规则', new)):
            d = [f(k, i) / f('A', i) - 1 for i in picks]
            lo, hi = np.percentile(d, [2.5, 97.5])
            sig = '显著' if hi < 0 or lo > 0 else '不显著'
            print(f'| {k} | {name} | {f(k, every) / f("A", every) - 1:+.2%} | [{lo:+.2%}, {hi:+.2%}] {sig} |')
    for name, f in (('原规则', old), ('恰当规则', new)):
        print(f'{name}选出：{min(cands, key=lambda k: f(k, every))}')


if __name__ == '__main__':
    main()
