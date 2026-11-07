"""按站点选模型：每个站用同一条事先声明的挑选规则（0.5 × 全部 CRPS + 0.5 × 极端天气 CRPS，验证集），
在"模型 A 单独"和"A、B 分位数平均（集成）"之间选；同时给出训练年份的台风频率（500 km 内的小时占比，事先可知）。

想法：Transformer 强在台风期，集成强在平时 → 台风多的站用 Transformer，台风少的站用集成。
差异的 95% 区间：按 block_id（事件 / 4 周背景块）的配对自助法（2000 次），对规则得分之差重抽样。
用法：python site_model_choice.py --a GKH_h15_transformer_pretrain --b GKH_h15_gru_pretrain
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


def objective(s, ext):
    return .5 * s.mean() + .5 * s[ext].mean() if ext.any() else s.mean()


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    p = argparse.ArgumentParser()
    p.add_argument('--a', default='GKH_h15_transformer_pretrain')
    p.add_argument('--b', default='GKH_h15_gru_pretrain')
    a = p.parse_args()
    run = os.path.join(ROOT, 'model_runs', a.a)
    cfg = json.load(open(os.path.join(run, 'config_used.json'), encoding='utf-8'))
    model = next(iter(json.load(open(os.path.join(run, 'selection.json'), encoding='utf-8'))['best']))
    cs = pd.read_csv(os.path.join(run, 'validation', 'case_scores.csv'),
                     usecols=['site_id', 'model', 'truth', 'typhoon', 'cold_surge', 'block_id'])
    cs = cs[cs.model == model].reset_index(drop=True)
    qa, qb = seed_mean(a.a), seed_mean(a.b)
    if not (len(qa) == len(qb) == len(cs)):
        raise SystemExit('行数不一致')
    y = cs.truth.to_numpy()
    sa, se = score(qa, y), score((qa + qb) / 2, y)
    rng = np.random.default_rng(0)
    print(f'A = {a.a}，集成 = A 与 {a.b} 的分位数平均（验证集）\n')
    print('| 站点 | 训练年台风频率 | 台风期时刻 | 规则得分 A | 规则得分 集成 | 集成相对 A | 95% 区间 | 按规则选 |')
    print('|---|---:|---:|---:|---:|---:|---|---|')
    for s in cfg['sites']:
        sid = str(s['site_id'])
        o = pd.read_csv(os.path.join(ROOT, 'processed', f'site_{sid}_reforecast', 'observations.csv'),
                        usecols=['valid_time', 'label_tc_distance_km'])
        yr = pd.to_datetime(o.valid_time, utc=True).dt.year
        freq = (o.label_tc_distance_km[yr.isin(s['train_years'])] <= 500).mean()
        k = (cs.site_id.astype(str) == sid).to_numpy()
        ext = (cs.typhoon | cs.cold_surge).to_numpy()[k]
        A, E, blk = sa[k], se[k], cs.block_id.to_numpy()[k]
        oa, oe = objective(A, ext), objective(E, ext)
        ub = np.unique(blk); idx = {b: np.flatnonzero(blk == b) for b in ub}
        diffs = []
        for _ in range(2000):
            pick = np.concatenate([idx[b] for b in rng.choice(ub, len(ub))])
            diffs.append(objective(E[pick], ext[pick]) / objective(A[pick], ext[pick]) - 1)
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        sig = '显著' if hi < 0 or lo > 0 else '不显著'
        print(f'| {sid} | {freq:.1%} | {ext.sum()} | {oa:.4f} | {oe:.4f} | {oe / oa - 1:+.1%} | [{lo:+.1%}, {hi:+.1%}] {sig} | '
              f'{"集成" if oe < oa else "A"} |')


if __name__ == '__main__':
    main()
