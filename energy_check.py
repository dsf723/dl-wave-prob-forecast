"""能量相关的验证检查：波高模型对 Hs² 的预报好不好（波能流 ∝ Hs²·Te）。

对一个或多个训练运行，读验证集上各种子平均的 Hs 分位数（<run>/<模型>_<输入组>_seed*_val.npz）和真值
（validation/case_scores.csv 里同一模型的 truth 列，行顺序与分位数一致），按站点给出：
  Hs 的 CRPS、Hs² 的 CRPS（分位数平方即 Hs² 的分位数）、平均 Hs 偏差、E[Hs²] 偏差。
（不按实测值分组看偏差：按"实测大浪"挑样本必然显得低估，属于 forecaster's dilemma。）
用法：python energy_check.py model_runs/GKH_h15_transformer_pretrain model_runs/GKH_h15_energy_transformer_pretrain
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs', 'wave_forecast'))
from wavecast.metrics import score   # noqa: E402


def load(run):
    best = json.load(open(os.path.join(run, 'selection.json'), encoding='utf-8'))['best']
    model, info = next(iter(best.items()))
    label = f"{model}_{info['ablation']}"
    seeds = json.load(open(os.path.join(run, 'config_used.json'), encoding='utf-8'))['seeds']
    q = np.mean([np.load(os.path.join(run, f'{label}_seed{s}_val.npz'))['quantiles'] for s in seeds], axis=0)
    cs = pd.read_csv(os.path.join(run, 'validation', 'case_scores.csv'), usecols=['site_id', 'model', 'truth'])
    cs = cs[cs.model == label].reset_index(drop=True)
    if len(cs) != len(q):
        raise SystemExit(f'{run}：分位数 {len(q)} 行与 case_scores {len(cs)} 行不一致')
    return q, cs.truth.to_numpy(), cs.site_id.astype(str).to_numpy()


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print('| 运行 | 站点 | Hs CRPS | Hs² CRPS | 平均 Hs 偏差 | E[Hs²] 偏差 |')
    print('|---|---|---:|---:|---:|---:|')
    for run in sys.argv[1:]:
        q, y, site = load(run)
        for s in ['全部'] + sorted(set(site)):
            k = np.ones(len(y), bool) if s == '全部' else site == s
            print(f'| {os.path.basename(run.rstrip("/"))} | {s} | {score(q[k], y[k]).mean():.4f} | {score(q[k] ** 2, y[k] ** 2).mean():.4f} | '
                  f'{q[k].mean() / y[k].mean() - 1:+.1%} | {(q[k] ** 2).mean() / (y[k] ** 2).mean() - 1:+.1%} |')


if __name__ == '__main__':
    main()
