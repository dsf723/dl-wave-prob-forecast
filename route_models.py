"""按"起报时已知的台风信息"在两个模型之间切换，验证集模拟（只用发报时能知道的信息，不用事后的实况台风标签）。

平时用模型 B（如 GRU），满足条件时改用模型 A（如 Transformer）。条件两种：
  issue：起报前 24 小时内台风中心在站点 1000 km 内，或出现寒潮（train_extreme_specialist.extreme_flags，整次起报切换）；
  tigge：ECMWF 台风路径预报认为该预见时刻 500 km 内有台风的成员比例 > 0（逐时刻切换；需 site_*_reforecast_tc）。
同时给出两模型分位数平均的集成。比较全部、各站、事后实况台风期（仅用于评价，不用于切换）、预见期段；
相对 A（Transformer）的变化及按 block_id 的配对自助法 95% 区间。
用法：python route_models.py --a GKH_tuned_transformer_pretrain --b GKH_tuned_gru_pretrain
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
from project_paths import resolve_config                    # noqa: E402
from wavecast.data import feature_names                     # noqa: E402
from wavecast.experiment import load_sites, build_all       # noqa: E402
from wavecast.metrics import score                          # noqa: E402
from train_extreme_specialist import extreme_flags          # noqa: E402


def seed_mean(run):
    best = json.load(open(os.path.join(ROOT, 'model_runs', run, 'selection.json'), encoding='utf-8'))['best']
    model, info = next(iter(best.items()))
    seeds = json.load(open(os.path.join(ROOT, 'model_runs', run, 'config_used.json'), encoding='utf-8'))['seeds']
    return np.mean([np.load(os.path.join(ROOT, 'model_runs', run, f"{model}_{info['ablation']}_seed{s}_val.npz"))['quantiles']
                    for s in seeds], axis=0)


def boot(a, b, blocks, rng, n=2000):
    df = pd.DataFrame({'a': a, 'b': b, 'k': blocks}).groupby('k').sum()
    idx = rng.integers(0, len(df), (n, len(df)))
    r = df.b.to_numpy()[idx].sum(1) / df.a.to_numpy()[idx].sum(1) - 1
    return df.b.sum() / df.a.sum() - 1, np.percentile(r, 2.5), np.percentile(r, 97.5)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    p = argparse.ArgumentParser()
    p.add_argument('--a', default='GKH_tuned_transformer_pretrain')
    p.add_argument('--b', default='GKH_tuned_gru_pretrain')
    a = p.parse_args()
    cfg = json.load(open(resolve_config(os.path.join(ROOT, 'model_configs', f'{a.a}.json')), encoding='utf-8'))
    sites, _ = load_sites(cfg)
    smp = build_all(sites, cfg, feature_names(cfg, 'season'))[0]['val']
    meta = pd.DataFrame([m for s in smp for m in s['metadata']])
    y = np.concatenate([s['y'] for s in smp])
    qa, qb = seed_mean(a.a), seed_mean(a.b)
    if not (len(qa) == len(qb) == len(y)):
        raise SystemExit('两个运行的验证集行数不一致')
    flags, _ = extreme_flags(sites, smp, cfg, 1000)
    issue_flag = np.repeat(flags, [len(s['y']) for s in smp])
    # TIGGE forecast flag per row (Wednesday reforecast datasets with track features)
    tg = []
    for c, _, _ in sites:
        f = pd.read_csv(os.path.join(ROOT, 'processed', f"site_{c['site_id']}_reforecast_tc", 'forecasts.csv'),
                        usecols=['issue_time', 'valid_time', 'tc_p500'])
        f['site_id'] = str(c['site_id']); tg.append(f)
    tg = pd.concat(tg)
    for col in ('issue_time', 'valid_time'):
        tg[col] = pd.to_datetime(tg[col], utc=True)
    key = meta[['site_id', 'issue_time', 'valid_time']].copy()
    key['site_id'] = key.site_id.astype(str)
    for col in ('issue_time', 'valid_time'):
        key[col] = pd.to_datetime(key[col], utc=True)
    tig_flag = (key.merge(tg, on=['site_id', 'issue_time', 'valid_time'], how='left').tc_p500.fillna(0) > 0).to_numpy()
    sa, sb = score(qa, y), score(qb, y)
    variants = {'A 单独（Transformer）': sa, 'B 单独（GRU）': sb,
                '切换：起报时台风/寒潮 → A，否则 B': np.where(issue_flag, sa, sb),
                '切换：TIGGE 预报有台风 → A，否则 B': np.where(tig_flag, sa, sb),
                '集成：A、B 分位数平均': score((qa + qb) / 2, y)}
    print(f'验证集 {len(y)} 个时刻；起报时有台风/寒潮的时刻占 {issue_flag.mean():.1%}，TIGGE 预报有台风的时刻占 {tig_flag.mean():.1%}\n')
    groups = [('全部', np.ones(len(y), bool))] + [(f'站点 {s}', (meta.site_id.astype(str) == s).to_numpy()) for s in ['G', 'K', 'H']]
    groups += [('事后实况台风期（仅评价）', meta.typhoon.astype(bool).to_numpy())]
    groups += [(f'预见期 {b}', (meta.lead_band == b).to_numpy()) for b in sorted(meta.lead_band.unique())]
    rng = np.random.default_rng(0)
    head = '| 方案 | ' + ' | '.join(g for g, _ in groups) + ' |'
    print(head); print('|---|' + '---:|' * len(groups))
    for name, s in variants.items():
        cells = []
        for g, k in groups:
            if name.startswith('A 单独'):
                cells.append(f'{s[k].mean():.4f}')
            else:
                rel, lo, hi = boot(sa[k], s[k], meta.block_id.to_numpy()[k], rng)
                cells.append(f"{s[k].mean():.4f}（{rel:+.1%}{'*' if hi < 0 or lo > 0 else ''}）")
        print(f'| {name} | ' + ' | '.join(cells) + ' |')
    print('\n括号内为相对 A 单独的变化，* = 配对自助法 95% 区间不含 0（显著）。')


if __name__ == '__main__':
    main()
