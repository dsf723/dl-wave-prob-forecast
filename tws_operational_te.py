"""台湾海峡周期 Te 的独立业务检验（11-04 事先登记的第 3 条）：冻结的 TWS2 周期多任务模型在 2020-09-23 至 2024-12-31 业务 GEFS 上，新竹一站。

真值：processed/buoy_spectral_cwa/CW46757B_46757B.csv（气象署平均周期 ÷ 0.863，k 只用 2000–2015 拟合；平均偏差按构造为 0，所以只检验起伏和时间变化）。
对照：GEFS 业务 MWSPER 按 ERA5 桥梁换算的 Te（点预报，看绝对误差）、训练年同日 ±15 天的 Te 气候态。不训练、不挑选。
区间：按 block_id 配对块自助 2000 次（一站，块名本来就是这一站的）。
输出：results/operational/TWS2_h15_te_transformer_pretrain_oper_te/report.md
用法：python tws_operational_te.py [--run TWS2_h15_te_transformer_pretrain --suffix _oper --years 2020 2024 --sites CW46757B]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'outputs', 'wave_forecast'))
import torch                                                                 # noqa: E402
from torch.utils.data import DataLoader                                      # noqa: E402
from evaluate_operational import frozen, operational_cfg                     # noqa: E402
from route_models import boot                                                # noqa: E402
from wavecast.data import feature_names, Scaler                              # noqa: E402
from wavecast.experiment import load_sites, build_all, make_model, predict_te, tensors  # noqa: E402
from wavecast.metrics import TAU, score                                      # noqa: E402


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    p = argparse.ArgumentParser()
    p.add_argument('--run', default='TWS2_h15_te_transformer_pretrain')
    p.add_argument('--suffix', default='_oper')
    p.add_argument('--years', nargs=2, type=int, default=[2020, 2024])
    p.add_argument('--sites', nargs='+', default=['CW46757B'])
    a = p.parse_args()
    cfg, model, ablation = frozen(a.run)
    op = operational_cfg(cfg, a.suffix, list(range(a.years[0], a.years[1] + 1)), a.sites)
    layout = [str(s['site_id']) for s in cfg['sites']]
    names = feature_names(cfg, ablation)
    sites_op, _ = load_sites(op)
    test = build_all(sites_op, op, names, site_ids=layout)[0]['test']
    sites_tr, _ = load_sites(cfg)
    train = build_all(sites_tr, cfg, names)[0]['train']
    names_full = build_all(sites_op, op, names, site_ids=layout)[2]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    qs = []
    for seed in cfg['seeds']:
        state = torch.load(os.path.join(cfg['output_dir'], f'{model}_{ablation}_seed{seed}.pt'), map_location=device, weights_only=False)
        sc = Scaler(); sc.mean = np.array(state['scaler']['mean']); sc.std = np.array(state['scaler']['std'])
        net = make_model(model, state['n_features'], state['history'], state['horizon'], cfg).to(device)
        net.load_state_dict(state['model'])
        qs.append(predict_te(net, DataLoader(tensors(test, sc), batch_size=cfg['batch_size']), device))
    q = np.mean(qs, axis=0)
    meta = pd.DataFrame([m for s in test for m in s['metadata']])
    y = np.concatenate([s['y_te'] for s in test]).astype(float)
    raw = np.concatenate([s['future'][:, names_full.index('te')] for s in test]).astype(float)
    tr = pd.DataFrame([dict(site=str(m['site_id']), t=m['valid_time']) for s in train for m in s['metadata']])
    tr['y'] = np.concatenate([s['y_te'] for s in train]); tr = tr.dropna()
    tr['doy'] = pd.to_datetime(tr.t, utc=True).dt.dayofyear
    doy = pd.to_datetime(meta.valid_time, utc=True).dt.dayofyear.to_numpy()
    clim = np.full((len(meta), 99), np.nan)
    for site, g in tr.groupby('site'):
        k = (meta.site_id.astype(str) == site).to_numpy()
        table = {d: np.quantile(g.y[np.minimum(abs(g.doy - d), 366 - abs(g.doy - d)) <= 15], TAU) for d in np.unique(doy[k])}
        if k.any():
            clim[k] = np.stack([table[d] for d in doy[k]])
    ok = np.isfinite(y) & np.isfinite(raw) & np.isfinite(clim).all(1)
    q, y, raw, clim, meta = q[ok], y[ok], raw[ok], clim[ok], meta[ok].reset_index(drop=True)
    blocks = meta.block_id.astype(str).to_numpy()
    rng = np.random.default_rng(0)
    cm, cc, ae = score(q, y), score(clim, y), np.abs(raw - y)
    p10, p90 = np.array([np.interp(.1, TAU, r) for r in q]), np.array([np.interp(.9, TAU, r) for r in q])
    L = [f'# 周期 Te 独立业务检验：{a.run}（{", ".join(a.sites)}）', '',
         f'业务起报 {meta.issue_time.min():%Y-%m-%d} 至 {meta.issue_time.max():%Y-%m-%d}，{meta.issue_time.nunique()} 次，{len(y):,} 个有 Te 真值的时刻。'
         '真值 = 气象署平均周期 ÷ 0.863（平均偏差按构造为 0）。模型冻结，不训练、不挑选。', '',
         '| 分组 | 时刻 | 模型 CRPS (s) | GEFS Te 绝对误差 (s) | 气候态 CRPS (s) | 模型相对 GEFS（区间） | 模型相对气候态（区间） | P10–P90 覆盖 | GEFS 平均偏差 (s) |',
         '|---|---:|---:|---:|---:|---|---|---:|---:|']
    for gname, k in [('全部', np.ones(len(y), bool))] + [(b, (meta.lead_band == b).to_numpy()) for b in sorted(meta.lead_band.unique())]:
        d1 = boot(ae[k], cm[k], blocks[k], rng); d2 = boot(cc[k], cm[k], blocks[k], rng)
        L.append(f'| {gname} | {k.sum():,} | {cm[k].mean():.3f} | {ae[k].mean():.3f} | {cc[k].mean():.3f} | {d1[0]:+.1%} [{d1[1]:+.1%}, {d1[2]:+.1%}] | '
                 f'{d2[0]:+.1%} [{d2[1]:+.1%}, {d2[2]:+.1%}] | {((y[k] >= p10[k]) & (y[k] <= p90[k])).mean():.0%} | {(raw[k] - y[k]).mean():+.2f} |')
    L += ['', '注：GEFS 一列是点预报，CRPS 退化为绝对误差，与模型 CRPS 直接可比。']
    out = os.path.join(ROOT, 'results', 'operational', f'{a.run}{a.suffix}_te')
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, 'report.md'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    print('\n'.join(L))
    print(f'\n→ {out}')


if __name__ == '__main__':
    main()
