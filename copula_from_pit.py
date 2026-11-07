"""成对情景的"条件"相关：在给定预报之下，波高与周期的误差是否同向（只用训练集估计）。

joint_scenarios.copula_rho 用的是实测 Hs 与 Te 的无条件（气候）相关；但配对的对象是两个预报分布，
需要的是预报条件下的残差相关。做法：
  1. 用波高模型（--hs-config）和周期多任务模型（--te-config）对训练集样本做预报（各种子平均，用检查点里保存的
     标准化参数，与训练时完全一致）；
  2. PIT：实测值在预报分位数函数中的位置 u = F(y)（在 99 个分位点之间线性插值，截到 [0.005, 0.995]）；
  3. 正态分数 z = Φ⁻¹(u)，按站点 × 月份算 Hs 与 Te 正态分数的相关系数 = 高斯 copula 参数 ρ
     （少于 200 对时用该站全年）。只有有实测 Te 的站（关岛 G、夏威夷 H）。
训练集是样本内预报（比验证集略"准"），只用来估相关；评价仍在验证集上做。
输出：results/joint/copula_rho_pit.csv（site_id, month, rho, n）
用法：python copula_from_pit.py --hs-config GKH_h15_energy_transformer_pretrain --te-config GKH_h15_te_transformer_pretrain
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'outputs', 'wave_forecast'))
from project_paths import resolve_config                                     # noqa: E402
from wavecast.data import Scaler, feature_names                              # noqa: E402
from wavecast.experiment import load_sites, build_all, make_model, tensors, predict, predict_te   # noqa: E402
from wavecast.metrics import TAU                                             # noqa: E402

OUT = os.path.join(ROOT, 'results', 'joint')


def train_predictions(name, split, te=False):
    cfg = json.load(open(resolve_config(os.path.join(ROOT, 'model_configs', f'{name}.json')), encoding='utf-8'))
    model_name = cfg.get('models', ['transformer'])[0]
    ablation = json.load(open(os.path.join(cfg['output_dir'], 'selection.json'), encoding='utf-8'))['best'][model_name]['ablation']
    sites, _ = load_sites(cfg)
    samples, _, _ = build_all(sites, cfg, feature_names(cfg, ablation))
    smp = samples[split]
    qs = []
    for seed in cfg['seeds']:
        state = torch.load(os.path.join(cfg['output_dir'], f'{model_name}_{ablation}_seed{seed}.pt'), map_location='cpu', weights_only=False)
        sc = Scaler(); sc.mean, sc.std = np.array(state['scaler']['mean']), np.array(state['scaler']['std'])
        model = make_model(model_name, state['n_features'], state['history'], state['horizon'], cfg)
        model.load_state_dict(state['model'])
        dl = DataLoader(tensors(smp, sc), batch_size=256)
        qs.append(predict_te(model, dl, 'cpu') if te else predict(model, dl, 'cpu'))
    meta = pd.DataFrame([m for s in smp for m in s['metadata']])[['site_id', 'valid_time']]
    y = np.concatenate([s['y_te'] if te else s['y'] for s in smp])
    print(f'  {name}{"（周期头）" if te else ""}：{split} {len(smp)} 次站点起报，{len(y)} 个时刻', flush=True)
    return np.mean(qs, axis=0), y, meta


def pit(q, y):
    u = np.array([np.interp(v, row, TAU) for v, row in zip(y, q)])
    return np.clip(u, .005, .995)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    p = argparse.ArgumentParser()
    p.add_argument('--hs-config', required=True)
    p.add_argument('--te-config', required=True)
    p.add_argument('--min-pairs', type=int, default=200)
    a = p.parse_args()
    qh, yh, meta = train_predictions(a.hs_config, 'train')
    qt, yt, meta_t = train_predictions(a.te_config, 'train', te=True)
    if len(qh) != len(qt) or not (meta.valid_time.astype(str).values == meta_t.valid_time.astype(str).values).all():
        raise SystemExit('两个模型的训练样本不一致（站点、年份、时效必须相同）')
    ok = np.isfinite(yh) & np.isfinite(yt)
    zh, zt = np.full(len(yh), np.nan), np.full(len(yh), np.nan)
    zh[ok], zt[ok] = norm.ppf(pit(qh[ok], yh[ok])), norm.ppf(pit(qt[ok], yt[ok]))
    d = pd.DataFrame({'site_id': meta.site_id.astype(str), 'month': pd.to_datetime(meta.valid_time, utc=True).dt.month,
                      'zh': zh, 'zt': zt}).dropna()
    rows = []
    for site, g in d.groupby('site_id'):
        whole = g.zh.corr(g.zt)
        for m in range(1, 13):
            gm = g[g.month == m]
            rho = gm.zh.corr(gm.zt) if len(gm) >= a.min_pairs else whole
            rows.append(dict(site_id=site, month=m, rho=round(float(rho), 4), n=len(gm)))
        print(f'  {site}：条件相关（正态分数）全年 {whole:+.2f}，按月 {min(r["rho"] for r in rows if r["site_id"] == site):+.2f} – '
              f'{max(r["rho"] for r in rows if r["site_id"] == site):+.2f}（{len(g)} 对）', flush=True)
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, 'copula_rho_pit.csv')
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f'→ {out}')


if __name__ == '__main__':
    main()
