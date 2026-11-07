"""未来一个月（30 天）平均波浪能能不能预报：用 30 天产品（第 1–15 天模型 + 第 16–30 天气候态）求时段平均波能流，
与实测的时段平均比较，并与"只用历史气候态"比较。

预报：WEF_mean_kWm（预报分布的均值）在时段内平均；实测：WEF_observed_kWm 在时段内平均（有实测的时刻 ≥ 80% 才算）；
气候态：该站训练年份实测波能流（浮标 Hs × 实测 Te，无谱的站用 ERA5 Te）在同日期 ±15 天的平均，同样在时段内平均。
时段：第 1–30 天（"未来一个月"）、第 1–15 天、第 16–30 天。
指标：平均绝对误差（kW/m 和相对实测平均的百分比）、技巧 = 1 − 预报误差 / 气候态误差（> 0 说明比气候态好）、
距平相关（预报 − 气候态 与 实测 − 气候态 的相关）。只用验证集（或已评过的测试集）。
年平均与 MVI 需要未来 12 个月，超出 30 天产品的范围，只能给历史期望与波动范围（见 historical_wave_energy.py）。
用法：python forecast_monthly_energy.py results/forecast_outputs/GKH_h15_transformer_pretrain_val_pitcop_30d.csv
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
FLUX = 1025.0 * 9.81 ** 2 / (64 * np.pi) / 1000
CFG = os.path.join(ROOT, 'model_configs', 'GKH_h15_transformer_pretrain.json')


def clim_table(sid, train_years):
    b = pd.read_csv(glob.glob(os.path.join(ROOT, 'processed', 'buoy', f'{sid}_*.csv'))[0], usecols=['valid_time', 'hs'])
    b.index = pd.to_datetime(b.valid_time, utc=True)
    hs = b.hs.groupby(b.index.round('1h')).mean()
    spec = glob.glob(os.path.join(ROOT, 'processed', 'buoy_spectral', f'{sid}_*.csv'))
    if spec:
        t = pd.read_csv(spec[0], usecols=['valid_time', 'te']); te = pd.Series(t.te.values, index=pd.to_datetime(t.valid_time, utc=True))
    else:
        e = pd.read_csv(os.path.join(ROOT, 'processed', f'site_{sid}', 'observations_hourly.csv'), usecols=['valid_time', 'te'])
        te = pd.Series(e.te.values, index=pd.to_datetime(e.valid_time, utc=True))
    w = (FLUX * hs ** 2 * te.reindex(hs.index)).dropna()
    w = w[w.index.year.isin(train_years)]
    dy = w.index.dayofyear.to_numpy()
    return {d: w[np.minimum(abs(dy - d), 366 - abs(dy - d)) <= 15].mean() for d in range(1, 367)}


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    path = sys.argv[1]
    cfg = json.load(open(CFG, encoding='utf-8'))
    d = pd.read_csv(path)
    d['vt'] = pd.to_datetime(d.valid_time, utc=True)
    tables = {s['site_id']: clim_table(s['site_id'], s['train_years']) for s in cfg['sites']}
    d['clim'] = [tables[s][t] for s, t in zip(d.site_id, d.vt.dt.dayofyear)]
    rows = []
    for (site, issue), g in d.groupby(['site_id', 'issue_time']):
        for name, lo, hi in (('第1–30天', 1, 30), ('第1–15天', 1, 15), ('第16–30天', 16, 30)):
            w = g[g.lead_day.between(lo, hi)]
            ok = w.WEF_observed_kWm.notna()
            if len(w) == 0 or ok.mean() < .8:
                continue
            rows.append(dict(site=site, issue=issue, window=name, fc=w.WEF_mean_kWm[ok].mean(), obs=w.WEF_observed_kWm[ok].mean(),
                             clim=w.clim[ok].mean()))
    r = pd.DataFrame(rows)
    print(f'{os.path.basename(path)}：按起报的时段平均波能流（kW/m）\n')
    print('| 站点 | 时段 | 起报数 | 实测平均 | 预报误差 | 气候态误差 | 技巧（>0 好于气候态） | 距平相关 |')
    print('|---|---|---:|---:|---:|---:|---:|---:|')
    for site in sorted(r.site.unique()):
        for name in ('第1–30天', '第1–15天', '第16–30天'):
            g = r[(r.site == site) & (r.window == name)]
            if g.empty:
                continue
            ef, ec = (g.fc - g.obs).abs().mean(), (g.clim - g.obs).abs().mean()
            corr = np.corrcoef(g.fc - g.clim, g.obs - g.clim)[0, 1] if (g.fc - g.clim).std() > 1e-9 else float('nan')
            print(f'| {site} | {name} | {len(g)} | {g.obs.mean():.1f} | {ef:.2f}（{ef / g.obs.mean():.0%}） | {ec:.2f}（{ec / g.obs.mean():.0%}） | '
                  f'{1 - ef / ec:+.2f} | {corr:.2f} |')


if __name__ == '__main__':
    main()
