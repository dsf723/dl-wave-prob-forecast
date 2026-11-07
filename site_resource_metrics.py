"""各站点波浪能资源与出力稳定性指标（与 Jin et al. 2025, Energy 323, 135730 的定义一致）。

指标：
  MVI 月变化指数 = (最大月平均功率 − 最小月平均功率) / 年平均功率      （多年逐月平均，Jin 2025 式 17）
  CV  变异系数   = 功率标准差 / 平均功率                               （逐时，Jin 2025 式 18）
  停机率        = 输出为零的小时占比                                  （Jin 2025 3.3.4 节）
  CF  容量因子   = 平均输出 / 额定功率                                 （只有给了装置功率矩阵才算，Jin 2025 式 16）
  另给出逐年 MVI 的均值 ± 标准差（年际差异）。
年际变化（10-21 新增）：每个站每年的平均功率、MVI、CV（yearly_metrics.csv），30 年线性趋势（每十年变化占多年平均的
  百分比，及线性回归的 p 值，p < 0.05 记为显著；trends.csv），以及逐年平均功率与逐年 MVI 的时间序列图（yearly_power_mvi.png）。

功率的两种算法：
  1) 默认：理论波能流 P = ρg²/(64π)·Hs²·Te ≈ 0.49·Hs²·Te（kW/m，深水），装置工作范围用 --hs-cutin（默认 0.5 m，
     以下输出为零）和 --hs-cutout（默认不设，超过则停机保护）近似；
  2) --power-matrix 文件.csv：装置功率矩阵（第一列为 Hs 档中值 m，表头为周期档中值 s，单元格为输出 kW），
     周期用 --matrix-period te|tp 指定（tp 时用 Tp ≈ Te / 0.9 近似，JONSWAP 谱的典型比值；本项目实测 Te/Tp 在
     0.55–1.04 之间变化，近似会带来误差，报告里要说明），矩阵外输出为零；额定功率取矩阵最大值。
数据：processed/site_<ID>/observations_hourly.csv（CDS ERA5，1995–2024）。注意 ERA5 在近岸（A/B/C）系统低估大浪，
     功率会被低估。
输出：results/resource_metrics/site_metrics.csv、monthly_climatology.csv、yearly_mvi.csv、monthly_power.png、report.md
用法：python site_resource_metrics.py [--sites A B C G K H] [--hs-cutin 0.5] [--hs-cutout 8] [--power-matrix 矩阵.csv --matrix-period tp]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
SITES = {s['id']: s for s in json.load(open(os.path.join(ROOT, 'metadata', 'sites.json'), encoding='utf-8'))['sites']}
OUT = os.path.join(ROOT, 'results', 'resource_metrics')
NAMES = {'A': '舟山', 'B': '珠海', 'C': '平潭', 'G': '关岛', 'K': '马罗岛', 'H': '夏威夷'}
RHO, G = 1025.0, 9.81
FLUX = RHO * G ** 2 / (64 * np.pi) / 1000          # kW/m per (m² s) ≈ 0.4906


def load_matrix(path):
    m = pd.read_csv(path, index_col=0)
    hs = m.index.to_numpy(float)
    t = np.array([float(c) for c in m.columns])
    return hs, t, m.to_numpy(float)


def device_power(hs, period, matrix):
    """Nearest-bin lookup in the power matrix; outside the matrix → 0."""
    mh, mt, mp = matrix
    ih = np.abs(hs[:, None] - mh[None, :]).argmin(1)
    it = np.abs(period[:, None] - mt[None, :]).argmin(1)
    p = mp[ih, it]
    dh = np.diff(mh).mean() / 2 if len(mh) > 1 else 0.25
    dt = np.diff(mt).mean() / 2 if len(mt) > 1 else 0.5
    outside = (hs < mh.min() - dh) | (hs > mh.max() + dh) | (period < mt.min() - dt) | (period > mt.max() + dt)
    p[outside] = 0
    return np.nan_to_num(p)


def mvi(p):
    mon = p.groupby(p.index.month).mean()
    return (mon.max() - mon.min()) / p.mean(), mon


def run(args):
    os.makedirs(OUT, exist_ok=True)
    matrix = load_matrix(args.power_matrix) if args.power_matrix else None
    rows, clim, yearly = [], {}, []
    for sid in args.sites:
        d = pd.read_csv(os.path.join(ROOT, 'processed', f'site_{sid}', 'observations_hourly.csv'),
                        usecols=['valid_time', 'hs', 'te'])
        d.index = pd.to_datetime(d['valid_time'].str.replace('Z', ''))
        d = d[['hs', 'te']].dropna()
        hs, te = d['hs'].to_numpy(), d['te'].to_numpy()
        if matrix is None:
            p = FLUX * hs ** 2 * te
            off = hs < args.hs_cutin
            if args.hs_cutout:
                off |= hs > args.hs_cutout
            p = np.where(off, 0.0, p)
            unit, rated = 'kW/m', None
        else:
            period = te if args.matrix_period == 'te' else te / 0.9
            p = device_power(hs, period, matrix)
            unit, rated = 'kW', float(np.nanmax(matrix[2]))
        p = pd.Series(p, index=d.index)
        m, mon = mvi(p)
        clim[sid] = mon / p.mean()
        per_year = [mvi(g)[0] for _, g in p.groupby(p.index.year) if len(g) > 24 * 300]
        yearly += [dict(site=sid, name=NAMES.get(sid, sid), year=y, mean_power=g.mean(), mvi=mvi(g)[0], cv=g.std() / g.mean())
                   for y, g in p.groupby(p.index.year) if len(g) > 24 * 300]
        rows.append(dict(site=sid, name=NAMES.get(sid, sid), period=f'{d.index.min():%Y}–{d.index.max():%Y}',
                         unit=unit, mean_power=p.mean(), max_month=int(mon.idxmax()), min_month=int(mon.idxmin()),
                         MVI=m, MVI_yearly_mean=np.mean(per_year), MVI_yearly_std=np.std(per_year, ddof=1),
                         CV=p.std() / p.mean(), downtime_pct=100 * (p <= 0).mean(),
                         CF=p.mean() / rated if rated else np.nan))
    res = pd.DataFrame(rows)
    res.round(4).to_csv(os.path.join(OUT, 'site_metrics.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame(clim).round(4).rename_axis('month').to_csv(os.path.join(OUT, 'monthly_climatology.csv'), encoding='utf-8-sig')
    yr = pd.DataFrame(yearly)
    yr.round(4).to_csv(os.path.join(OUT, 'yearly_metrics.csv'), index=False, encoding='utf-8-sig')
    from scipy.stats import linregress
    trends = []
    for sid, g in yr.groupby('site', sort=False):
        rec = dict(site=sid, name=NAMES.get(sid, sid), years=f'{g.year.min()}–{g.year.max()}')
        for col, label in [('mean_power', 'power'), ('mvi', 'mvi')]:
            lr = linregress(g.year, g[col])
            rec[f'{label}_trend_pct_per_decade'] = 100 * lr.slope * 10 / g[col].mean()
            rec[f'{label}_p_value'] = lr.pvalue
        trends.append(rec)
    tr = pd.DataFrame(trends)
    tr.round(4).to_csv(os.path.join(OUT, 'trends.csv'), index=False, encoding='utf-8-sig')
    setting = (f'装置功率矩阵 {os.path.basename(args.power_matrix)}（周期 {args.matrix_period}）' if matrix is not None else
               f'理论波能流 0.49·Hs²·Te，Hs < {args.hs_cutin} m 输出为零' + (f'，Hs > {args.hs_cutout} m 停机' if args.hs_cutout else ''))
    lines = ['# 站点波浪能资源与出力稳定性', '', f'功率算法：{setting}；数据：CDS ERA5 逐时。', '',
             '| 站点 | 平均功率 | 最大/最小月 | MVI | 逐年 MVI（均值 ± 标准差） | CV | 停机率 |' + (' CF |' if matrix is not None else ''),
             '|---|---:|---|---:|---:|---:|---:|' + ('---:|' if matrix is not None else '')]
    for r in res.itertuples():
        lines.append(f'| {r.site} {r.name} | {r.mean_power:.2f} {r.unit} | {r.max_month} 月 / {r.min_month} 月 | {r.MVI:.2f} | '
                     f'{r.MVI_yearly_mean:.2f} ± {r.MVI_yearly_std:.2f} | {r.CV:.2f} | {r.downtime_pct:.1f}% |'
                     + (f' {r.CF:.2f} |' if matrix is not None else ''))
    lines += ['', 'MVI、CV、停机率越小，出力越稳定。定义同 Jin et al. 2025（Energy 323, 135730）。ERA5 近岸低估大浪，A/B/C 的功率偏低。',
              '', '## 年际变化（逐年计算，线性趋势）', '',
              '| 站点 | 年份 | 年平均功率趋势（%/10 年） | p 值 | 逐年 MVI 趋势（%/10 年） | p 值 | 最大 / 最小年平均功率 |',
              '|---|---|---:|---:|---:|---:|---|']
    for r in tr.itertuples():
        g = yr[yr.site == r.site]
        hi, lo = g.loc[g.mean_power.idxmax()], g.loc[g.mean_power.idxmin()]
        sig = lambda pv: f'{pv:.3f}' + ('（显著）' if pv < 0.05 else '')
        lines.append(f'| {r.site} {r.name} | {r.years} | {r.power_trend_pct_per_decade:+.1f} | {sig(r.power_p_value)} | '
                     f'{r.mvi_trend_pct_per_decade:+.1f} | {sig(r.mvi_p_value)} | '
                     f'{int(hi.year)} 年 {hi.mean_power:.2f} / {int(lo.year)} 年 {lo.mean_power:.2f} |')
    open(os.path.join(OUT, 'report.md'), 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        fig, ax = plt.subplots(figsize=(9, 5))
        for sid, c in clim.items():
            ax.plot(c.index, c.values, marker='o', label=f'{sid} {NAMES.get(sid, sid)}')
        ax.axhline(1, color='k', lw=1, ls='--')
        ax.set_xticks(range(1, 13)); ax.set_xlabel('月份'); ax.set_ylabel('月平均功率 / 年平均功率')
        ax.set_title('各站点波浪能的年内变化（ERA5 1995–2024）'); ax.legend(ncol=3)
        fig.tight_layout(); fig.savefig(os.path.join(OUT, 'monthly_power.png'), dpi=150); plt.close(fig)
        fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        for sid, g in yr.groupby('site', sort=False):
            ax[0].plot(g.year, g.mean_power, marker='o', ms=3, label=f'{sid} {NAMES.get(sid, sid)}')
            ax[1].plot(g.year, g.mvi, marker='o', ms=3, label=f'{sid} {NAMES.get(sid, sid)}')
        ax[0].set_ylabel(f'年平均功率（{unit}）'); ax[0].set_title('各站逐年平均波浪能（ERA5 1995–2024）'); ax[0].legend(ncol=3)
        ax[1].set_ylabel('逐年 MVI'); ax[1].set_xlabel('年份'); ax[1].set_title('各站逐年月变化指数 MVI（越大表示年内越不均匀）')
        fig.tight_layout(); fig.savefig(os.path.join(OUT, 'yearly_power_mvi.png'), dpi=150); plt.close(fig)
    except Exception as e:
        print('画图失败：', e)
    print('\n'.join(lines))
    print(f'\n结果在 {OUT}')


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    p = argparse.ArgumentParser()
    p.add_argument('--sites', nargs='+', default=['A', 'B', 'C', 'G', 'K', 'H'])
    p.add_argument('--hs-cutin', type=float, default=0.5)
    p.add_argument('--hs-cutout', type=float, default=None)
    p.add_argument('--power-matrix')
    p.add_argument('--matrix-period', choices=['te', 'tp'], default='te')
    run(p.parse_args())
