"""历史波浪能的年平均、月平均与 MVI 变化表（MVI 指标见 Jin et al. 2025, Energy 323；看各站长期变化）。

数据：ERA5 单点长序列（1940–2024，逐时 Hs 与能量周期 mwp），波能流 P = 0.49·Hs²·Te（kW/m）。
尺度订正：ERA5 与浮标有系统差别；用两者都有的年份算一个比例 = 实测平均波能流 / ERA5 平均波能流（实测 = 浮标 Hs ×
  关岛、夏威夷的 CDIP 实测 Te，马罗岛没有海浪谱、用 ERA5 Te），整条序列乘这个比例。单一比例不改变 MVI（比值）
  和相对趋势，只让绝对值接近浮标尺度。
指标（每站每年）：年平均波能流；12 个月的月平均；MVI = (最大月平均 − 最小月平均) / 年平均（Jin 2025 式 17）；
  CV = 逐时标准差 / 平均；另列浮标实测年份的同样指标（年内有效小时 ≥ 80% 才算）。
趋势：年平均与 MVI 的线性趋势（每十年占多年平均的百分比，及 p 值），分 1940–2024 与 1979–2024 两段
  （1979 年前没有卫星资料同化，ERA5 可信度较低）。
输出：results/wave_energy_history/ 下 annual.csv、monthly.csv、trends.csv、decades.csv、report.md、annual_mvi.png
用法：python historical_wave_energy.py [G K H]
"""
import glob
import os
import sys
import tempfile

import numpy as np
import pandas as pd
from scipy.stats import linregress

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from era5_to_observations import SITES, load_site_file   # noqa: E402

OUT = os.path.join(ROOT, 'results', 'wave_energy_history')
FLUX = 1025.0 * 9.81 ** 2 / (64 * np.pi) / 1000
NAMES = {'G': '关岛', 'K': '马罗岛', 'H': '夏威夷'}


def era5_series(site):
    lat, lon = site.get('ts_point') or site['point']
    parts, used = [], {}
    with tempfile.TemporaryDirectory() as tmp:
        for f in sorted(glob.glob(os.path.join(ROOT, 'era5_data', f"ts_{site['name']}_*.nc"))):
            parts += load_site_file(f, lat, lon, tmp, used)
    d = pd.concat(parts)[['hs', 'te']].groupby(level=0).first().sort_index()
    d.index = d.index.tz_localize('UTC') if d.index.tz is None else d.index
    return (FLUX * d.hs ** 2 * d.te).dropna()


def observed_series(sid, station):
    b = pd.read_csv(os.path.join(ROOT, 'processed', 'buoy', f'{sid}_{station}.csv'), usecols=['valid_time', 'hs'])
    b.index = pd.to_datetime(b.valid_time, utc=True)
    hs = b.hs.groupby(b.index.round('1h')).mean()
    spec = glob.glob(os.path.join(ROOT, 'processed', 'buoy_spectral', f'{sid}_*.csv'))
    if spec:
        t = pd.read_csv(spec[0], usecols=['valid_time', 'te'])
        te = pd.Series(t.te.values, index=pd.to_datetime(t.valid_time, utc=True))
    else:
        e = pd.read_csv(os.path.join(ROOT, 'processed', f'site_{sid}', 'observations_hourly.csv'), usecols=['valid_time', 'te'])
        te = pd.Series(e.te.values, index=pd.to_datetime(e.valid_time, utc=True))
    return (FLUX * hs ** 2 * te.reindex(hs.index)).dropna(), ('CDIP 实测' if spec else 'ERA5')


def yearly(series, min_cover=0.8):
    rows, mon = [], []
    for y, g in series.groupby(series.index.year):
        hours = 8784 if pd.Timestamp(f'{y}-12-31').dayofyear == 366 else 8760
        if len(g) < min_cover * hours:
            continue
        m = g.groupby(g.index.month).mean()
        if len(m) < 12:
            continue
        rows.append(dict(year=y, annual_mean=g.mean(), mvi=(m.max() - m.min()) / g.mean(), cv=g.std() / g.mean(),
                         max_month=int(m.idxmax()), min_month=int(m.idxmin())))
        mon += [dict(year=y, month=k, mean=v) for k, v in m.items()]
    return pd.DataFrame(rows), pd.DataFrame(mon)


def trend(df, col, y0, y1):
    d = df[(df.year >= y0) & (df.year <= y1)]
    r = linregress(d.year, d[col])
    return r.slope * 10 / d[col].mean(), r.pvalue, len(d)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    sites = {s['id']: s for s in SITES}
    ids = sys.argv[1:] or ['G', 'K', 'H']
    annual, monthly, trends, decades, lines = [], [], [], [], ['# 历史波浪能：年平均、月平均与 MVI 的变化', '']
    for sid in ids:
        site = sites[sid]
        e5 = era5_series(site)
        obs, te_src = observed_series(sid, site['buoy']['station'])
        both = pd.concat([e5, obs], axis=1, keys=['e', 'o']).dropna()
        scale = both.o.mean() / both.e.mean()
        e5 = e5 * scale
        ya, ym = yearly(e5)
        oa, _ = yearly(obs)
        ya['site_id'], ym['site_id'], oa['site_id'] = sid, sid, sid
        ya['source'], oa['source'] = 'ERA5（订正）', f'浮标（Te：{te_src}）'
        annual += [ya, oa]; monthly.append(ym)
        lines += [f'## {NAMES[sid]}（{sid}）', '',
                  f'ERA5 尺度订正比例 {scale:.3f}（与浮标重叠 {both.index.year.nunique()} 年；实测 Te 来源：{te_src}）。'
                  f'ERA5 订正后 {ya.year.min()}–{ya.year.max()}（{len(ya)} 年），浮标实测 {len(oa)} 年。', '',
                  '| 时段 | 年平均波能流 kW/m 趋势（每十年） | p | MVI 趋势（每十年） | p | 年数 |', '|---|---:|---:|---:|---:|---:|']
        for y0, y1 in ((1940, 2024), (1979, 2024)):
            ta, pa, n = trend(ya, 'annual_mean', y0, y1)
            tm, pm, _ = trend(ya, 'mvi', y0, y1)
            trends.append(dict(site_id=sid, period=f'{y0}-{y1}', annual_mean_trend_pct_per_decade=round(ta * 100, 2), p_annual=round(pa, 4),
                               mvi_trend_pct_per_decade=round(tm * 100, 2), p_mvi=round(pm, 4), years=n))
            lines.append(f'| {y0}–{y1} | {ta:+.1%}{"（显著）" if pa < .05 else ""} | {pa:.3f} | {tm:+.1%}{"（显著）" if pm < .05 else ""} | {pm:.3f} | {n} |')
        ya['decade'] = ya.year // 10 * 10
        dec = ya.groupby('decade').agg(annual_mean=('annual_mean', 'mean'), mvi=('mvi', 'mean'), mvi_min=('mvi', 'min'),
                                       mvi_max=('mvi', 'max'), years=('year', 'size')).reset_index()
        dec['site_id'] = sid; decades.append(dec)
        lines += ['', '| 年代 | 年平均波能流 kW/m | 平均 MVI | MVI 范围 | 年数 |', '|---|---:|---:|---|---:|']
        lines += [f'| {int(r.decade)}s | {r.annual_mean:.1f} | {r.mvi:.2f} | {r.mvi_min:.2f}–{r.mvi_max:.2f} | {int(r.years)} |' for r in dec.itertuples()]
        last = ya[ya.year >= 2010].merge(oa[['year', 'annual_mean', 'mvi']], on='year', how='left', suffixes=('', '_buoy'))
        lines += ['', '| 年份 | 年平均（ERA5 订正） | MVI（ERA5 订正） | 年平均（浮标实测） | MVI（浮标实测） | 最大 / 最小月 |', '|---|---:|---:|---:|---:|---|']
        lines += [f"| {int(r.year)} | {r.annual_mean:.1f} | {r.mvi:.2f} | {'' if pd.isna(r.annual_mean_buoy) else f'{r.annual_mean_buoy:.1f}'} | "
                  f"{'' if pd.isna(r.mvi_buoy) else f'{r.mvi_buoy:.2f}'} | {int(r.max_month)} / {int(r.min_month)} 月 |" for r in last.itertuples()]
        lines += ['', f'多年平均：年平均 {ya.annual_mean.mean():.1f} kW/m（年际标准差 {ya.annual_mean.std():.1f}，'
                  f'即 ±{ya.annual_mean.std() / ya.annual_mean.mean():.0%}）；MVI {ya.mvi.mean():.2f}（{ya.mvi.min():.2f}–{ya.mvi.max():.2f}）。', '']
        print(f'{sid}：ERA5 {len(ya)} 年、浮标 {len(oa)} 年，订正比例 {scale:.3f}', flush=True)
    os.makedirs(OUT, exist_ok=True)
    pd.concat(annual).round(4).to_csv(os.path.join(OUT, 'annual.csv'), index=False, encoding='utf-8-sig')
    pd.concat(monthly).round(3).to_csv(os.path.join(OUT, 'monthly.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame(trends).to_csv(os.path.join(OUT, 'trends.csv'), index=False, encoding='utf-8-sig')
    pd.concat(decades).round(3).to_csv(os.path.join(OUT, 'decades.csv'), index=False, encoding='utf-8-sig')
    lines += ['说明：MVI = (最大月平均 − 最小月平均) / 年平均；趋势为线性回归斜率 × 10 年 / 多年平均，p < 0.05 记为显著。',
              '1979 年以前 ERA5 没有卫星资料同化，早期年份的绝对值与趋势要谨慎解读。']
    open(os.path.join(OUT, 'report.md'), 'w', encoding='utf-8').write('\n'.join(lines) + '\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    a = pd.concat(annual)
    fig, axes = plt.subplots(len(ids), 2, figsize=(11, 3 * len(ids)), squeeze=False, sharex=True)
    for i, sid in enumerate(ids):
        for j, (col, lab) in enumerate((('annual_mean', '年平均波能流 (kW/m)'), ('mvi', 'MVI'))):
            ax = axes[i][j]
            for src, style in (('ERA5（订正）', 'C0-'), (None, 'k.')):
                g = a[(a.site_id == sid) & ((a.source == src) if src else (a.source != 'ERA5（订正）'))]
                ax.plot(g.year, g[col], style, lw=1, ms=5, label=src or '浮标实测')
            ax.axvline(1979, color='gray', ls=':', lw=.8)
            ax.set_title(f'{NAMES[sid]} {lab}', fontsize=10)
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'annual_mvi.png'), dpi=150)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
