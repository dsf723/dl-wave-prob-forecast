"""预报的波浪能与 MVI 汇总：看订正后的预报能不能复现各站月平均波浪能的季节变化（MVI），方便最后总结观察。

读取 export_quantile_forecasts.py 导出的表（含 WEF_* 波能流列），按 站点 × 预见期段（第 1–3、4–7、8–模型时效天，
以及 --days 30 导出时的气候态段）统计：
  - 逐月平均波能流（kW/m）：实测（浮标 Hs + 实测 Te：关岛、夏威夷为 CDIP 谱算的真实 Te，马罗岛为 ERA5 Te）、订正后预报（预报分布的均值；气候态段为气候态均值）、
    原始 GEFS（集合中位数；气候态段没有）；
  - MVI = (最大月平均 − 最小月平均) / 全期平均（Jin 2025 式 17；这里是检验期内的多年逐月平均）；
  - 全期平均波能流的相对偏差，订正后 P10–P90 对实测波能流的覆盖率。
只用实测和预报都有值的时刻，三者样本完全相同。检验期较短（G、K 3 年，H 2 年），某些月份样本少，MVI 只作定性比较。
输出：results/forecast_outputs/wave_energy_<方案>_<split>_monthly.csv、..._mvi.csv、..._report.md、..._monthly.png
用法：python forecast_wave_energy_mvi.py [--file results/forecast_outputs/final_test.csv]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
FLUX = 1025.0 * 9.81 ** 2 / (64 * np.pi) / 1000
NAMES = {'G': '关岛', 'K': '马罗岛', 'H': '夏威夷'}
SERIES = [('WEF_observed_kWm', '实测'), ('WEF_mean_kWm', '订正后预报'), ('WEF_raw_kWm', '原始 GEFS')]


def mvi(monthly):
    return (monthly.max() - monthly.min()) / monthly.mean()


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    p = argparse.ArgumentParser()
    p.add_argument('--file', default=os.path.join(ROOT, 'results', 'forecast_outputs', 'final_test.csv'))
    a = p.parse_args()
    d = pd.read_csv(a.file)
    if 'WEF_mean_kWm' not in d:
        raise SystemExit('表里没有波能流列：先用新版 export_quantile_forecasts.py 重新导出')
    d['WEF_raw_kWm'] = FLUX * d.GEFS_raw_P50 ** 2 * d.GEFS_Te_mean_s
    d = d.dropna(subset=['WEF_observed_kWm', 'WEF_mean_kWm'])
    if 'source' not in d:
        d['source'] = 'model'
    model_days = int(d.loc[d.source != 'climatology', 'lead_day'].max())
    WINDOWS = [('第1–3天', 1, 3), ('第4–7天', 4, 7), (f'第8–{model_days}天', 8, model_days)]
    if (d.source == 'climatology').any():
        last = int(d.lead_day.max())
        WINDOWS.append((f'第{model_days + 1}–{last}天（气候态）', model_days + 1, last))
    d['month'] = pd.to_datetime(d.valid_time, utc=True).dt.month
    stem = os.path.splitext(os.path.basename(a.file))[0]
    out = os.path.join(os.path.dirname(a.file), f'wave_energy_{stem}')

    monthly, rows = [], []
    for site in sorted(d.site_id.unique()):
        for label, lo, hi in WINDOWS:
            w = d[(d.site_id == site) & d.lead_day.between(lo, hi)]
            m = w.groupby('month')[[c for c, _ in SERIES]].mean()
            if len(m) < 12:
                print(f'注意：站点 {site} {label} 只有 {len(m)} 个月有数据，MVI 不完整')
            m.insert(0, '预见期', label); m.insert(0, 'site_id', site)
            monthly.append(m.reset_index())
            obs = w.WEF_observed_kWm
            row = {'site_id': site, '预见期': label, '样本时刻': len(w)}
            for col, name in SERIES:
                row[f'{name}_平均_kWm'] = w[col].mean()
                row[f'{name}_MVI'] = mvi(m[col])
            for col, name in SERIES[1:]:
                row[f'{name}_平均偏差'] = w[col].mean() / obs.mean() - 1 if w[col].notna().any() else float('nan')
            row['订正后P10–P90覆盖率'] = ((obs >= w.WEF_P10_kWm) & (obs <= w.WEF_P90_kWm)).mean()
            rows.append(row)
    monthly = pd.concat(monthly, ignore_index=True)
    table = pd.DataFrame(rows)
    monthly.round(3).to_csv(out + '_monthly.csv', index=False, encoding='utf-8-sig')
    table.round(4).to_csv(out + '_mvi.csv', index=False, encoding='utf-8-sig')

    lines = [f'# 预报的波浪能与 MVI（{os.path.basename(a.file)}）', '',
             '波能流 P = 0.49·Hs²·Te（kW/m）。实测 = 浮标 Hs + 实测 Te（关岛、夏威夷为 CDIP 海浪谱的真实 Te，马罗岛为 ERA5 Te）；订正后 = 预报分布的均值（周期暂用 GEFS 集合平均 Te）；'
             '原始 GEFS = 集合中位数 Hs 与集合平均 Te。MVI = (最大月平均 − 最小月平均) / 全期平均。', '',
             '| 站点 | 预见期 | 实测平均 | 订正后平均（偏差） | 原始 GEFS 平均（偏差） | 实测 MVI | 订正后 MVI | 原始 MVI | P10–P90 覆盖率 |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {NAMES.get(r['site_id'], r['site_id'])} | {r['预见期']} | {r['实测_平均_kWm']:.1f} | "
                     f"{r['订正后预报_平均_kWm']:.1f}（{r['订正后预报_平均偏差']:+.0%}） | "
                     + (f"{r['原始 GEFS_平均_kWm']:.1f}（{r['原始 GEFS_平均偏差']:+.0%}）" if r['原始 GEFS_平均_kWm'] == r['原始 GEFS_平均_kWm'] else '—')
                     + f" | {r['实测_MVI']:.2f} | "
                     f"{r['订正后预报_MVI']:.2f} | {r['原始 GEFS_MVI']:.2f} | {r['订正后P10–P90覆盖率']:.0%} |")
    lines += ['', '说明：检验期短（验证期、测试期各站只有 2–3 年），MVI 只作定性比较；马罗岛浮标没有海浪谱，其"实测"波能流用的是 ERA5 Te。']
    open(out + '_report.md', 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    sites = sorted(d.site_id.unique())
    fig, axes = plt.subplots(len(sites), len(WINDOWS), figsize=(4 * len(WINDOWS), 3 * len(sites)), squeeze=False, sharex=True)
    for i, site in enumerate(sites):
        for j, (label, _, _) in enumerate(WINDOWS):
            ax = axes[i][j]
            m = monthly[(monthly.site_id == site) & (monthly['预见期'] == label)]
            for (col, name), style in zip(SERIES, ['k-o', 'C0-s', 'C3--^']):
                ax.plot(m.month, m[col], style, ms=3, lw=1.2, label=name)
            ax.set_title(f'{NAMES.get(site, site)} {label}', fontsize=10)
            ax.set_xticks(range(1, 13))
            if j == 0:
                ax.set_ylabel('月平均波能流 (kW/m)')
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out + '_monthly.png', dpi=150)
    print(f'→ {out}_monthly.csv / _mvi.csv / _report.md / _monthly.png')


if __name__ == '__main__':
    main()
