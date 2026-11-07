"""舟山站 A 的波能流与 MVI，与 Jin et al. 2026（Energy 366, 142735）表 3 的 H1–H10 对比。

论文：SWAN（0.02°，ERA5 风场驱动）2001–2020 逐时；P = ρg²/(64π)·Te·Hs²（式 7），CV = σ/均值（式 9），
      MVI = (最高月平均 − 最低月平均) / 年平均（式 10）；未提到低波高截断。论文没有说明 MVI 是多年逐月气候态还是逐年算后平均，这里两种都算。
本脚本：processed/site_A/observations_hourly.csv（ERA5 单点 30.5°N、123.0°E，mwp 为能量周期），同一时段、同一公式，不设截断。
        site_resource_metrics.py 里的 3.45 kW/m、MVI 0.85 用 1995–2024 且 Hs < 0.5 m 记零，所以与这里不同。
不确定性：按年块自助（2000 次）给 95% 区间。
高度计核对：沿用 compare_era5_altimeter.passes（CCI L3，swh_adjusted，每次过境取中位数），只用 2001–2020，半径 30、55 km；
        分位数映射把 ERA5 Hs 订正到高度计分布，Te 不变（没有独立的周期观测），作敏感性分析，不是真值。
输出：results/resource_metrics/zhoushan_vs_jin2026.md
用法：python zhoushan_vs_jin2026.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

import compare_era5_altimeter as cea

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, 'results', 'resource_metrics', 'zhoushan_vs_jin2026.md')
FLUX = 1025.0 * 9.81 ** 2 / (64 * np.pi) / 1000          # kW/m per (m² s) ≈ 0.4906
START, END = '2001-01-01', '2020-12-31 23:59'
# Jin et al. 2026 表 3：经度、纬度、Hs (m)、Te (s)、波能流 (W/m)、CV、MVI
PAPER = pd.DataFrame(
    [('H1', 122.62, 29.48, 1.35, 6.42, 6610, 1.18, 1.39), ('H2', 122.63, 29.66, 1.30, 6.34, 6040, 1.17, 1.37),
     ('H3', 122.71, 29.84, 1.32, 6.36, 6220, 1.17, 1.33), ('H4', 122.85, 30.00, 1.33, 6.32, 6320, 1.16, 1.29),
     ('H5', 122.93, 30.14, 1.35, 6.38, 6590, 1.15, 1.26), ('H6', 122.93, 30.38, 1.35, 6.41, 6600, 1.17, 1.17),
     ('H7', 122.95, 30.58, 1.31, 6.32, 6220, 1.17, 1.15), ('H8', 122.92, 30.82, 1.28, 6.24, 5870, 1.23, 1.16),
     ('H9', 122.96, 30.88, 1.26, 6.20, 5790, 1.23, 1.15), ('H10', 123.00, 30.98, 1.31, 6.28, 6270, 1.24, 1.12)],
    columns=['site', 'lon', 'lat', 'hs', 'te', 'flux', 'cv', 'mvi']).set_index('site')


def pooled_mvi(p):
    mon = p.groupby(p.index.month).mean()
    return (mon.max() - mon.min()) / mon.mean()


def yearly_mvi(p):
    return p.groupby(p.index.year).apply(pooled_mvi)


def metrics(p):
    monthly = p.groupby([p.index.year, p.index.month]).mean()
    return dict(mean=p.mean(), mvi=pooled_mvi(p), mvi_y=yearly_mvi(p).mean(), cv=p.std() / p.mean(), cv_month=monthly.std() / monthly.mean())


def year_boot(p, rng, n=2000):
    """按年重抽样：逐年逐月均值是足够统计量，所以先压缩再抽。"""
    tab = p.groupby([p.index.year, p.index.month]).agg(['sum', 'count']).unstack()
    s, c = tab['sum'].to_numpy(), tab['count'].to_numpy()
    idx = rng.integers(0, len(s), (n, len(s)))
    ss, cc = s[idx].sum(1), c[idx].sum(1)
    mon = ss / cc
    mean = ss.sum(1) / cc.sum(1)
    ym = (s / c)
    yv = (ym.max(1) - ym.min(1)) / (s.sum(1) / c.sum(1))
    return dict(mean=np.percentile(mean, [2.5, 97.5]),
                mvi=np.percentile((mon.max(1) - mon.min(1)) / mon.mean(1), [2.5, 97.5]),
                mvi_y=np.percentile(yv[idx].mean(1), [2.5, 97.5]))


def quantile_map(hs, era_pass, alt_pass):
    """分位数映射：1%–99% 之间按分位点插值，两端外推用端点的差值。"""
    q = np.linspace(.01, .99, 99)
    xe, xa = np.quantile(era_pass, q), np.quantile(alt_pass, q)
    out = np.interp(hs, xe, xa)
    out = np.where(hs > xe[-1], hs + xa[-1] - xe[-1], out)
    return np.where(hs < xe[0], np.maximum(hs + xa[0] - xe[0], .01), out)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    site = next(s for s in json.load(open(os.path.join(ROOT, 'metadata', 'sites.json'), encoding='utf-8'))['sites'] if s['id'] == 'A')
    lat, lon = site['point']
    d = pd.read_csv(os.path.join(ROOT, 'processed', 'site_A', 'observations_hourly.csv'), usecols=['valid_time', 'hs', 'te'])
    d.index = pd.to_datetime(d['valid_time'].str.replace('Z', ''))
    full = d[['hs', 'te']].dropna()
    d = full.loc[START:END]
    assert len(d) == 175320 and d.index.year.nunique() == 20, len(d)
    rng = np.random.default_rng(0)
    P = pd.Series(FLUX * d.hs.to_numpy() ** 2 * d.te.to_numpy(), index=d.index)
    m, ci = metrics(P), year_boot(P, rng)
    paper = PAPER.copy()
    paper['dist'] = cea.haversine(lat, lon, paper.lat, paper.lon)
    near = paper.dist.idxmin()
    H = paper.loc[near]

    # 分解差异：P = FLUX · 均值(Hs)² · 均值(Te) · 协变因子
    cov_our = P.mean() / (FLUX * d.hs.mean() ** 2 * d.te.mean())
    cov_pap = (H.flux / 1000) / (FLUX * H.hs ** 2 * H.te)
    r_total = (H.flux / 1000) / P.mean()
    r_hs, r_te, r_cov = (H.hs / d.hs.mean()) ** 2, H.te / d.te.mean(), cov_pap / cov_our

    # 高度计
    alt_rows, corrected = [], {}
    for radius in (30, 55):
        cea.RADIUS_KM = radius                     # 限制的是每个沿轨点的距离，不是过境的平均距离
        g = cea.passes(site)
        g = g[(g.time >= START) & (g.time <= END)].copy()
        g['hs_era'] = cea.interp_to(full.hs, g.time)
        g = g.dropna(subset=['hs_era'])
        e, a = g.hs_era.to_numpy(), g.hs_alt.to_numpy()
        hc = quantile_map(d.hs.to_numpy(), e, a)
        pc = pd.Series(FLUX * hc ** 2 * d.te.to_numpy(), index=d.index)
        corrected[radius] = pc
        mc, cc = metrics(pc), year_boot(pc, rng)
        alt_rows.append(dict(radius=radius, n=len(g), dist=g.dist_km.mean(), alt=a.mean(), era=e.mean(), bias=(e - a).mean(), rmse=np.sqrt(((e - a) ** 2).mean()),
                             r2=(a ** 2).mean() / (e ** 2).mean(), hs_c=hc.mean(), te=d.te.mean(), m=mc, c=cc))

    lines = ['# 舟山站 A 与 Jin et al. 2026（Energy 366, 142735）对比', '',
             f'同一时段（2001–2020）、同一公式（P = 0.4906·Hs²·Te，不设低波高截断）。站点 A：{lat}°N、{lon}°E，ERA5 逐时，{len(d):,} 个时刻。',
             f'论文最近的点是 {near}（{H.lat}°N、{H.lon}°E，距 A {H.dist:.0f} km）。论文数据是 0.02° SWAN，本项目是 ERA5（波浪原始格距约 0.5°）。', '',
             '## 主要结果', '',
             '| 数据 | 平均 Hs (m) | 平均 Te (s) | 平均波能流 (kW/m) | MVI（多年逐月） | MVI（逐年算后平均） | CV（逐时） |',
             '|---|---:|---:|---:|---:|---:|---:|',
             f'| ERA5 站点 A | {d.hs.mean():.2f} | {d.te.mean():.2f} | {m["mean"]:.2f} [{ci["mean"][0]:.2f}, {ci["mean"][1]:.2f}] | '
             f'{m["mvi"]:.2f} [{ci["mvi"][0]:.2f}, {ci["mvi"][1]:.2f}] | {m["mvi_y"]:.2f} [{ci["mvi_y"][0]:.2f}, {ci["mvi_y"][1]:.2f}] | {m["cv"]:.2f} |']
    for r in alt_rows:
        lines.append(f'| ERA5 + 高度计订正（{r["radius"]} km）| {r["hs_c"]:.2f} | {r["te"]:.2f} | {r["m"]["mean"]:.2f} [{r["c"]["mean"][0]:.2f}, {r["c"]["mean"][1]:.2f}] | '
                     f'{r["m"]["mvi"]:.2f} [{r["c"]["mvi"][0]:.2f}, {r["c"]["mvi"][1]:.2f}] | {r["m"]["mvi_y"]:.2f} [{r["c"]["mvi_y"][0]:.2f}, {r["c"]["mvi_y"][1]:.2f}] | {r["m"]["cv"]:.2f} |')
    lines += [f'| 论文 {near} | {H.hs:.2f} | {H.te:.2f} | {H.flux / 1000:.2f} | {H.mvi:.2f}（未说明算法） | | {H.cv:.2f} |',
              f'| 论文 H1–H10 范围 | {paper.hs.min():.2f}–{paper.hs.max():.2f} | {paper.te.min():.2f}–{paper.te.max():.2f} | {paper.flux.min() / 1000:.2f}–{paper.flux.max() / 1000:.2f} | '
              f'{paper.mvi.min():.2f}–{paper.mvi.max():.2f} | | {paper.cv.min():.2f}–{paper.cv.max():.2f} |', '',
              '方括号是按年块自助的 95% 区间。CV 用逐时序列；逐月平均的 CV 为 ' + f'{m["cv_month"]:.2f}，论文的数值量级对应逐时。', '',
              f'## 波能流差异的分解（论文 {near} ÷ ERA5）', '',
              f'- 总比值 {r_total:.2f} = Hs 平方项 {r_hs:.2f} × Te {r_te:.2f} × 协变因子 {r_cov:.2f}（协变因子 = 均值(Hs²Te) / (均值(Hs)²·均值(Te))：ERA5 {cov_our:.2f}，论文 {cov_pap:.2f}）。', '',
              '## 高度计核对（2001–2020，过境中位数）', '',
              '| 半径 | 过境次数 | 平均距离 (km) | 高度计平均 Hs | 同时刻 ERA5 | 偏差 | RMSE | 高度计 Hs² ÷ ERA5 Hs² |', '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in alt_rows:
        lines.append(f'| {r["radius"]} km | {r["n"]:,} | {r["dist"]:.0f} | {r["alt"]:.2f} | {r["era"]:.2f} | {r["bias"]:+.2f} | {r["rmse"]:.2f} | {r["r2"]:.2f} |')
    lines += ['', f'过境时刻的 ERA5 平均比全部时刻（{d.hs.mean():.2f} m）高，说明过境样本本身偏大，不能直接拿高度计平均 Hs 与论文比；比较用分位数映射后全部时刻的平均 Hs。',
              '', f'论文对 CFOSAT 2019–2020 的 SWAN 检验：偏差 −0.039 m，RMSE 0.262 m，SI 0.181，相关 0.949。论文数据未公开，这里无法与 SWAN 逐时对比。', '',
              '## 逐月平均波能流（kW/m，2001–2020）', '', '| 月 | ERA5 | 订正（30 km）| 订正（55 km）|', '|---:|---:|---:|---:|']
    mon = P.groupby(P.index.month).mean()
    m30, m55 = [corrected[k].groupby(corrected[k].index.month).mean() for k in (30, 55)]
    lines += [f'| {i} | {mon[i]:.2f} | {m30[i]:.2f} | {m55[i]:.2f} |' for i in mon.index]
    lines += ['', '## 论文 H1–H10 与距离', '', '| 点 | 纬度 | 经度 | 距 A (km) | Hs | Te | 波能流 (kW/m) | CV | MVI |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    lines += [f'| {i} | {r.lat} | {r.lon} | {r.dist:.0f} | {r.hs:.2f} | {r.te:.2f} | {r.flux / 1000:.2f} | {r.cv:.2f} | {r.mvi:.2f} |' for i, r in paper.iterrows()]
    lines += ['', '说明：订正只改 Hs，Te 仍是 ERA5，舟山没有独立的周期观测；高度计点与站点有一定距离，订正结果是敏感性范围，不是真值。',
              '逐月极差取最大减最小，重抽样的区间偏向高值，所以多年逐月 MVI 的点估计靠近区间下端。']
    open(OUT, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'\n→ {OUT}')


if __name__ == '__main__':
    main()
