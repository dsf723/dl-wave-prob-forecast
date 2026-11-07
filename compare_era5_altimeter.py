"""ERA5 与卫星高度计（CMEMS L3 再处理沿轨，ESA CCI）有效波高对比：给没有浮标的 A、B、C 一个独立检验。

输入：processed/altimeter/<ID>_cci_l3.csv（download_cmems_altimeter.py 生成）
      processed/site_<ID>/observations_hourly.csv（CDS ERA5，含台风标签）
      processed/buoy/<ID>_*.csv（有浮标的站点：再检查高度计与浮标是否一致，作为高度计可信度的旁证）
配对：
  1. 只用距站点 ≤ RADIUS_KM 的沿轨点，变量 swh_adjusted（相互校准后的 Hs）；
  2. 同一卫星、相邻点间隔 ≤ 60 s 的连续点算一次过境，过境值取中位数，至少 MIN_POINTS 个点
     （1 Hz 沿轨点约 7 km 一个，单点噪声大，取中位数是常用做法）；
  3. ERA5（站点格点逐时值）线性插值到过境时刻。注意 ERA5 取的是站点格点，而过境点平均距站点约 30 km，
     这部分空间差异会计入误差，因此误差略偏大于点对点比较。
输出：results/era5_vs_altimeter/<ID>_metrics.csv、<ID>_passes.csv、<ID>_hs.png
用法：python compare_era5_altimeter.py [A B C ...]
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

from compare_era5_buoy import stats

ROOT = os.path.dirname(os.path.abspath(__file__))
SITES = json.load(open(os.path.join(ROOT, 'metadata', 'sites.json'), encoding='utf-8'))['sites']
OUT = os.path.join(ROOT, 'results', 'era5_vs_altimeter')
RADIUS_KM = 55
MIN_POINTS = 3


def haversine(lat1, lon1, lat2, lon2):
    p = np.radians
    a = np.sin(p(lat2 - lat1) / 2) ** 2 + np.cos(p(lat1)) * np.cos(p(lat2)) * np.sin(p(lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def passes(site):
    p = os.path.join(ROOT, 'processed', 'altimeter', f"{site['id']}_cci_l3.csv")
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p)
    d['time'] = pd.to_datetime(d['time'].str.replace('Z', ''))
    lat, lon = site['point']
    d['dist_km'] = haversine(lat, lon, d['latitude'], d['longitude'])
    d = d[(d['dist_km'] <= RADIUS_KM) & d['swh_adjusted'].between(0.05, 25)].sort_values(['platform_id', 'time'])
    new = (d['platform_id'] != d['platform_id'].shift()) | (d['time'].diff() > pd.Timedelta('60s'))
    d['pass_id'] = new.cumsum()
    g = d.groupby('pass_id').agg(time=('time', 'mean'), platform=('platform_id', 'first'), n=('swh_adjusted', 'size'),
                                 hs_alt=('swh_adjusted', 'median'), hs_alt_sd=('swh_adjusted', 'std'),
                                 dist_km=('dist_km', 'mean'))
    return g[g['n'] >= MIN_POINTS].reset_index(drop=True)


def interp_to(series, times):
    """逐时序列线性插值到任意时刻。"""
    s = series.dropna()
    x = s.index.values.astype('datetime64[s]').astype(np.int64)
    return np.interp(times.values.astype('datetime64[s]').astype(np.int64), x, s.values, left=np.nan, right=np.nan)


def run(site):
    ps = passes(site)
    if ps is None or ps.empty:
        print(f"[跳过] {site['id']}：没有高度计资料")
        return
    era = pd.read_csv(os.path.join(ROOT, 'processed', f"site_{site['id']}", 'observations_hourly.csv'),
                      usecols=['valid_time', 'hs', 'label_tc_distance_km'])
    era.index = pd.to_datetime(era['valid_time'].str.replace('Z', ''))
    ps['hs_era5'] = interp_to(era['hs'], ps['time'])
    ps['tc_km'] = era['label_tc_distance_km'].reindex(ps['time'].dt.round('1h')).values
    buoy = glob.glob(os.path.join(ROOT, 'processed', 'buoy', f"{site['id']}_*.csv"))
    if buoy:                                  # 有浮标：高度计 vs 浮标（±30 分钟平均）作为旁证
        b = pd.read_csv(buoy[0], usecols=['valid_time', 'hs'])
        bs = pd.Series(b['hs'].values, index=pd.to_datetime(b['valid_time'].str.replace('Z', ''))).dropna()
        bs = bs.groupby(bs.index.round('1h')).mean()
        ps['hs_buoy'] = bs.reindex(ps['time'].dt.round('1h')).values
    j = ps.dropna(subset=['hs_era5'])
    o, m = j['hs_alt'], j['hs_era5']
    rows = [{'group': '总体', **stats(o, m)}]
    for lo, hi in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 99)]:
        k = j[(o >= lo) & (o < hi)]
        if len(k) > 20:
            rows.append({'group': f'高度计 Hs {lo}–{hi if hi < 99 else "以上"} m', **stats(k['hs_alt'], k['hs_era5'])})
    for name, months in (('冬季 11–3 月', [11, 12, 1, 2, 3]), ('夏季 6–9 月', [6, 7, 8, 9]), ('转换期 4–5、10 月', [4, 5, 10])):
        k = j[j['time'].dt.month.isin(months)]
        rows.append({'group': name, **stats(k['hs_alt'], k['hs_era5'])})
    for name, cond in (('台风 300 km 内', j['tc_km'] <= 300), ('台风 300–800 km', (j['tc_km'] > 300) & (j['tc_km'] <= 800)),
                       ('无台风（>800 km）', j['tc_km'] > 800)):
        k = j[cond]
        if len(k) > 20:
            rows.append({'group': name, **stats(k['hs_alt'], k['hs_era5'])})
    if 'hs_buoy' in j:
        k = j.dropna(subset=['hs_buoy'])
        if len(k) > 20:
            rows.append({'group': '旁证：浮标 vs 高度计（浮标作真值）', **stats(k['hs_buoy'], k['hs_alt'])})
            rows.append({'group': '旁证：同批过境 浮标 vs ERA5', **stats(k['hs_buoy'], k['hs_era5'])})
    res = pd.DataFrame(rows).round(3)
    os.makedirs(OUT, exist_ok=True)
    res.to_csv(os.path.join(OUT, f"{site['id']}_metrics.csv"), index=False, encoding='utf-8-sig')
    ps.round(4).to_csv(os.path.join(OUT, f"{site['id']}_passes.csv"), index=False)
    print(f"\n[{site['id']}] {site['label']}：{len(j)} 次过境（{j['time'].min():%Y-%m} – {j['time'].max():%Y-%m}），"
          f"卫星 {j['platform'].nunique()} 颗，平均距站点 {j['dist_km'].mean():.0f} km；真值 = 高度计 swh_adjusted")
    print(res[['group', 'n', 'obs_mean', 'bias', 'rmse', 'si', 'corr']].to_string(index=False))
    plot(site, j)


def plot(site, j):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(figsize=(6, 5.5))
    tc = j['tc_km'] <= 300
    ax.scatter(j.loc[~tc, 'hs_alt'], j.loc[~tc, 'hs_era5'], s=6, alpha=0.4, color='#1C6E7D', label='其他')
    ax.scatter(j.loc[tc, 'hs_alt'], j.loc[tc, 'hs_era5'], s=14, color='#D9822B', label='台风 300 km 内')
    m = max(j['hs_alt'].max(), j['hs_era5'].max())
    ax.plot([0, m], [0, m], 'k--', lw=1)
    ax.set_xlabel('卫星高度计 Hs (m)'); ax.set_ylabel('ERA5 Hs (m)'); ax.legend()
    ax.set_title(f"{site['id']} ERA5 vs 高度计（每点一次过境）")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, f"{site['id']}_hs.png"), dpi=130); plt.close(fig)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    sel = set(sys.argv[1:])
    for s in sorted(SITES, key=lambda s: s['priority']):
        if not sel or s['id'] in sel:
            run(s)
