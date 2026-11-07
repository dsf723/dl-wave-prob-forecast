"""GEFS 海浪回算（真实历史集合预报）与浮标实测对比：原始集合在各预见期的技巧。

输入：processed/gefs_reforecast/<站号>/<年>.csv.gz（download_gefs_reforecast.py 生成）
      processed/buoy/<站点ID>_<站号>.csv（浮标，含台风标签）
对应：G ↔ 52200，H ↔ 51207，K ↔ 22107（K 的浮标实测需先从韩国气象厅下载并整理成同样格式）
指标（按预见天 1–35 分别统计，Hs）：
  集合平均的偏差、RMSE；集合离散度（成员标准差）均值；离散度/RMSE 比（<1 表示集合过于自信）；
  集合 CRPS（经验分布：mean|x−y| − ½ mean|x−x′|）；P10–P90 覆盖率（理想 80%）。
  另按台风距离分组（≤300 km、>800 km）。这是论文中"原始集合"基线，也是伪预报误差参数的标定依据。
输出：results/reforecast_vs_buoy/<ID>_by_lead.csv、<ID>_by_lead.png
用法：python compare_reforecast_buoy.py [G H K]
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
SITES = json.load(open(os.path.join(ROOT, 'metadata', 'sites.json'), encoding='utf-8'))['sites']
OUT = os.path.join(ROOT, 'results', 'reforecast_vs_buoy')


def load_reforecast(station):
    files = sorted(glob.glob(os.path.join(ROOT, 'processed', 'gefs_reforecast', station, '*.csv.gz')))
    if not files:
        return None
    df = pd.concat(pd.read_csv(f, usecols=['valid_time', 'issue_time', 'member', 'hs']) for f in files)
    df['valid_time'] = pd.to_datetime(df['valid_time'], format='ISO8601').dt.round('1min')   # 分段读取模式写的时间带纳秒小数
    df['issue_time'] = pd.to_datetime(df['issue_time'], format='ISO8601')
    return df.drop_duplicates(['issue_time', 'valid_time', 'member'])


def load_buoy(site):
    p = glob.glob(os.path.join(ROOT, 'processed', 'buoy', f"{site['id']}_*.csv"))
    if not p:
        return None, None
    b = pd.read_csv(p[0], low_memory=False)
    t = pd.to_datetime(b['valid_time'].str.replace('Z', ''))
    hs = pd.Series(b['hs'].values, index=t).dropna()
    hs = hs.groupby(hs.index.round('1h')).mean()
    tc = pd.Series(b['label_tc_distance_km'].values, index=t) if 'label_tc_distance_km' in b else None
    tc = tc.groupby(tc.index.round('1h')).min() if tc is not None else None
    return hs, tc


def crps_ens(x, y):
    """x: [n, m] 成员；y: [n]。经验分布 CRPS。"""
    t1 = np.mean(np.abs(x - y[:, None]), axis=1)
    xs = np.sort(x, axis=1)
    m = x.shape[1]
    w = 2 * np.arange(1, m + 1) - m - 1                       # Σ_i Σ_j |xi − xj| = 2 Σ_k (2k − m − 1) x(k)
    t2 = (xs * w).sum(axis=1) * 2 / (m * m)
    return t1 - 0.5 * t2


def evaluate(fc, obs, tc):
    wide = fc.pivot_table(index=['issue_time', 'valid_time'], columns='member', values='hs')
    wide = wide.dropna(thresh=max(3, int(0.8 * wide.shape[1])))
    vt = wide.index.get_level_values('valid_time')
    y = obs.reindex(vt).values
    ok = np.isfinite(y)
    wide, y, vt = wide[ok], y[ok], vt[ok]
    x = wide.to_numpy()
    lead = (vt - wide.index.get_level_values('issue_time')) / pd.Timedelta('1D')
    rows = pd.DataFrame({'lead_day': np.ceil(lead).astype(int), 'y': y, 'mean': np.nanmean(x, 1), 'sd': np.nanstd(x, 1, ddof=1),
                         'crps': crps_ens(np.where(np.isfinite(x), x, np.nanmean(x, 1)[:, None]), y),
                         'p10': np.nanquantile(x, 0.1, axis=1), 'p90': np.nanquantile(x, 0.9, axis=1)})
    rows['tc_km'] = tc.reindex(vt).values if tc is not None else np.nan
    return rows


def summarize(r):
    e = r['mean'] - r['y']
    rmse = np.sqrt((e ** 2).mean())
    return pd.Series({'n': len(r), 'bias': e.mean(), 'rmse': rmse, 'spread': r['sd'].mean(),
                      'spread_skill': r['sd'].mean() / rmse if rmse > 0 else np.nan, 'crps': r['crps'].mean(),
                      'cover_p10_p90': ((r['y'] >= r['p10']) & (r['y'] <= r['p90'])).mean()})


def run(site, station):
    fc = load_reforecast(station)
    obs, tc = load_buoy(site)
    if fc is None or obs is None:
        print(f"[跳过] {site['id']}：缺回算（{station}）或浮标资料")
        return
    r = evaluate(fc, obs, tc)
    if r.empty:
        print(f"[跳过] {site['id']}：回算与浮标没有重叠时段（回算 {fc.issue_time.min():%Y}–{fc.issue_time.max():%Y}，"
              f"浮标 {obs.index.min():%Y}–{obs.index.max():%Y}）")
        return
    out = r.groupby('lead_day')[['y', 'mean', 'sd', 'crps', 'p10', 'p90']].apply(summarize).reset_index()
    out['group'] = '全部'
    parts = [out]
    for name, cond in (('台风 300 km 内', r['tc_km'] <= 300), ('无台风（>800 km）', r['tc_km'] > 800)):
        if cond.sum() > 100:
            g = r[cond].groupby('lead_day')[['y', 'mean', 'sd', 'crps', 'p10', 'p90']].apply(summarize).reset_index()
            g['group'] = name
            parts.append(g)
    res = pd.concat(parts).round(4)
    os.makedirs(OUT, exist_ok=True)
    res.to_csv(os.path.join(OUT, f"{site['id']}_by_lead.csv"), index=False, encoding='utf-8-sig')
    print(f"[{site['id']}] NDBC/站号 {station}：{r['y'].size} 个（起报×时刻）配对")
    print(res[res.group == '全部'][['lead_day', 'n', 'bias', 'rmse', 'spread', 'spread_skill', 'crps', 'cover_p10_p90']].to_string(index=False))
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        for g, d in res.groupby('group'):
            ax[0].plot(d['lead_day'], d['crps'], marker='o', ms=3, label=g)
            ax[1].plot(d['lead_day'], d['spread_skill'], marker='o', ms=3, label=g)
        ax[0].set_xlabel('预见天'); ax[0].set_ylabel('CRPS (m)'); ax[0].legend(); ax[0].set_title(f"{site['id']} 原始集合 CRPS")
        ax[1].axhline(1, color='k', lw=1, ls='--'); ax[1].set_xlabel('预见天'); ax[1].set_ylabel('离散度 / RMSE')
        ax[1].set_title('离散度—误差比（<1：过于自信）')
        fig.tight_layout(); fig.savefig(os.path.join(OUT, f"{site['id']}_by_lead.png"), dpi=130); plt.close(fig)
    except Exception as e:
        print('画图失败：', e)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    sel = set(sys.argv[1:])
    for s in SITES:
        st = (s.get('buoy') or {}).get('station')
        if st and st.isdigit() and (not sel or s['id'] in sel):
            run(s, st)
